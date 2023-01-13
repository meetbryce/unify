import time
from datetime import datetime, timedelta
import json
import logging
import math
import os
from threading import Thread
import threading
import queue
import subprocess
from pprint import pprint
import re
import sys
import traceback
import typing
from typing import Dict
import psutil
import uuid
import warnings

import sqlglot
import pandas as pd
import pyarrow as pa
import numpy as np

# Background tasks
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.key_binding import KeyBindings

from sqlalchemy.orm.session import Session

from unify.rest_adapter import (
    Adapter, 
    RESTView,
    TableDef,
    TableUpdater,
)
from unify.db_wrapper import (
    ConnectionScan,
    ColumnInfo,
    dbmgr,
    DBManager, 
    DBSignals,
    TableMissingException,
    TableHandle,
)
from unify.sqla_storage_manager import UnifyDBStorageManager

from unify.adapters import Connection, OutputLogger
from unify.file_adapter import LocalFileAdapter
from unify.zmq_logging import ZeroMQSocketHandler, ZeroMQSocketListener

logger: logging.Logger = logging.getLogger('unify')

def add_logging_handler(handler):
    logger.addHandler(handler)

class LoaderJob:
    ACTION_LOAD_TABLE = "load"
    ACTION_REFRESH_TABLE = "refresh"
    ACTION_RELOAD_TABLE = "reload"
    ACTION_CANCEL = "cancel"
    ACTION_STOP_DAEMON = "stop_daemon"

    def __init__(self, tenant_id: str, action: str, table: TableHandle = None, parameters: dict = {}):
        self._job_id = str(uuid.uuid4()) 
        self._tenant_id = tenant_id
        self._action = action
        if table:
            self._schema = table.schema()
            self._table_root = table.table_root()
        else:
            self._schema = None
            self._table_root = None
        self._parameters = parameters
        self._rows_loaded = 0
        self._canceled = False
        self._progress_bar: ProgressBar = None
        self._backgrounded = False

    @staticmethod
    def load_table_job(schema: str, table_root: str):
        return LoaderJob(None, LoaderJob.ACTION_LOAD_TABLE, TableHandle(table_root, schema))

    @property
    def tenant_id(self):
        return self._tenant_id

    @property
    def action(self):
        return self._action

    @property
    def job_id(self):
        return self._job_id

    @property
    def table(self):
        if self._table_root and self._schema:
            return TableHandle(self._table_root, self._schema)
        else:
            return None

    @property
    def parameters(self):
        return self._parameters

    @property 
    def canceled(self):
        return self._canceled

    def background(self):
        if self._progress_bar:
            self._progress_bar.counters[0].done = True
            self._progress_bar.app.exit()
        self._backgrounded = True

    @property
    def progress_bar(self) -> ProgressBar:
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, pb):
        self._progress_bar = pb

    def set_progress_title(self, title: str):
        if self._progress_bar:
            self._progress_bar.title = title

    def set_progress_count(self, count):
        if self._progress_bar and len(self._progress_bar.counters) > 0:
            if count == (self._progress_bar.counters[0].total+1):
                self._progress_bar.counters[0].total = self._progress_bar.counters[0].total*10

    def get_progress_iter(self):
        if self._progress_bar:
            return self._progress_bar
        else:
            def noop(gen, *args, **kwargs):
                for i in gen:
                    yield i
            return noop

    def __eq__(self, other):
        return self._job_id == other._job_id

    def __repr__(self) -> str:
        return str(self.__dict__)

    def incr_row_count(self, count):
        self._rows_loaded += count

    def cancel(self):
        self._canceled = True


class BaseTableScan(Thread):
    """
        Thread class for querying a REST API, by pages, and storing the results
        into the local database. This class holds common logic across inital table
        load plus table update operations.
    """
    def __init__(self, tableMgr=None, tableLoader=None, select: list = None, strip_prefixes: list = None, job: LoaderJob=None):
        Thread.__init__(self)
        self.tableMgr: TableMgr = tableMgr
        self.tableLoader = tableLoader
        self.table_ref = tableMgr.schema + "." + tableMgr.table_spec.name
        self.table_handle = TableHandle(tableMgr.table_spec.name, tableMgr.schema)
        self.select = select
        # FIXME: We should generalize a pattern for adapters to transform table results rather than
        # passing specific parameters like this
        self.strip_prefixes = strip_prefixes
        self.loader_job: LoaderJob = job

    def cleanup_df_page(self, df, cols_to_drop=[]) -> pd.DataFrame:
        # Performs a set of cleaning/coercion operations before we write a set of rows
        # to the database.
        #
        # 1. Coerce infered "mixed-" typed cols to str
        # 2. Parse columns that "look like" datetimes as datetimes
        # 3. Drop embedded lists of complex objects (need to implement serializing these as json)
        # 4. Strip unhelpful prefixes from column names
        #
        # Returns the DataFrame which should be written to the database

        for column in df.columns:
            #
            # 'mixed-' types indicates pandas found multiple types. We should logic
            # to corece to the "best" type, but for now we just convert to string
            #
            if pd.api.types.infer_dtype(df[column]).startswith("mixed"):
                df[column] = df[column].apply(lambda x: str(x))
            
            # 
            # datetime parsing
            # We look for things that look like dates and try to parse them into datetime types
            #

            testcol = df[column].dropna(axis='index')
            testvals = testcol.sample(min(10,testcol.size)).values
            detected = 0
            # See if any values look like dates
            for testval in testvals:
                if not hasattr(testval, 'count'):
                    continue
                tlen = len(testval)
                if tlen > 6 and tlen < 34 and \
                    (testval.count("/") == 2 or testval.count(":") >= 2 or testval.count("-") >= 2) and \
                    sum(c.isdigit() for c in testval) >= (tlen/2):
                        detected += 1

            if detected > (len(testvals) / 2): # more than 50% look like dates
                try:
                    #print("Corecing date column: ", column)
                    df[column] = pd.to_datetime(df[column], errors="coerce")
                    df[column].replace({np.nan: None}, inplace = True)
                except:
                    pass

        schema = pa.Schema.from_pandas(df)

        rename_col_dict = {}

        for col in schema.names:
            f = schema.field(col)
            # Drop embedded lists of objects. We should serialize to json instead
            if pa.types.is_list(f.type) and \
                (f.type.value_type == pa.null() or not pa.types.is_primitive(f.type.value_type)):
                # Remove the column
                df.drop(columns=f.name, inplace=True)
                continue

            if f.name in cols_to_drop:
                df.drop(columns=f.name, inplace=True)

            # 
            # apply 'select' table configuration property
            #
            # Caller can optionally specify a set of columns to keep. We keep any column with
            # either an exact match, or a parent property match ("name_??").
            if self.select:
                if not (f.name in self.select or \
                    any(sname for sname in self.select if \
                        (re.match(sname + r"_.*", f.name) or re.match(sname, f.name)))):
                    #print("Dropping column missing from 'select' clause: ", f.name)
                    df.drop(columns=f.name, inplace=True)
            #
            # Strip annonying prefixes from column names
            #
            if self.strip_prefixes:
                for prefix in self.strip_prefixes:
                    if f.name.startswith(prefix):
                        newc = f.name[len(prefix):]
                        rename_col_dict[f.name] = newc

        if rename_col_dict:
            # Return a new DF with renamed cols. This lets us keep using the original as a column filter
            # for subsequent pages
            return df.rename(columns=rename_col_dict)
        else:
            return df


    def _set_duck(self, duck):
        self.session = Session(bind=duck.engine)

    def run(self):
        with dbmgr() as duck:
            self.session = Session(bind=duck.engine)
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                try:
                    self.perform_scan(duck)
                except ValueError as e:
                    if 'list.remove' in str(e):
                        pass # ignore error from ProgressBar being closed
                    else:
                        raise
            self.session.close()

    def save_scan_record(self, values: dict):
        logger.debug("*SAVING SCAN RECORD: {}".format(values))

        self.session.add(
            ConnectionScan(
                table_name=self.table_handle.table_root(), 
                table_schema=self.table_handle.schema(),
                connection=self.tableMgr.schema,
                values=values
            )
        )
        self.session.commit()

    def get_last_scan_records(self, limit = 3):
        scans = self.session.query(ConnectionScan).filter(
            ConnectionScan.table_name==self.table_handle.table_root(),
            ConnectionScan.table_schema==self.table_handle.schema()
        ).order_by(ConnectionScan.created.desc())[0:limit]
        return [scan.values for scan in scans]

    def clear_scan_record(self):
        self.session.query(ConnectionScan).filter(
            ConnectionScan.table_name==self.table_handle.table_root(),
            ConnectionScan.table_schema==self.table_handle.schema()
        ).delete()
        self.session.commit()

    def create_table_with_first_page(self, duck: DBManager, next_df, schema, table_root):
        # before writing the DF to the database, we do some cleaning on columns and types.
        next_df = self.cleanup_df_page(next_df)

        source = self.tableMgr.table_spec.get_table_source()
        if source is not None and not isinstance(source, str):
            source = json.dumps(source)

        table = TableHandle(table_root, schema, table_opts={
                'connection': self.tableMgr.schema,
                'description': self.tableMgr.table_spec.description,
                'source': source
        })
        if duck.table_exists(table):
            # table already exists, so assume we are updating
            duck.append_dataframe_to_table(next_df, table)
        else:
            duck.write_dataframe_as_table(next_df, table)

    def get_query_mgr(self):
        return self.tableMgr.table_spec

    def perform_scan(self, duck: DBManager):
        progress_iter = self.loader_job.get_progress_iter()
        self.loader_job.set_progress_title(f"Loading data for table: {self.tableMgr.name}")

        scan_start = time.time()
        self.save_scan_record({"scan_start": scan_start})

        page = 1
        page_flush_count = 5 # flush 5 REST calls worth of data to the db
        row_buffer_df = None
        table_cols = set()

        resource_query_mgr: TableDef = self.get_query_mgr()

        for query_result in progress_iter(resource_query_mgr.query_resource(self.tableLoader, logger), total=10, remove_when_done=True):
            json_page = query_result.json
            size_return = query_result.size_return
            record_path = self.tableMgr.table_spec.result_body_path
            if record_path is not None:
                record_path = record_path.split(".")
            metas = None
            if self.tableMgr.table_spec.result_meta_paths:
                metas = self.tableMgr.table_spec.result_meta_paths
                if not isinstance(metas, list):
                    metas = [metas]
                metas = [p.split(".") for p in metas]
            if isinstance(json_page, pd.DataFrame):
                df = json_page
            elif self.tableMgr.table_spec.result_object_path:
                df = pd.DataFrame(json_page.get(self.tableMgr.table_spec.result_object_path))
            else:
                df = pd.json_normalize(
                    json_page, 
                    record_path=record_path, 
                    meta=metas,
                    sep='_')
            # adapters can provide extra data to merge into the result table. See
            # the `copy_params_to_output` property.
            if query_result.merge_cols:
                for col, val in query_result.merge_cols.items():
                    df[col] = val
            size_return.append(df.shape[0])
            if self.loader_job is not None:
                self.loader_job.incr_row_count(df.shape[0])

            if df.empty:
                continue

            if page == 1:
                row_buffer_df = df
            else:
                if table_cols:
                    # Once we set the table columns from the first page, then we enforce that list
                    usable = list(table_cols.intersection(df.columns.tolist()))
                    #print("Coercing to usable cols: ", usable)
                    df = df[usable]
                row_buffer_df = pd.concat([row_buffer_df, df], axis='index', ignore_index=True)

            # Flush rows
            if (page % page_flush_count) == 0:
                self._flush_rows_to_db_catch_error(duck, self.tableMgr, row_buffer_df, page, page_flush_count)
                if page == page_flush_count:
                    table_cols = set(row_buffer_df.columns.tolist())
                row_buffer_df = row_buffer_df[0:0] # clear flushed rows, but keep columns

            page += 1
            self.loader_job.set_progress_count(page)

        if row_buffer_df is not None and row_buffer_df.shape[0] > 0:
            self._flush_rows_to_db_catch_error(duck, self.tableMgr, row_buffer_df, page, page_flush_count)

        # We save at the end in case we encounter an error
        self.save_scan_record({"scan_complete": scan_start})
        logger.debug("Finished table scan for: %s", self.tableMgr.name)
        duck.send_signal(signal=DBSignals.TABLE_LOADED, table=self.table_handle)

    def _flush_rows_to_db_catch_error(self, duck: DBManager, tableMgr, next_df, page, flush_count):
        try:
            self._flush_rows_to_db(duck, tableMgr, next_df, page, flush_count)
            
        except Exception as e:
            # "Core dump" the bad page
            core_file = f"/tmp/{self.tableMgr.name}_bad_page_{page}.csv"
            logger.critical("Error saving page to db, dumping to file: %s", core_file)
            with open(core_file, "w") as f:
                next_df.to_csv(f)
            raise

class InitialTableLoad(BaseTableScan):
    def _flush_rows_to_db(self, duck: DBManager, tableMgr, next_df, page, flush_count):
        ## Default version works when loading a new table
        if page <= flush_count:
            # First set of pages, so create the table
            self.create_table_with_first_page(duck, next_df, tableMgr.schema, tableMgr.table_spec.name)
        else:
            next_df = self.cleanup_df_page(next_df)
            msg = f"Saved {self.loader_job._rows_loaded} rows ({next_df.shape[1]} data columns)"
            if self.loader_job and self.loader_job.progress_bar:
                self.loader_job.progress_bar.title = msg
            logger.info(msg)
            duck.append_dataframe_to_table(
                next_df, 
                TableHandle(tableMgr.table_spec.name, tableMgr.schema)
            )

class TableUpdateScan(BaseTableScan):
    """
       Specialization of BaseTableScan for updating an existing table. Requests
       the "updater" object to use as the query mgr from the table spec. This
       object will implement the strategy to query updates from the source system 
       per the specification for the table.
       We pass the last scan time to the updater for its use (depending on its strategy).

       Finally, if the updater strategy indicates the results `should_replace` the existing
       table, then we download results into a temp table and swap it in for the old table when done.
    """
    def perform_scan(self, duck):
        # Grab timestamp threshold from last scan
        scan_records = self.get_last_scan_records()
        print("Scans: ", scan_records)
        # we are looking for the most recent scan_start. 
        last_scan_time = None
        try:
            start_rec = next(record for record in scan_records if 'scan_start' in record)
            slop = timedelta(hours=24)
            last_scan_time = datetime.utcfromtimestamp(start_rec['scan_start']) - slop
        except StopIteration:
            # Failed to find a start time
            pass
        self._query_mgr: TableUpdater = self.tableMgr.table_spec.get_table_updater(last_scan_time)
        self._target_table = self.tableMgr.name

        if self._query_mgr.should_replace():
            # Updater will replace the existing table rather than appending to it. So
            # download data into a temp file
            self._target_table_root = self.tableMgr.table_spec.name + "__temp"
            self._target_table = self.tableMgr.schema + "." + self._target_table_root
            # drop the temp table in case it's lying around
            duck.execute(f"DROP TABLE IF EXISTS {self._target_table}")
            # Use parent class to download data from the target API (calling _flush_rows along the way)
            super().perform_scan(duck)
            # Now swap the new table into place
            duck.replace_table(TableHandle(self._target_table), TableHandle(self.tableMgr.name))
        else:
            self._target_table_root = self.tableMgr.table_spec.name
            self._target_table = self.tableMgr.schema + "." + self._target_table_root
            super().perform_scan(duck)

    def get_query_mgr(self):
        return self._query_mgr

    def _flush_rows_to_db(self, duck: DBManager, tableMgr, next_df, page, flush_count):
        if self._query_mgr.should_replace():
            # Loading into a temp table so just do simple load
            if page <= flush_count:
                # First set of pages, so create the table
                self.create_table_with_first_page(
                    duck, 
                    next_df, 
                    tableMgr.schema, 
                    self._target_table_root
                )
            else:
                next_df = self.cleanup_df_page(next_df)
                duck.append_dataframe_to_table(
                    next_df, 
                    TableHandle(self._target_table_root, self.tableMgr.schema)
                )
            return

        # To update a table we have to both remove existing copies of any rows
        # we downloaded and align our downloaded columns with the existing table.
        cols = duck.get_table_columns(TableHandle(self._target_table))
        # filter and order to the right columns
        next_df = next_df[cols]
        next_df = self.cleanup_df_page(next_df)
        print(f"Saving update page {page} with {next_df.shape[0]} rows and {next_df.shape[1]} columns")

        # First delete any existing records
        keys = next_df[tableMgr.table_spec.key]  #.values.tolist()
        keys_table = duck.create_memory_table("__keys", pd.DataFrame(keys))

        duck.delete_rows(TableHandle(self._target_table), 
            where_clause = f"{tableMgr.table_spec.key} IN (SELECT * FROM {keys_table})")
        duck.drop_memory_table("__keys")

        # Now append the new records             
        duck.append_dataframe_to_table(next_df, TableHandle(self._target_table_root, tableMgr.schema))


class TableExporter(Thread):
    def __init__(
        self, query: str=None, 
        table: str=None, 
        adapter: Adapter=None, 
        target_file: str=None,
        allow_overwrite=False,
        allow_append=False):
        self.query = query
        self.table = table
        self.adapter: Adapter = adapter
        self.target_file = target_file
        self.overwrite = allow_overwrite
        self.append = allow_append

    def run(self, output_logger: OutputLogger):
        # Runs the query against DuckDB in chunks, and sends those chunks to the adapter
        self.output_handle = self.adapter.create_output_table(self.target_file, output_logger, overwrite=self.overwrite)

        with dbmgr() as duck:
            if self.query:
                r = duck.execute(self.query)
            else:
                r = duck.execute(f"select * from {self.table}")

            # FIXME: Use DF chunking and multiple pages
            self.adapter.write_page(self.output_handle, r, output_logger, append=self.append, page_num=1)

            self.adapter.close_output_table(self.output_handle)


TableLoader = typing.NewType("TableLoader", None)


class TableMgr:
    def __init__(self, schema, adapter, table_spec, auth = None, params={}):
        self.schema = schema
        if schema is None or table_spec.name is None:
            raise RuntimeError(f"Bad schema {schema} or missing table_spec name {table_spec}")
        self.name = schema + "." + table_spec.name
        self.adapter = adapter
        self.table_spec: TableDef = table_spec
        self.synchronous_scanning = True

    def has_data(self):
        return True

    def truncate(self, duck: DBManager):
        # Remove table from duck db
        duck.execute(f"truncate {self.name}")

    def _create_scanner(self, tableLoader: TableLoader, job: LoaderJob):
        return InitialTableLoad(
            self, 
            tableLoader=tableLoader, 
            select=self.table_spec.select_list,
            strip_prefixes=self.table_spec.strip_prefixes,
            job=job
        )

    def load_table(self, tableLoader=None, job: LoaderJob=None):
        # Spawn a thread to query the table source
        self.scan = self._create_scanner(tableLoader, job=job)
        self.scan.run()

    def refresh_table(self, tableLoader, job: LoaderJob=None):
        scanner = TableUpdateScan(self, tableLoader=tableLoader, select=self.table_spec.select_list, job=job)
        scanner.run()


class TableLoader:
    """ Master loader class. This class can load the data for any table, either by 
        querying the table from DuckDB or else by calling the underlying REST API.

        We maintain a SQLite queue and dispatch table loading requests to our background worker.

        Loading jobs can be executed synchronously or asynchronously. In either case they will
        go through the job queue, but in synchronous mode the job will be executed immediately.

        Tables can be in 4 states:
        1. Unknown (schema and table name never registered)
        2. Never queried. Call the Adapter to query the data and store it to local db.
        3. Table exists in the local db.
    """
    def __init__(self, silence_errors=False, given_connections: list[Connection]=[], leader=False):
        with dbmgr() as duck:
            try:
                if given_connections:
                    self.connections = given_connections
                else:
                    self.connections: list[Connection] = Connection.setup_connections(
                        storage_mgr_maker=lambda schema: UnifyDBStorageManager(schema, duck)
                    )
            except:
                if not silence_errors:
                    raise
                else:
                    self.connections = []
            self.tables: dict[str, TableMgr] = {}

            self.adapters: dict[str, Adapter] = dict(
                [(conn.schema_name, conn.adapter) for conn in self.connections]
            )
            schema = 'files'
            self.adapters[schema] = LocalFileAdapter(
                {'name':schema},
                root_path=os.path.join(os.environ['UNIFY_HOME'], "files"),
                storage=UnifyDBStorageManager(schema, duck),
                schema_name='files'
            )
            duck.create_schema('files')

            # Connections defines the set of schemas we will create in the database.
            # For each connection/schema then we will define the tables as defined
            # in the REST spec for the target system.
            for conn in self.connections:
                duck.create_schema(conn.schema_name)
                for t in conn.adapter.list_tables():
                    tmgr = TableMgr(conn.schema_name, conn.adapter, t)
                    self.tables[tmgr.name] = tmgr

# Loading jobs

    def run_table_load_job(self, schema, table_root, run_async=True):
        self.run_job(LoaderJob("", LoaderJob.ACTION_LOAD_TABLE, TableHandle(table_root, schema=schema)), run_async)

    def _handle_job(self, job: LoaderJob, run_async: bool=True):
        t = threading.Thread(target=self._dispatch_job, args=(job,))
        if run_async:
            t.daemon = True
            t.start()
            
            while t.is_alive():
                t.join(timeout=0.2)
                if job._backgrounded:
                    break
        else:
            t.run()

    """ Async worker does any slow TableLoader tasks """
    def _dispatch_job(self, job: LoaderJob):
        kb = KeyBindings()
        @kb.add('<any>')
        def _(event):
            print("Finishing in background...")
            job.background()

        if job.action == LoaderJob.ACTION_LOAD_TABLE and job.table:
            bottom = f"Loading {str(job.table)}. Press any key to move job to the background."
            with ProgressBar(key_bindings=kb, title=f"Loading {str(job.table)}...", bottom_toolbar=bottom) as pb:
                job.progress_bar = pb
                self.materialize_table(job=job)
        elif job.action == LoaderJob.ACTION_REFRESH_TABLE:
            self.tables[str(job.table)].refresh_table(tableLoader=self, job=job)
        elif job.action == LoaderJob.ACTION_CANCEL:
            # FIXME: allow canceling loading jobs
            pass
        else:
            raise RuntimeError("Uknown job action: " + job.action)

    def add_connection(self, adapter_name: str, schema_name: str, opts: dict):
        with dbmgr() as db:
            # Adds a connection to the loader. This is used by the REST API to add
            # new connections to the database.
            conn = Connection.create_connection(
                adapter_name, 
                schema_name, 
                opts, 
                storage_mgr_maker=lambda schema: UnifyDBStorageManager(schema_name, db)
            )
            self.connections.append(conn)
            self.adapters[schema_name] = conn.adapter
            db.create_schema(conn.schema_name)
            for t in conn.adapter.list_tables():
                tmgr = TableMgr(conn.schema_name, conn.adapter, t)
                self.tables[tmgr.name] = tmgr

    def delete_connection(self, connection):
        with dbmgr() as db:
            # Delete the connection from the database
            db.drop_schema(connection.schema_name, cascade=True)
            self.connections.remove(connection)
            del self.adapters[connection.schema_name]

        Connection.delete_connection_config(connection.schema_name)

    def _get_table_mgr(self, table):
        if table in self.tables:
            return self.tables[table]
        else:
            schema, table_root = table.split(".")
            if schema not in self.adapters:
                logger.critical(f"Adapter {schema} not found - maybe you need to restart the job daemon?")
                return
            table_spec = self.adapters[schema].lookupTable(table_root)
            tmgr = TableMgr(schema, self.adapters[schema], table_spec)
            self.tables[tmgr.name] = tmgr
            return tmgr

    def qualify_tables_in_view_query(self, query: str, from_list: list[str], schema: str,
                                    dialect: str):
        """ Replaces non-qualified table references in a view query with qualified ones.
            Returns the corrected query.
        """       
        query += " "
        for table_root in sqlglot.parse_one(query).find_all(sqlglot.exp.Table):
            table = schema + "." + table_root.name.split(".")[0]
            query = re.sub(r"[\s,]+" + table_root.name + r"[\s]+", f" {table} ", query)

        return query

    def create_views(self, schema, table):
        # (Re)create any views defined that depend on the the indicated table
        qual = schema + "." + table
        tmgr = self._get_table_mgr(qual)
        views: typing.List[RESTView] = tmgr.adapter.list_views()
        if not views:
            return
        with dbmgr() as duck:
            for view in views:
                if table in view.from_list:
                    query = None
                    if isinstance(view.query, dict):
                        if duck.dialect() in view.query:
                            query = view.query[duck.dialect()]
                        else:
                            print(f"Skipping view {view.name} with no dialect for current database")
                            return
                    else:
                        query = view.query
                    query = self.qualify_tables_in_view_query(query, view.from_list, tmgr.schema, duck.dialect())
                    duck.execute(f"DROP VIEW IF EXISTS {tmgr.schema}.{view.name}")
                    duck.execute(f"CREATE VIEW {tmgr.schema}.{view.name} AS {query}")


    def refresh_table(self, table_ref, run_async=True):
        self.run_job(LoaderJob("", LoaderJob.ACTION_REFRESH_TABLE, TableHandle(table_ref)), run_async=run_async)

    def query_table(self, schema: str, query: str):
        def transformer(node):
            if isinstance(node, sqlglot.exp.Table):
                if "." in str(node):
                    raise RuntimeError(f"Adapter queries cannot use qualified table names: {node}. Query: {query}")
                table = schema + "." + str(node)
                if not self.table_exists_in_db(table):
                    tmgr = self.tables[table]
                    # TODO: Might want a timeout for large parent tables
                    tmgr.load_table(tableLoader=self)
                return sqlglot.parse_one(schema + "." + str(node))    
            return node

        newq = sqlglot.parse_one(query, read='clickhouse').transform(transformer)
        with dbmgr() as db:
            parent_query = newq.sql(dialect=db.dialect())
            yield db.execute(parent_query)

    def read_table_rows(self, table, limit=None):
        with dbmgr() as duck:
            if not self.table_exists_in_db(table):
                tmgr = self.tables[table]
                tmgr.load_table(tableLoader=self)

            if self.table_exists_in_db(table):
                # FIXME: Use chunking and multiple pages
                if limit is not None:
                    limit = f" limit {limit}"
                else:
                    limit = ""
                yield duck.execute(f"select * from {table} {limit}")
            else:
                raise RuntimeError(f"Could not get rows for table {table}")

    def lookup_connection(self, name):
        return next(c for c in self.connections if c.schema_name == name)

    def truncate_table(self, table):
        if table in self.tables:
            with dbmgr() as duck:
                self.tables[table].truncate(duck)

    def drop_table(self, table: TableHandle):
        if isinstance(table, str):
            table = TableHandle(table)
        with dbmgr() as db:
            db.drop_table(table)

    def table_exists_in_db(self, table):
        try:
            # FIXME: memoize the results here
            with dbmgr() as duck:
                duck.execute(f"select 1 from {table}")
            return True
        except TableMissingException:
            return False

    def materialize_table(self, job: LoaderJob):
        with dbmgr() as duck:
            qual = str(job.table)
            try:
                tmgr = self._get_table_mgr(qual)
            except KeyError:
                raise TableMissingException(qual)
            tmgr.load_table(tableLoader=self, job=job)
            self.create_views(job.table.schema(), job.table.table_root())
            return tmgr.has_data()
##
### Refresh daemon
##
    def setup_task_daemon(self):
        # The handler will actually print the log messages
        #handler = logging.StreamHandler()
        #logger.addHandler(handler)

        # The listener runs the dispatch thread, and passes events to the handler. We
        # don't pass the queue as the listener will make a ZMQ internally.
        listener = ZeroMQSocketListener(*logger.handlers)
        listener.start()

        if not self.task_daemon_is_running():
            self.start_daemon()

    def get_pid_path(self):
        return os.path.join(os.path.dirname(__file__), "unify_daemon.pid")

    def task_daemon_is_running(self):
        pid_file = self.get_pid_path()
        if os.path.exists(pid_file):
            pid = int(open(pid_file).read())
            try:
                process = psutil.Process(pid)
                if process.status() == 'running':
                    return True
            except:
                pass
        return False

    def start_daemon(self):
        # Exec the task daemon
        logger.debug("Starting the task daemon...")
        p = subprocess.Popen(["python", __file__], start_new_session=True)

    def stop_daemon(self):
        if self.task_daemon_is_running():
            # FIXME: just send SIGINT to the daemon
            while True:
                if not self.task_daemon_is_running():
                    break
                time.sleep(1)


    def run_job(self, job: LoaderJob, run_async=True):
        """ Runs the loading job as specified. Returns True if the job was executed
            or False if the job will run asynchronously.
        """
        logger.debug(f"Running job: {job}")
        self._handle_job(job, run_async)
        return True


if __name__ == "__main__":
    import pdb_attach
    pdb_attach.listen(50000) 

    logger.setLevel(logging.DEBUG)
    logger.debug("Starting Unify worker...")
    worker: TableLoader = TableLoader()
    worker.job_loop()
