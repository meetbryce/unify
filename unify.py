from email.parser import Parser
from inspect import isfunction
import time
from datetime import datetime
import io
import json
import os
from threading import Thread
import pickle
import pydoc
import re
import sys
import traceback
import typing
import lark
from lark import Lark, Visitor
from lark.visitors import v_args
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from typing import Dict
# CHARTING
import matplotlib.pyplot as plt

import pandas as pd
import pyarrow as pa
import numpy as np

from timeit import default_timer as timer

# DuckDB
import duckdb
from setuptools import Command

from rest_schema import (
    Adapter, 
    Connection, 
    OutputLogger, 
    TableDef,
    TableUpdater,
    UnifyLogger
)

from storage_manager import StorageManager
from schemata import LoadTableRequest, Queries
from parsing_utils import (
    find_subtree, 
    find_node_return_child, 
    find_node_return_children,
    collect_child_strings,
    collect_child_text,
    collect_strings
)

DATA_HOME = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_HOME, exist_ok=True)


class DuckContext:
    DUCK_CONN = None

    """ A cheap hack around DuckDB only usable in a single process. We just open/close
        each time to manage. Context manager for accessing the DuckDB database """
    def __init__(self):
        pass

    def __enter__(self):
        if self.__class__.DUCK_CONN is None:
            self.__class__.DUCK_CONN = duckdb.connect(os.path.join(DATA_HOME, "duckdata"), read_only=False)
            self.__class__.DUCK_CONN.execute("PRAGMA log_query_path='/tmp/duckdb_log'")
        return self.__class__.DUCK_CONN

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__class__.DUCK_CONN is not None:
            try:
                self.__class__.DUCK_CONN.execute("COMMIT")
            except:
                pass
            self.__class__.DUCK_CONN.close()
            self.__class__.DUCK_CONN = None
        

class SimpleLogger(UnifyLogger):
    def __init__(self, adapter: Adapter):
        self.adapter = adapter

    def log_table(self, table: str, level: int, *args):
        print(f"[{str(self.adapter)}: {table}] ", *args, file=sys.stderr)

class BaseTableScan(Thread):
    """
        Thread class for querying a REST API, by pages, and storing the results
        into the local database. This class holds common logic across inital table
        load plus table update operations.
    """
    def __init__(self, tableMgr=None, tableLoader=None, select: list = None):
        Thread.__init__(self)
        self.tableMgr: TableMgr = tableMgr
        self.tableLoader = tableLoader
        self.select = select
        self.storage_mgr: DuckdbStorageManager = None

    def cleanup_df_page(self, df, cols_to_drop=[]):
        # Pandas json_normalize does not have a clever way to normalize embedded lists
        # of records, so it just returns a list of dicts. These will cause deserialization
        # issues if different REST response pages have different results for these
        # embedded lists. To work around that for now we are just removing those columns
        # from our results.
        # TODO: Convert the structured element into a JSON field

        # It's easy to get columns containing mixed types, but this won't work well
        # when we convert to a table. So we look for those and force convert them to
        # strings

        for column in df.columns:
            if pd.api.types.infer_dtype(df[column]).startswith("mixed"):
                df[column] = df[column].apply(lambda x: str(x))
            # FIXME: sample 10 values and if more than 50% are numeric than convert to numeric
            # convert numeric: df[column] = df[column].apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
            testval = df[column][0]
            if hasattr(testval, 'count') and \
                (testval.count("/") == 2 or testval.count(":") >= 2):
                try:
                    df[column] = pd.to_datetime(df[column], errors="coerce")
                    df[column].replace({np.nan: None}, inplace = True)
                except:
                    pass

        schema = pa.Schema.from_pandas(df)

        for col in schema.names:
            f = schema.field(col)
            if pa.types.is_list(f.type) and \
                (f.type.value_type == pa.null() or not pa.types.is_primitive(f.type.value_type)):
                # Remove the column
                #print("Dropping non-primitive or null column from df: ", f.name)
                df.drop(columns=f.name, inplace=True)
                continue

            if f.name in cols_to_drop:
                df.drop(columns=f.name, inplace=True)

            # Caller can optionally specify a set of columns to keep. We keep any column with
            # either an exact match, or a parent property match ("name_??").
            if self.select:
                if not (f.name in self.select or \
                    any(sname for sname in self.select if \
                        (re.match(sname + r"_.*", f.name) or re.match(sname, f.name)))):
                    #print("Dropping column missing from 'select' clause: ", f.name)
                    df.drop(columns=f.name, inplace=True)


    def _set_duck(self, duck):
        self.storage_mgr: DuckdbStorageManager = DuckdbStorageManager("_system_", duck)

    def run(self):
        with DuckContext() as duck:
            self.storage_mgr: DuckdbStorageManager = DuckdbStorageManager("_system_", duck)
            self.perform_scan(duck)

    def save_scan_record(self, values: dict):
        print("*SAVING SCAN RECORD: ", values)

        self.storage_mgr.log_object(
            "tablescans", 
            self.tableMgr.name,
            values
        )

    def get_last_scan_records(self, limit = 3):
        return self.storage_mgr.get_log_objects("tablescans", self.tableMgr.name, limit=limit)

    def clear_scan_record(self):
        self.storage_mgr.delete_log_objects("tablescans", self.tableMgr.name)

    def _flush_rows_to_db(self, duck, tableMgr, next_df, page, flush_count):
        # Subclasses must implement
        pass

    def get_query_mgr(self):
        return self.tableMgr.table_spec

    def perform_scan(self, duck):
        print("Running table scan for: ", self.tableMgr.name)
        scan_start = time.time()
        self.save_scan_record({"scan_start": scan_start})

        page = 1
        page_flush_count = 5 # flush 5 REST calls worth of data to the db
        row_buffer_df = None
        table_cols = set()
        logger = SimpleLogger(self.tableMgr.adapter)

        resource_query_mgr = self.get_query_mgr()

        for json_page, size_return in resource_query_mgr.query_resource(self.tableLoader, logger):
            record_path = self.tableMgr.table_spec.result_body_path
            df = pd.json_normalize(json_page, record_path=record_path, sep='_')
            size_return.append(df.shape[0])

            if df.shape[0] == 0:
                continue

            self.cleanup_df_page(df)

            if page == 1:
                row_buffer_df = df
            else:
                if table_cols:
                    # Once we set the table columns from the first flush, then we enforce that list
                    usable = list(table_cols.intersection(df.columns.tolist()))
                    df = df[usable]
                row_buffer_df = pd.concat([row_buffer_df, df], axis='index', ignore_index=True)

            # Flush rows
            if (page % page_flush_count) == 0:
                self._flush_rows_to_db(duck, self.tableMgr, row_buffer_df, page, page_flush_count)
                if page == page_flush_count:
                    table_cols = set(row_buffer_df.columns.tolist())
                row_buffer_df = row_buffer_df[0:0] # clear flushed rows, but keep columns

            page += 1

        if row_buffer_df.shape[0] > 0:
            self._flush_rows_to_db(duck, self.tableMgr, row_buffer_df, page, page_flush_count)

        # We save at the end in case we encounter an error
        self.save_scan_record({"scan_complete": scan_start})

        print("Finished table scan for: ", self.tableMgr.name)

class InitialTableLoad(BaseTableScan):
    """
       Specialization of BaseTableScan for the initial loading of a table.
    """
    def _flush_rows_to_db(self, duck, tableMgr, next_df, page, flush_count):
        print(f"Saving page {page} with {next_df.shape[1]} columns")
        if page <= flush_count:
            # First set of pages, so create the table
            duck.register('df1', next_df)
            # create the table AND flush current row_buffer values to the db
            try:
                duck.execute(f"describe {tableMgr.name}")
                # table already exists, so assume we are updating
                duck.execute(f"set search_path='{tableMgr.adapter.name}'")
                duck.append(tableMgr.table_spec.name, next_df)
            except Exception as e:
                if 'Catalog Error' in str(e):
                    duck.execute(f"create table {tableMgr.name} as select * from df1")
                else:
                    raise
            # FIXME: The following shouldn't be neccessary, but it seems like `Duck.append`
            # does not work properly with qualified table names, so we hack around it
            # by setting the search path
            duck.execute(f"set search_path='{tableMgr.adapter.name}'")
        else:
            duck.append(tableMgr.table_spec.name, next_df)

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
            last_scan_time = datetime.utcfromtimestamp(start_rec['scan_start'])
        except StopIteration:
            # Failed to find a start time
            pass
        self._query_mgr: TableUpdater = self.tableMgr.table_spec.get_table_updater(last_scan_time)
        self._target_table = self.tableMgr.name

        if self._query_mgr.should_replace():
            # Updater will replace the existing table rather than appending to it. So
            # download data into a temp file
            self._target_table_root = self.tableMgr.table_spec.name + "__temp"
            self._target_table = self.tableMgr.adapter.name + "." + self._target_table_root
            # drop the temp table in case it's lying around
            duck.execute(f"DROP TABLE IF EXISTS {self._target_table}")
            # Use parent class to download data from the target API (calling _flush_rows along the way)
            super().perform_scan(duck)
            # Now swap the new table into place
            duck.execute(f"""
            BEGIN;
            DROP TABLE {self.tableMgr.name};
            ALTER TABLE {self._target_table} RENAME TO {self.tableMgr.table_spec.name};
            COMMIT;
            """)
        else:
            self._target_table_root = self.tableMgr.table_spec.name
            self._target_table = self.tableMgr.adapter.name + "." + self._target_table_root
            super().perform_scan(duck)

    def get_query_mgr(self):
        return self._query_mgr

    def _flush_rows_to_db(self, duck, tableMgr, next_df, page, flush_count):
        print(f"Saving page {page} with {next_df.shape[1]} columns")
        if page <= flush_count:
            duck.execute(f"set search_path='{tableMgr.adapter.name}'")
        # First delete any existing records
        keys = next_df[tableMgr.table_spec.key]  #.values.tolist()
        duck.register("_keys", pd.DataFrame(keys))       
        #print("Keys count: ", duck.execute("select count(*) from _keys").fetchall())
        rows = duck.execute(
f"DELETE FROM {self._target_table} WHERE {tableMgr.table_spec.key} IN (select * from _keys)"
        ).fetchone()
        duck.unregister("_keys")
        # Now append the new records             
        duck.append(self._target_table_root, next_df)


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

        with DuckContext() as duck:
            if self.query:
                r = duck.execute(self.query)
            else:
                r = duck.execute(f"select * from {self.table}")

            # FIXME: Use DF chunking and multiple pages
            self.adapter.write_page(self.output_handle, r.df(), output_logger, append=self.append)

            self.adapter.close_output_table(self.output_handle)


TableLoader = typing.NewType("TableLoader", None)


class TableMgr:
    def __init__(self, schema, adapter, table_spec, auth = None, params={}):
        self.schema = schema
        self.name = schema + "." + table_spec.name
        self.adapter = adapter
        self.table_spec: TableDef = table_spec
        self.synchronous_scanning = True

    def has_data(self):
        return True

    def truncate(self, duck):
        # Remove table from duck db
        duck.execute(f"drop table {self.name}")

    def _create_scanner(self, tableLoader: TableLoader):
        return InitialTableLoad(self, tableLoader=tableLoader, select=self.table_spec.select_list)

    def load_table(self, waitForScan=False, tableLoader=None):
        # Spawn a thread to query the table source
        self.scan = self._create_scanner(tableLoader)
        if self.synchronous_scanning or waitForScan:
            self.scan.run()
        else:
            self.scan.start()

    def refresh_table(self, tableLoader):
        scanner = TableUpdateScan(self, tableLoader=tableLoader, select=self.table_spec.select_list)
        scanner.run()

class TableLoader:
    """Master loader class. This class can load the data for any table, either by 
       querying the table from DuckDB or else by calling the underlying REST API.

       Tables can be in 4 states:
       1. Unknown (schema and table name never registered)
       2. Never queried. Call the Adapter to query the data and store it to local db.
       3. Table exists in the local db.
    """
    def __init__(self, silence_errors=False, given_connections: list[Connection]=None):
        with DuckContext() as duck:       
            try:
                if given_connections:
                    self.connections = given_connections
                else:
                    self.connections: list[Connection] = Connection.setup_connections(
                        os.path.join(os.path.dirname(__file__), "connections.yaml"), 
                        # FIXME: provide Duckdb here
                        storage_mgr_maker=lambda schema: DuckdbStorageManager(schema, duck)
                    )
            except:
                if not silence_errors:
                    raise
                else:
                    self.connections = []
            self.tables: dict[str, TableMgr] = {}

            self.adapters: dict[str, Adapter] = dict(
                [(conn.schema_name, conn.adapter) for conn in self.connections if conn.adapter.supports_commands()]
            )
            # Connections defines the set of schemas we will create in the database.
            # For each connection/schema then we will define the tables as defined
            # in the REST spec for the target system.
            for conn in self.connections:
                duck.execute(f"create schema if not exists {conn.schema_name}")
                for t in conn.adapter.list_tables():
                    tmgr = TableMgr(conn.schema_name, conn.adapter, t)
                    self.tables[tmgr.name] = tmgr

    def materialize_table(self, schema, table):
        with DuckContext() as duck:
            qual = schema + "." + table
            if qual in self.tables:
                tmgr = self.tables[qual]
            else:
                table_spec = self.adapters[schema].lookupTable(table)
                tmgr = TableMgr(schema, self.adapters[schema], table_spec)
                self.tables[tmgr.name] = tmgr           
            tmgr.load_table(tableLoader=self)
            return tmgr.has_data()

    def refresh_table(self, table_ref):
        self.tables[table_ref].refresh_table(tableLoader=self)

    def read_table_rows(self, table):
        with DuckContext() as duck:
            if not self.table_exists_in_db(table):
                tmgr = self.tables[table]
                tmgr.load_table(tableLoader=self, waitForScan=True)

            if self.table_exists_in_db(table):
                r = duck.execute(f"select * from {table}")
                while True:
                    df = r.fetch_df_chunk()
                    if df.size == 0 or df.shape == (1,1):
                        break
                    else:
                        yield df
            else:
                raise RuntimeError(f"Could not get rows for table {table}")

    def lookup_connection(self, name):
        return next(c for c in self.connections if c.schema_name == name)

    def truncate_table(self, table):
        with DuckContext() as duck:
            self.tables[table].truncate(duck)

    def table_exists_in_db(self, table):
        try:
            # FIXME: memoize the results here
            with DuckContext() as duck:
                duck.execute(f"select 1 from {table}")
            return True
        except Exception as e:
            if 'Catalog Error' in str(e):
                return False
            else:
                raise

class DuckdbStorageManager(StorageManager):
    """
        Stores adapter metadata in DuckDB. Creates a "meta" schema, and creates
        tables named <adapter schema>_<collection name>.
    """
    TABLE_SCHEMA = "(id VARCHAR PRIMARY KEY, blob JSON)"
    LOG_TABLE_SCHEMA = "(id VARCHAR, created DATETIME, blob JSON)"
    VAR_NAME_SENTINAL = '__var__'

    def __init__(self, adapter_schema: str, duck):
        self.adapter_schema = adapter_schema
        self.duck = duck
        self.duck.execute("create schema if not exists meta")

    def ensure_col_table(self, duck, name) -> dict:
        table = self.adapter_schema + "_" + name
        duck.execute(
            f"create table if not exists meta.{table} {self.TABLE_SCHEMA}"
        )
        return table

    def ensure_log_table(self, duck, name) -> dict:
        table = self.adapter_schema + "_" + name
        duck.execute(
            f"create table if not exists meta.{table} {self.LOG_TABLE_SCHEMA}"
        )
        return table

    def create_var_storage_table(self, duck):
        table = "system__vars"
        SCHEMA = "(name VARCHAR PRIMARY KEY, value BLOB)"
        duck.execute(
            f"create table if not exists meta.{table} {SCHEMA}"
        )
        return table

    def put_var(self, name, value):
        table = self.create_var_storage_table(self.duck)
        self.duck.execute(f"delete from meta.{table} where name = '{name}'")
        if isinstance(value, pd.DataFrame):
            self.duck.register('df1', value)
            # create the table AND flush current row_buffer values to the db
            name = name.lower() + self.VAR_NAME_SENTINAL
            self.duck.execute(f"create or replace table meta.{name} as select * from df1")
            self.duck.unregister("df1")
        else:
            self.duck.execute(
                f"insert into meta.{table} (name, value) values (?, ?)",
                (name, pickle.dumps(value))
            )

    def list_vars(self):
        table = self.create_var_storage_table(self.duck)
        scalars = self.duck.execute(f"select name from meta.{table}").fetchall()
        scalars.extend(list(
            self.duck.execute(
                Queries.saved_var_tables.format(self.VAR_NAME_SENTINAL)
            ).fetchall()
        ))
        return scalars

    def get_var(self, name):
        table = self.create_var_storage_table(self.duck)
        rows = self.duck.execute(f"select value from meta.{table} where name = ?", [name]).fetchall()
        if len(rows) > 0:
            return pickle.loads(rows[0][0])
        else:
            name = name.lower() + self.VAR_NAME_SENTINAL
            return self.duck.execute(f"select * from meta.{name}").df()
            #raise RuntimeError(f"No value stored for global variable '{name}'")

    def put_object(self, collection: str, id: str, values: dict) -> None:
        table = self.ensure_col_table(self.duck, collection)
        self.duck.execute(f"delete from meta.{table} where id = ?", [id])
        self.duck.execute(f"insert into meta.{table} values (?, ?)", [id, json.dumps(values)])

    def log_object(self, collection: str, id: str, values: dict) -> None:
        table = self.ensure_log_table(self.duck, collection)
        created = datetime.utcnow()
        self.duck.execute(f"insert into meta.{table} values (?, ?, ?)", [id, created, json.dumps(values)])

    def get_object(self, collection: str, id: str) -> dict:
        table = self.ensure_col_table(self.duck, collection)
        rows = self.duck.execute(f"select id, blob from meta.{table} where id = ?", [id]).fetchall()
        if len(rows) > 0:
            return json.loads(rows[0][1])
        return None

    def get_log_objects(self, collection: str, id: str, limit=1) -> dict:
        table = self.ensure_col_table(self.duck, collection)
        rows = self.duck.execute(
            f"select id, blob from meta.{table} where id = ? order by created desc limit {limit}", 
            [id]
        ).fetchall()
        return [json.loads(row[1]) for row in rows]

    def delete_object(self, collection: str, id: str) -> bool:
        table = self.ensure_col_table(self.duck, collection)
        r = self.duck.execute(f"delete from meta.{table} where id = ?", [id]).fetchall()
        print("Delete resulted in: ", r)

    def delete_log_objects(self, collection: str, id: str):
        self.delete_object(collection, id)

    def list_objects(self, collection: str) -> list[tuple]:
        table = self.ensure_col_table(self.duck, collection)
        return [
            (key, json.loads(val)) for key, val in \
                self.duck.execute(f"select id, blob from meta.{table}").fetchall()
        ]

class ParserVisitor(Visitor):
    """ Utility class for visiting our parse tree and assembling the relevant parts
        for the parsed command. Also works on incomplete results so can be used
        for autocompletion in the Jupyter kernel.
    """
    MATPLOT_CHART_MAP: Dict[str, str] = {
        'bar_chart': 'bar',
        'pie_chart': 'pie',
        'line_chart': 'line',
        'area_chart': 'area',
        'hbar_chart': 'barh'
    }

    def perform_new_visit(self, parse_tree, full_code):
        self._the_command = None
        self._the_command_args = {}
        self._full_code = full_code
        self.visit(parse_tree)
        return self._the_command

    def count_table(self, tree):
        self._the_command = "count_table"
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )
        return tree

    def clear_table(self, tree):
        self._the_command = 'clear_table'
        self._the_command_args['table_schema_ref'] = find_node_return_children("table_schema_ref", tree)
        if self._the_command_args['table_schema_ref']:
            self._the_command_args['table_schema_ref'] = ".".join(self._the_command_args['table_schema_ref'])
        return tree

    def create_statement(self, tree):
        self._the_command = 'create_statement'

    def create_view_statement(self, tree):
        self._the_command = 'create_view_statement'

    def create_chart(self, tree):
        self._the_command = 'create_chart'
        self._the_command_args['chart_name'] = find_node_return_child('chart_name', tree)
        self._the_command_args['chart_type'] = find_node_return_child('chart_type', tree)
        self._the_command_args['chart_source'] = \
            find_node_return_children(['chart_source', 'table_schema_ref'], tree)
        self._the_command_args['chart_where'] = find_subtree('create_chart_where', tree)
        # collect chart params
        key = value = None
        params = {}
        for child in self._the_command_args['chart_where'].children:
            key = key or find_node_return_child("chart_param", child)
            value = value or find_node_return_child("param_value", child)
            if value is not None:
                value = value.strip("'")
            if key and value:
                params[key] = value
                key = value = None
        self._the_command_args['chart_params'] = params

    def delete_statement(self, tree):
        self._the_command = 'delete_statement'

    def describe(self, tree):
        self._the_command = 'describe'
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )
        return tree

    def drop_table(self, tree):
        self._the_command = "drop_table"
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )

    def drop_schema(self, tree):
        self._the_command = "drop_schema"
        self._the_command_args["schema_ref"] = find_node_return_child("schema_ref", tree)

    def export_table(self, tree):
        self._the_command = "export_table"
        self._the_command_args['table_ref'] = collect_child_text("table_ref", tree, full_code=self._full_code)
        self._the_command_args['adapter_ref'] = find_node_return_child("adapter_ref", tree)
        self._the_command_args['file_ref'] = find_node_return_child("file_ref", tree).strip("'")
        self._the_command_args['write_option'] = find_node_return_child("write_option", tree)

    def help(self, tree):
        self._the_command = 'help'
        self._the_command_args['help_choice'] = collect_child_strings('HELP_CHOICE', tree)

    def insert_statement(self, tree):
        self._the_command = 'insert_statement'

    def refresh_table(self, tree):
        self._the_command = 'refresh_table'
        self._the_command_args['table_ref'] = \
            collect_child_text("table_ref", tree, full_code=self._full_code)

    def select_query(self, tree):
        self._the_command = 'select_query'
    
    def select_for_writing(self, tree):
        self._the_command = "select_for_writing"
        self._the_command_args["adapter_ref"] = find_node_return_child("adapter_ref", tree)
        self._the_command_args["file_ref"] = find_node_return_child("file_ref", tree).strip("'")
        self._the_command_args["select_query"] = collect_child_text("select_query", tree, self._full_code)

    def set_variable(self, tree):
        self._the_command = "set_variable"
        self._the_command_args["var_ref"] = find_node_return_child("var_ref", tree)
        self._the_command_args["var_expression"] = collect_child_text("var_expression", tree, self._full_code).strip()

    def show_columns(self, tree):
        """" Always returns 'table_ref' either qualified or unqualified by the schema name."""
        self._the_command = 'show_columns'
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )
        filter = collect_child_strings("column_filter", tree)
        if filter:
            self._the_command_args['column_filter'] = filter.strip()
        return tree

    def show_schemas(self, tree):
        self._the_command = 'show_schemas'
        return tree

    def show_tables(self, tree):
        self._the_command = 'show_tables'
        self._the_command_args['schema_ref'] = find_node_return_child("schema_ref", tree)
        return tree

    def show_variable(self, tree):
        self._the_command = "show_variable"
        self._the_command_args["var_ref"] = find_node_return_child("var_ref", tree)        

    def show_variables(self, tree):
        self._the_command = "show_variables"

class CommandInterpreter:
    """
        The interpretr for Unify. You call `run_command` with the code you want to execute
        and this class will parse it and execute the command, returning the result as a
        tuple of (output_lines[], result_object) where result_object is usually a DataFrame
        but could be a dict containing an image result instead.
    """
    _last_result: pd.DataFrame = None

    def __init__(self, debug=False, silence_errors=False):
        self.debug = debug
        path = os.path.join(os.path.dirname(__file__), "grammar.lark")
        self.parser = Lark(open(path).read(), propagate_positions=True)
        self.parser_visitor = ParserVisitor()
        self.loader = TableLoader(silence_errors)
        self.adapters: dict[str, Adapter] = self.loader.adapters
        self.logger: OutputLogger = None
        self.session_vars: dict[str, object] = {}

    def _list_schemas(self, match_prefix=None):
        with DuckContext() as duck:
            all = sorted(list(r[0] for r in duck.execute(Queries.list_schemas).fetchall()))
            if match_prefix:
                all = [s[len(match_prefix):] for s in all if s.startswith(match_prefix)]
            return all
        
    def _list_tables_filtered(self, schema, table=None):
        conn = self.loader.lookup_connection(schema)
        table = table or ''
        return sorted(list(t.name[len(table):] for t in conn.list_tables() if t.name.startswith(table)))

    def pre_handle_command(self, code):
        m = re.match(r"\s*([\w_0-9]+)\s+(.*$)", code)
        if m:
            first_word = m.group(1)
            rest_of_command = m.group(2)
            if first_word in self.adapters:
                logger: OutputLogger = OutputLogger()
                handler: Adapter = self.adapters[first_word]
                return handler.run_command(rest_of_command, logger)

    def substitute_variables(self, code):
        breakpoint()
        if re.match(r"\s*\$[\w_0-9]+\s*$", code):
            return code # simple request to show the variable value

        def lookup_var(match):
            var_name = match.group(1)
            value = self._get_variable(var_name)
            if isinstance(value, pd.DataFrame):
                ref_var =f"{var_name}__actualized"
                self.duck.register(ref_var, value)
                return ref_var
            elif value is not None:
                # literal substitution
                if isinstance(value, str):
                    return f"'{value}'"
                else:
                    return str(self.session_vars[var_name])
            else:
                return "$" + var_name
                #raise RuntimeError(f"Unknown variable reference '{var_name}'")

        match = re.match(r"\s*(\$[\w_0-9]+)\s*=(.*)", code, re.DOTALL)
        if match:
            # var assignment, only interpolate the right hand side
            return match.group(1) + "=" + self.substitute_variables(match.group(2))
        else:
            # interpolate the whole command
            return re.sub(r"\$([\w_0-9]+)", lookup_var, code)

    def run_command(self, cmd, input_func=input) -> tuple[list, pd.DataFrame]:
        """
            Executes a command through our interpreter and returns the results
            as a tuple of (output_lines, output_object) where output_line contains
            a list of string to print and output_object is an object which should
            be rendered to the user. 

            Support object types are: 
            - a DataFrame
            - A dict containing keys: "mime_type" and "data" for images
        """
        def clean_df(object):
            if isinstance(object, pd.DataFrame):
                self._last_result = object
                if 'count_star()' in object.columns:
                    object.rename(columns={'count_star()': 'count'}, inplace=True)

        try:
            with DuckContext() as duck:
                self.duck = duck

                cmd = self.substitute_variables(cmd)
                # Allow Adapters to process their special commands
                output: OutputLogger = self.pre_handle_command(cmd)
                if output is not None:
                    return (output.get_output(), output.get_df())

                self.print_buffer = []
                self._cmd = cmd
                result = None
                self.input_func = input_func
                self.logger: OutputLogger = OutputLogger()
                try:
                    parse_tree = self.parser.parse(self._cmd)
                    command = self.parser_visitor.perform_new_visit(parse_tree, full_code=cmd)
                    if command:
                        result = getattr(self, command)(**self.parser_visitor._the_command_args)
                except lark.exceptions.LarkError:
                    # Let any parsing exceptions send the command down to the db
                    result = self._execute_duck(self._cmd)
                if isinstance(result, pd.DataFrame):
                    self.print("{} row{}".format(result.shape[0], "s" if result.shape[0] != 1 else ""))
                clean_df(result)
                return (self.logger.get_output(), result)
        finally:
            self.duck = None

    def _execute_duck(self, query, header=True):
        return self.duck.execute(query).df()

    def print(self, *args):
        self.logger.print(*args)

    def get_input(self, prompt: str):
        return self.input_func(prompt)

    ################
    ## Commands 
    #
    # All commands either "print" to the result buffer, or else they return
    # a DataFrame result (or both). It the the responsibilty of the host
    # program to render the result. Commands should call `get_input` to
    # retrieve input from the user interactively.
    ################
    def count_table(self, table_ref):
        return self._execute_duck(f"select count(*) from {table_ref}")

    def help(self, help_choice):
        # FIXME: collect help from each function
        helps = {
            "schemas": """Every connected system is represented in the Unify database as a schema.
            The resources for the system appear as tables within this schema. Initially all tables
            are empty, but they are imported on demand from the connected system whenver you access
            the table.

            Some systems support custom commands, which you can invoke by using the schema name
            as the command prefix. You can get help on the connected system and its commands by
            typing "help <schema>".
            """,
            "charts": """Help for charts"""
        }
        if help_choice:
            msg = helps[help_choice]
        else:
            msg = """
help - show this message
help schemas
help charts
help import
help export
        """
        for l in msg.splitlines():
            self.print(l.strip())
    
    def drop_table(self, table_ref):
        val = self.get_input(f"Are you sure you want to drop the table '{table_ref}' (y/n)? ")
        if val == "y":
            return self._execute_duck(self._cmd)

    def drop_schema(self, schema_ref):
        val = input(f"Are you sure you want to drop the schema '{schema_ref}' (y/n)? ")
        if val == "y":
            return self._execute_duck(self._cmd)

    def export_table(self, adapter_ref, table_ref, file_ref, write_option=None):
        if file_ref.startswith("(") and file_ref.endswith(")"):
            # Evaluate an expression for the file name
            result = self.duck.execute(f"select {file_ref}").fetchone()[0]
            file_ref = result

        if adapter_ref in self.adapters:
            adapter = self.adapters[adapter_ref]
            exporter = TableExporter(
                table=table_ref, 
                adapter=adapter, 
                target_file=file_ref,
                allow_overwrite=(write_option == "overwrite"),
                allow_append=(write_option == "append")
            )
            exporter.run(self.logger)
            self.print(f"Exported query result to '{file_ref}'")
        else:
            self.print(f"Error, uknown adapter '{adapter_ref}'")

    def refresh_table(self, table_ref):
        self.loader.refresh_table(table_ref)

    def set_variable(self, var_ref: str, var_expression: str):
        is_global = var_ref.upper() == var_ref
        if not var_expression.lower().startswith("select "):
            # Need to evaluate the scalar expression
            val = self.duck.execute("select " + var_expression).fetchone()[0]
            self._save_variable(var_ref, val, is_global)
            self.print(val)
        else:
            val = self.duck.execute(var_expression).df()
            self._save_variable(var_ref, val, is_global)
            return val

    def _get_variable(self, name: str):
        if name.upper() == name:
            store: DuckdbStorageManager =DuckdbStorageManager(None, self.duck)
            return store.get_var(name)
        else:
            return self.session_vars[name]

    def _save_variable(self, name: str, value, is_global: bool):
        if is_global:
            # Save a scalar to our system table in meta
            store: DuckdbStorageManager =DuckdbStorageManager(None, self.duck)
            store.put_var(name, value)
        else:
            self.session_vars[name] = value

    def show_schemas(self):
        return self._execute_duck(Queries.list_schemas)

    def show_tables(self, schema_ref=None):
        if schema_ref:
            potential = []
            if schema_ref in self.adapters:
                for tableDef in self.adapters[schema_ref].list_tables():
                    potential.append(schema_ref + "." + tableDef.name)

            actual = [r[0] for r in \
                self.duck.execute(Queries.list_tables_filtered.format(schema_ref)).fetchall()]
            full = set(potential) | set(actual)
            #self.print("tables\n--------------")
            #self.print("\n".join(list(full)))
            return pd.DataFrame({"tables": list(full)})
        else:
            self.print("{:20s} {}".format("schema", "table"))
            self.print("{:20s} {}".format("---------", "----------"))
            for r in self.duck.execute(Queries.list_all_tables).fetchall():
                self.print("{:20s} {}".format(r[0], r[1]))

    def show_columns(self, table_ref, column_filter=None):
        if table_ref:
            if column_filter:
                parts = table_ref.split(".")
                column_filter = re.sub(r"\*", "%", column_filter)
                return self._execute_duck(Queries.list_columns.format(parts[0], parts[1], column_filter))
            else:
                return self._execute_duck(f"describe {table_ref}")
        else:
            return self._execute_duck("describe")

    def describe(self, table_ref):
        if table_ref is None:
            return self._execute_duck("describe")
        else:
            return self._execute_duck(f"describe {table_ref}")

    def create_statement(self):
        return self._execute_duck(self._cmd)

    def create_view_statement(self):
        return self._execute_duck(self._cmd)

    def insert_statement(self):
        return self._execute_duck(self._cmd)

    def delete_statement(self):
        return self._execute_duck(self._cmd)

    def load_adapter_data(self, schema_name, table_name):
        if self.loader.materialize_table(schema_name, table_name):
            return True
        else:
            self.print("Loading table...")
            return False

    def select_query(self, fail_if_missing=False):
        try:
            return self._execute_duck(self._cmd)

        except RuntimeError as e:
            if fail_if_missing:
                self.print(e)
                return
            m = re.search(r"Table with name (\S+) does not exist", str(e))
            if m:
                table = m.group(1)
                # Extract the schema from the original query
                m = re.search(f"(\\w+)\\.{table}", self._cmd)
                if m:
                    if self.load_adapter_data(m.group(1), table):
                        return self.select_query(fail_if_missing=True)
                else:
                    self.print(e)
            else:
                self.print(e)

    def select_for_writing(self, select_query, adapter_ref, file_ref):
        if adapter_ref in self.adapters:
            adapter = self.adapter[adapter_ref]
            exporter = TableExporter(select_query, adapter, file_ref)
            exporter.run()
            self.print(f"Exported query result to '{file_ref}'")
        else:
            self.print(f"Error, uknown adapter '{adapter_ref}'")

    def show_variable(self, var_ref):
        if var_ref in self.session_vars:
            self.print(self.session_vars[var_ref])
        elif var_ref.upper() == var_ref:
            store: DuckdbStorageManager =DuckdbStorageManager(None, self.duck)
            value = store.get_var(var_ref)
            if isinstance(value, pd.DataFrame):
                return value
            else:
                self.print(value)
        else:
            self.print(f"Error, unknown variable '{var_ref}'")

    def show_variables(self):
        rows = [(k, "[query result]" if isinstance(v, pd.DataFrame) else v) for k, v in self.session_vars.items()]
        store: DuckdbStorageManager =DuckdbStorageManager(None, self.duck)
        vars = [k[0][0:-len(store.VAR_NAME_SENTINAL)].upper() for k in store.list_vars()]
        rows.extend([(k, "[query result]") for k in vars])
        return pd.DataFrame(rows, columns=["variable", "value"])

    def clear_table(self, table_schema_ref=None):
        breakpoint()
        self.loader.truncate_table(table_schema_ref)
        self.print("Table cleared: ", table_schema_ref)

    def create_chart(
        self, 
        chart_name=None, 
        chart_type=None, 
        chart_source=None, 
        chart_where=None,
        chart_params={}):
        # FIXME: observe chart_source
        if "x" not in chart_params:
            raise RuntimeError("Missing 'x' column parameter for chart X axis")
        df = self._last_result
        if df is None:
            raise RuntimeError("No query result available")
        plt.rcParams["figure.figsize"]=10,8
        plt.rcParams['figure.dpi'] = 100 
        if chart_type == "pie_chart":
            df = df.set_index(chart_params["x"])
        kind = ParserVisitor.MATPLOT_CHART_MAP[chart_type]
        title = chart_params.get("title", "")

        fig, ax = plt.subplots()

        df.plot(x = chart_params["x"], y = chart_params.get("y"), kind=kind,
                title=title, stacked=chart_params.get("stacked", False))
        plt.tight_layout()
        
        imgdata = io.BytesIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        return {"mime_type": "image/png", "data": imgdata.getvalue()}

class UnifyRepl:
    def __init__(self, interpreter: CommandInterpreter, wide_display=False):
        self.interpreter = interpreter
        if wide_display:
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)

        
    def loop(self):
        session = PromptSession(history=FileHistory(os.path.expanduser("~/.pphistory")))
        suggester = AutoSuggestFromHistory()
        try:
            while True:
                try:
                    cmd = session.prompt("> ", auto_suggest=suggester)
                    outputs, df = self.interpreter.run_command(cmd)
                    print("\n".join(outputs))
                    if df is not None:
                        with pd.option_context('display.max_rows', None):
                            if df.shape[0] == 0:
                                continue
                            fmt_opts = {
                                "index": False,
                                "max_rows" : None,
                                "min_rows" : 10,
                                "max_colwidth": 50,
                                "header": True
                            }
                            if df.shape[0] > 40:
                                pydoc.pager(df.to_string(**fmt_opts))
                            else:
                                print(df.to_string(**fmt_opts))
                except RuntimeError as e:
                    print(e)
                except Exception as e:
                    if isinstance(e, EOFError):
                        raise
                    traceback.print_exc()
        except EOFError:
            sys.exit(0)

if __name__ == '__main__':
    if '-silent' in sys.argv:
        silent = True
    else:
        silent = False
    interpreter = CommandInterpreter(debug=True, silence_errors=silent)
    UnifyRepl(interpreter).loop()

