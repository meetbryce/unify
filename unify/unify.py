from email.parser import Parser
import inspect
from inspect import isfunction
import time
from datetime import datetime, timedelta
import io
import inspect
import json
import logging
import math
import os
from threading import Thread
import pydoc
from pprint import pprint
import re
import sys
import traceback
import typing
import yaml

import lark
from lark import Lark, Visitor
from lark.visitors import v_args
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import sqlparse
from typing import Dict
import sqlglot
# CHARTING
import matplotlib.pyplot as plt
import altair

import pandas as pd
import pyarrow as pa
import numpy as np

from timeit import default_timer as timer

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

# DuckDB
import duckdb
from setuptools import Command
from .email_helper import EmailHelper

from .rest_schema import (
    Adapter, 
    Connection, 
    OutputLogger, 
    RESTView,
    TableDef,
    TableUpdater,
    UnifyLogger
)
from .db_wrapper import (
    ClickhouseWrapper, 
    DBWrapper, 
    DuckDBWrapper, 
    TableMissingException,
    UnifyDBStorageManager
)

from .storage_manager import StorageManager
from .schemata import LoadTableRequest, Queries
from .parsing_utils import (
    find_subtree, 
    find_node_return_child, 
    find_node_return_children,
    collect_child_strings,
    collect_child_text,
    collect_strings,
    collect_child_string_list
)

# Verify DB settings
if 'DATABASE_BACKEND' not in os.environ or os.environ['DATABASE_BACKEND'] not in ['duckdb','clickhouse']:
    raise RuntimeError("Must set DATABASE_BACKEND to 'duckdb' or 'clickhouse'")

dbmgr: DBWrapper = ClickhouseWrapper if os.environ['DATABASE_BACKEND'] == 'clickhouse' else DuckDBWrapper
       

def load_connections_config():
    trylist = [
        os.path.expanduser("~/unify/unify_connections.yaml"),
        os.path.expanduser("~/unify_connections.yaml"),
        os.path.realpath("./unify_connections.yaml")
    ]
    if 'UNIFY_CONNECTIONS' in os.environ:
        trylist.insert(0, os.environ['UNIFY_CONNECTIONS'])
    for p in trylist:
        if os.path.exists(p):
            logger.info("Loading connections config from: {}".format(p))
            return yaml.safe_load(open(p))
    raise RuntimeError("Could not find unify_connections.yaml in HOME or current directory.")

class SimpleLogger(UnifyLogger):
    def __init__(self, adapter: Adapter):
        self.adapter = adapter

    def log_table(self, table: str, level: int, *args):
        print(f"[{str(self.adapter)}: {table}] ", *args, file=sys.stderr)

class ColumnIntel:
    def __init__(self, name:str, attrs: dict) -> None:
        self.name: str = name
        self.attrs: dict = attrs

    @property
    def column_weight(self):
        # Heuristic to assign greater weight to likely more "interesting" columns
        weight = 0
        type = self.attrs["type_str"]

        # Preference shorter columns
        width = int(self.attrs["col_width"] / 5)
        if width <= 6:
            weight += 6 - width

        if self.attrs["key"] and width < 20:
            weight += 10
        if "time" in type or "date" in type:
            weight += 5
        elif type == "string":
            weight += 5
        elif type == "boolean":
            weight -= 20
        elif type == "integer" and width < 3:
            weight -= 10 # short ints aren't very useful. Often are flags
        weight += 15 - self.attrs["name_width"]
        # Push down very long columns
        weight -= int(width/25)

        # Push down columns with very low entropy
        if self.attrs["entropy"] < 0.1:
            weight -= 15

        if self.attrs["url"]:
            weight -= 5

        return weight
        
    @property
    def width(self):
        return self.attrs["col_width"]

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
        self.table_ref = tableMgr.schema + "." + tableMgr.table_spec.name
        self.select = select
        self.storage_mgr: UnifyDBStorageManager = None
        self.analyzed_columns = []

    def analyze_columns(self, df: pd.DataFrame):
        # Analyze a page of rows and extract useful column intelligence information for 
        # potential use later, such as by the `peek` command.
        if os.environ.get('UNIFY_SKIP_COLUMN_INTEL'):
            return
        for order, column in enumerate(df.columns):
            lowercol = column.lower()
            attrs = {
                "name": column,
                "type_str": pd.api.types.infer_dtype(df[column]),
                "url": False,
                "key": False,
                "order": order
            }
            if re.search("key|id", lowercol) and len(lowercol) <= 6:
                attrs["key"] = True
            elif re.search("key[_ -]|[_ -]key|id[_ -]|[_ -]id", lowercol) and len(lowercol) < 12:
                attrs["key"] = True
            width = math.ceil(sum([len(str(v)) for v in df[column].values]) / df.shape[0])
            attrs["col_width"] = width
            rows = df[column].sample(min(25, df.shape[0])).values
            for row in rows:
                if isinstance(row, str) and re.match("^http[s]*:/", row, re.IGNORECASE):
                    attrs["url"] = True
            name_width = len(column) + column.count("_")
            attrs["name_width"] = name_width
            entropy = df[column].value_counts().size / (float)(df.shape[0])
            attrs["entropy"] = entropy

            self.analyzed_columns.append(ColumnIntel(column, attrs))

    def save_analyzed_columns(self):
        if os.environ.get('UNIFY_SKIP_COLUMN_INTEL'):
            return
        cols: list[ColumnIntel] = sorted(self.analyzed_columns, key=lambda ci: ci.attrs['order'])
        for col in cols:
            self.storage_mgr.insert_column_intel(
                schema=self.tableMgr.schema,
                table_root=self.tableMgr.table_spec.name,
                column=col.name,
                attrs=col.attrs
        )

    def clear_analyzed_columns(self, table_ref):
        schema, table = table_ref.split(".")
        self.storage_mgr.delete_all_column_intel(schema=schema, subject_table=table)

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
        self.storage_mgr: UnifyDBStorageManager = UnifyDBStorageManager("_system_", duck)

    def run(self):
        with dbmgr() as duck:
            self.storage_mgr: UnifyDBStorageManager = UnifyDBStorageManager("_system_", duck)
            self.perform_scan(duck)

    def save_scan_record(self, values: dict):
        logger.debug("*SAVING SCAN RECORD: {}".format(values))
        self.storage_mgr.log_object(
            "tablescans", 
            self.tableMgr.name,
            values
        )

    def get_last_scan_records(self, limit = 3):
        return self.storage_mgr.get_log_objects("tablescans", self.tableMgr.name, limit=limit)

    def clear_scan_record(self):
        self.storage_mgr.delete_log_objects("tablescans", self.tableMgr.name)

    def create_table_with_first_page(self, duck: DBWrapper, next_df, schema, table_root):
        table = schema + "." + table_root
        if duck.table_exists(table):
            # table already exists, so assume we are updating
            duck.append_dataframe_to_table(next_df, table)
        else:
            duck.write_dataframe_as_table(next_df, schema, table_root)

    def get_query_mgr(self):
        return self.tableMgr.table_spec

    def scan_start_handler(self):
        pass

    def perform_scan(self, duck):
        self.scan_start_handler()
        print("Running table scan for: ", self.tableMgr.name)
        scan_start = time.time()
        self.save_scan_record({"scan_start": scan_start})

        page = 1
        page_flush_count = 5 # flush 5 REST calls worth of data to the db
        row_buffer_df = None
        table_cols = set()
        logger = SimpleLogger(self.tableMgr.adapter)

        resource_query_mgr = self.get_query_mgr()

        for query_result in resource_query_mgr.query_resource(self.tableLoader, logger):
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
                self._flush_rows_to_db_catch_error(duck, self.tableMgr, row_buffer_df, page, page_flush_count)
                if page == page_flush_count:
                    table_cols = set(row_buffer_df.columns.tolist())
                row_buffer_df = row_buffer_df[0:0] # clear flushed rows, but keep columns

            page += 1

        if row_buffer_df is not None and row_buffer_df.shape[0] > 0:
            self._flush_rows_to_db_catch_error(duck, self.tableMgr, row_buffer_df, page, page_flush_count)

        # We save at the end in case we encounter an error
        self.save_scan_record({"scan_complete": scan_start})

        print("Finished table scan for: ", self.tableMgr.name)

    def _flush_rows_to_db_catch_error(self, duck: DBWrapper, tableMgr, next_df, page, flush_count):
        try:
            self._flush_rows_to_db(duck, tableMgr, next_df, page, flush_count)
            if len(self.analyzed_columns) == 0:
                self.analyze_columns(next_df)
                if len(self.analyzed_columns) > 0:
                    self.save_analyzed_columns()
            
        except Exception as e:
            # "Core dump" the bad page
            core_file = f"/tmp/{self.tableMgr.name}_bad_page_{page}.csv"
            logger.critical("Error saving page to db, dumping to file: " + core_file)
            with open(core_file, "w") as f:
                next_df.to_csv(f)
            raise

class InitialTableLoad(BaseTableScan):
    def scan_start_handler(self):
        self.clear_analyzed_columns(self.table_ref)

    def _flush_rows_to_db(self, duck: DBWrapper, tableMgr, next_df, page, flush_count):
        ## Default version works when loading a new table
        print(f"Saving page {page} with {next_df.shape[1]} columns")
        if page <= flush_count:
            # First set of pages, so create the table
            self.create_table_with_first_page(duck, next_df, tableMgr.schema, tableMgr.table_spec.name)
        else:
            duck.append_dataframe_to_table(
                next_df, 
                tableMgr.schema, 
                tableMgr.table_spec.name
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
            duck.replace_table(self._target_table, self.tableMgr.name)
        else:
            self._target_table_root = self.tableMgr.table_spec.name
            self._target_table = self.tableMgr.schema + "." + self._target_table_root
            super().perform_scan(duck)

    def get_query_mgr(self):
        return self._query_mgr

    def _flush_rows_to_db(self, duck: DBWrapper, tableMgr, next_df, page, flush_count):
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
                duck.append_dataframe_to_table(
                    next_df, 
                    self.tableMgr.schema, 
                    self._target_table_root
                )
            return

        # To update a table we have to both remove existing copies of any rows
        # we downloaded and align our downloaded columns with the existing table.
        cols = duck.get_table_columns(self._target_table)
        # filter and order to the right columns
        next_df = next_df[cols]
        print(f"Saving update page {page} with {next_df.shape[0]} rows and {next_df.shape[1]} columns")

        # First delete any existing records
        keys = next_df[tableMgr.table_spec.key]  #.values.tolist()
        keys_table = duck.create_memory_table("__keys", pd.DataFrame(keys))

        duck.delete_rows(self._target_table, 
            where_clause = f"{tableMgr.table_spec.key} IN (SELECT * FROM {keys_table})")
        duck.drop_memory_table("__keys")

        # Now append the new records             
        duck.append_dataframe_to_table(next_df, tableMgr.schema, self._target_table_root)


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
                r = duck.execute_df(self.query)
            else:
                r = duck.execute_df(f"select * from {self.table}")

            # FIXME: Use DF chunking and multiple pages
            self.adapter.write_page(self.output_handle, r, output_logger, append=self.append)

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
    def __init__(self, silence_errors=False, given_connections: list[Connection]=None, lark_parser: Lark=None):
        self.lark_parser = lark_parser
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
            # Connections defines the set of schemas we will create in the database.
            # For each connection/schema then we will define the tables as defined
            # in the REST spec for the target system.
            for conn in self.connections:
                duck.create_schema(conn.schema_name)
                for t in conn.adapter.list_tables():
                    tmgr = TableMgr(conn.schema_name, conn.adapter, t)
                    self.tables[tmgr.name] = tmgr

    def _get_table_mgr(self, table):
        if table in self.tables:
            return self.tables[table]
        else:
            schema, table_root = table.split(".")
            table_spec = self.adapters[schema].lookupTable(table_root)
            tmgr = TableMgr(schema, self.adapters[schema], table_spec)
            self.tables[tmgr.name] = tmgr
            return tmgr

    def materialize_table(self, schema, table):
        with dbmgr() as duck:
            qual = schema + "." + table
            try:
                tmgr = self._get_table_mgr(qual)
            except KeyError:
                raise TableMissingException(f"{schema}.{table}")
            tmgr.load_table(tableLoader=self)
            return tmgr.has_data()

    def qualify_tables_in_view_query(self, query: str, from_list: list[str], schema: str,
                                    dialect: str):
        """ Replaces non-qualified table references in a view query with qualified ones.
            Returns the corrected query.
        """

        # Following *almost* works, except type capitalization still breaks on some queries
        # if not isinstance(from_list, list):
        #     from_list = [from_list]
        # froms = ",".join([(schema + "." + t) for t in from_list])
        # sql = sqlglot.parse_one(query, read=dialect).from_(froms, append=False).sql(dialect=dialect)
        # print("*******: ", sql)
        # return sql
       
        query += " "
        for table_root in sqlglot.parse_one(query).find_all(sqlglot.exp.Table):
            table = schema + "." + table_root.name.split(".")[0]
            query = re.sub("[\s,]+" + table_root.name + "[\s]+", f" {table} ", query)

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


    def analyze_columns(self, table_ref):
        with dbmgr() as duck:
            tmgr = self._get_table_mgr(table_ref)
            scanner = InitialTableLoad(tableLoader=self, tableMgr=tmgr)
            scanner._set_duck(duck)
            # Load a df page from the table
            for df in self.read_table_rows(table_ref, limit=25):
                scanner.clear_analyzed_columns(table_ref)
                scanner.analyze_columns(df)
                scanner.save_analyzed_columns()
                break

    def refresh_table(self, table_ref):
        self.tables[table_ref].refresh_table(tableLoader=self)

    def query_table(self, schema: str, query: str):
        # Our parser wasn't smart enough:
        # parser_visitor = ParserVisitor()
        # parse_tree = self.lark_parser.parse(query)
        # command = parser_visitor.perform_new_visit(parse_tree, full_code=query)
        # query_parts = parser_visitor._the_command_args

        parsed = sqlparse.parse(query)[0]
        if parsed[0].ttype != sqlparse.tokens.DML or parsed[0].value != 'select':
            raise RuntimeError(f"Invalid statement type: '{query}'")

        # Verify, for safety, no other DMLs inside
        col_list = []
        past_cols = False
        past_from = False
        past_tables = False
        table_list = []
        funcs = []
        where_clause = ""
        for idx in range(1, len(parsed.tokens)):
            token = parsed.tokens[idx]
            if isinstance(token, sqlparse.sql.IdentifierList):
                if not past_cols:
                    col_list = re.split(r"\s*,\s*", token.value)
                    past_cols = True
                elif past_from and not past_tables:
                    table_list = re.split(r"\s*,\s*", token.value)
                    past_tables = True

            if token.ttype == sqlparse.tokens.Keyword:
                if token.value == 'from':
                    past_from = True
            if isinstance(token, sqlparse.sql.Where):
                where_clause = token.value
            if token.ttype == sqlparse.tokens.DML:
                raise RuntimeError(f"Invalid sql query has embedded DML '{query}'")
            if isinstance(token, sqlparse.sql.Function):
                funcs.append(token.value)

        if len(col_list) == 0 and len(funcs) > 0:
            col_list.append(funcs[0]) # allow a simple select of a function value

        if len(col_list) == 0:
            raise RuntimeError(f"Invalid query references no columns or functions: '{query}'")

        table_refs = []
        for table in table_list:
            if "." in table:
                raise RuntimeError(f"Adapter queries cannot use qualified table names: {table}. Query: {query}")
            table = schema + "." + table
            if not self.table_exists_in_db(table):
                tmgr = self.tables[table]
                # TODO: Might want a timeout for large parent tables
                tmgr.load_table(tableLoader=self, waitForScan=True)
            table_refs.append(table)

        if table_refs:
            from_clause = " FROM " + ",".join(table_refs)
        else:
            from_clause = ""

        parent_query = "SELECT " + ",".join(col_list) + from_clause + where_clause
        with dbmgr() as duck:
            yield duck.execute_df(parent_query)

    def read_table_rows(self, table, limit=None):
        with dbmgr() as duck:
            if not self.table_exists_in_db(table):
                tmgr = self.tables[table]
                tmgr.load_table(tableLoader=self, waitForScan=True)

            if self.table_exists_in_db(table):
                # FIXME: Use chunking and multiple pages
                if limit is not None:
                    limit = f" limit {limit}"
                else:
                    limit = ""
                yield duck.execute_df(f"select * from {table} {limit}")
            else:
                raise RuntimeError(f"Could not get rows for table {table}")

    def lookup_connection(self, name):
        return next(c for c in self.connections if c.schema_name == name)

    def truncate_table(self, table):
        with dbmgr() as duck:
            self.tables[table].truncate(duck)

    def table_exists_in_db(self, table):
        try:
            # FIXME: memoize the results here
            with dbmgr() as duck:
                duck.execute(f"select 1 from {table}")
            return True
        except TableMissingException:
            return False

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
        try:
            self.visit(parse_tree)
        except Exception as e:
            print(parse_tree.pretty)
            raise
        return self._the_command

    def clear_table(self, tree):
        self._the_command = 'clear_table'
        self._the_command_args['table_schema_ref'] = find_node_return_children("table_schema_ref", tree)
        if self._the_command_args['table_schema_ref']:
            self._the_command_args['table_schema_ref'] = ".".join(self._the_command_args['table_schema_ref'])
        return tree

    def count_table(self, tree):
        self._the_command = "count_table"
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )
        return tree

    def create_chart(self, tree):
        self._the_command = 'create_chart'
        self._the_command_args['chart_name'] = find_node_return_child('chart_name', tree)
        self._the_command_args['chart_type'] = find_node_return_child('chart_type', tree)
        src = find_node_return_children(['chart_source', 'table_ref'], tree)
        if src:
            self._the_command_args['chart_source'] = src[0]
        else:
            self._the_command_args['chart_source'] = None
            
        where_clause = find_subtree('create_chart_where', tree)
        # collect chart params
        params = {}
        if where_clause:
            key = value = None
            for child in where_clause.children:
                key = key or find_node_return_child("chart_param", child)
                value = value or find_node_return_child("param_value", child)
                if value is not None:
                    value = value.strip("'")
                if key and value:
                    params[key] = value
                    key = value = None
        self._the_command_args['chart_params'] = params

    def create_statement(self, tree):
        self._the_command = 'create_statement'

    def create_view_statement(self, tree):
        self._the_command = 'create_view_statement'

    def delete_schedule(self, tree):
        self._the_command = 'delete_schedule'
        self._the_command_args['schedule_id'] = find_node_return_child("schedule_ref", tree).strip("'")

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
        self._the_command_args["cascade"] = self._full_code.strip().lower().endswith("cascade")

    def email_command(self, tree):
        self._the_command = "email_command"
        self._the_command_args['email_object'] = collect_child_text("email_object", tree, self._full_code)
        self._the_command_args['recipients'] = find_node_return_child("recipients", tree).strip("'")
        subject = find_node_return_child("subject", tree)
        if subject:
            self._the_command_args['subject'] = subject.strip("'")

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

    def peek_table(self, tree):
        self._the_command = 'peek_table'
        self._the_command_args['table_ref'] = \
            collect_child_text("table_ref", tree, full_code=self._full_code)
        count = find_node_return_child("line_count", tree)
        if count:
            self._the_command_args['line_count'] = int(count)

    def refresh_table(self, tree):
        self._the_command = 'refresh_table'
        self._the_command_args['table_ref'] = \
            collect_child_text("table_ref", tree, full_code=self._full_code)

    def reload_table(self, tree):
        self._the_command = 'reload_table'
        self._the_command_args['table_ref'] = \
            collect_child_text("table_ref", tree, full_code=self._full_code)

    def run_notebook_command(self, tree):
        self._the_command = 'run_notebook_command'
        nb = find_node_return_child("notebook_ref", tree)
        if nb:
            self._the_command_args["notebook_path"] = nb.strip("'")
        dt = collect_child_strings('datetime', tree)
        self._the_command_args["run_at_time"] = dt
        if find_subtree('run_every_command', tree):
            self._the_command_args["repeater"] = collect_child_strings("repeater", tree)

    def run_schedule(self, tree):
        self._the_command = 'run_schedule'

    def run_info(self, tree):
        self._the_command = 'run_info'
        self._the_command_args['schedule_id'] = find_node_return_child("schedule_ref", tree).strip("'")

    def select_query(self, tree):
        # lark: select_query: "select" WS col_list WS "from" WS table_list (WS where_clause)? (WS order_clause)? (WS limit_clause)?
        self._the_command = 'select_query'
        cols = collect_child_string_list("col_list", tree)
        cols = [c for c in cols if c.strip() != ""]
        self._the_command_args["col_list"] = cols
        tabs = collect_child_string_list("table_list", tree)
        tabs = [t for t in tabs if t.strip() != ""]
        self._the_command_args["table_list"] = tabs
        self._the_command_args["where_clause"] = collect_child_text("where_clause", tree, self._full_code)
        self._the_command_args["order_clause"] = collect_child_text("order_clause", tree, self._full_code)
        lim = collect_child_strings("limit_clause", tree)
        if lim:
            self._the_command_args["limit_clause"] = lim.strip()

    
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
        The interpreter for Unify. You call `run_command` with the code you want to execute
        and this class will parse it and execute the command, returning the result as a
        tuple of (output_lines[], result_object) where result_object is usually a DataFrame
        but could be a dict containing an image result instead.
    """
    _last_result: pd.DataFrame = None

    def __init__(self, silence_errors=False):
        path = os.path.join(os.path.dirname(__file__), "grammar.lark")
        self.parser = Lark(open(path).read(), propagate_positions=True)
        self.parser_visitor = ParserVisitor()
        self.loader = TableLoader(silence_errors, lark_parser=self.parser)
        self.adapters: dict[str, Adapter] = self.loader.adapters
        self.logger: OutputLogger = None
        self.session_vars: dict[str, object] = {}
        self.email_helper: EmailHelper = EmailHelper()
        self.duck: DBWrapper = None
        self.last_chart = None
        # Some commands we only support in interactive sessions. In non-interactive cases (background execution)
        # these commands will be no-ops.
        self.commands_needing_interaction = [
            'run_notebook_command'
        ]

    def _list_schemas(self, match_prefix=None):
        with dbmgr() as duck:
            all = sorted(list(r[0] for r in duck.execute(Queries.list_schemas).fetchall()))
            if match_prefix:
                all = [s[len(match_prefix):] for s in all if s.startswith(match_prefix)]
            return all
        
    def _list_tables_filtered(self, schema, table=None):
        try:
            conn = self.loader.lookup_connection(schema)
            table = table or ''
            return sorted(list(t.name[len(table):] for t in conn.list_tables() if t.name.startswith(table)))
        except StopIteration:
            return []

    def _list_schedules(self):
        with dbmgr() as duck:
            store: UnifyDBStorageManager = UnifyDBStorageManager("_system_", duck)
            return [{"id":id, "schedule":blob} for id, blob in store.list_objects("schedules")]
            
    def _truncate_schedules(self):
        with dbmgr() as duck:
            store: UnifyDBStorageManager = UnifyDBStorageManager("_system_", duck)
            for id, blob in store.list_objects("schedules"):
                store.delete_object("schedules", id)

    def _analyze_columns(self, table_ref: str):
        self.loader.analyze_columns(table_ref)

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
        if re.match(r"\s*\$[\w_0-9]+\s*$", code):
            return code # simple request to show the variable value

        def lookup_var(match):
            var_name = match.group(1)
            value = self._get_variable(var_name)
            if isinstance(value, pd.DataFrame):
                ref_var =f"{var_name}__actualized"
                self.duck.create_memory_table(ref_var, value)
                #self.duck.register(ref_var, value)
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

    def _command_needs_db_closed(self, command, parser_visitor: ParserVisitor) -> bool:
        return command == 'email_command' and parser_visitor._the_command_args.get('email_object') == 'notebook'

    def run_command(
        self, 
        cmd, 
        input_func=input, 
        get_notebook_func=None, 
        interactive: bool=True) -> tuple[list, pd.DataFrame]:
        # Executes a command through our interpreter and returns the results
        # as a tuple of (output_lines, output_object) where output_line contains
        # a list of string to print and output_object is an object which should
        # be rendered to the user. 

        # Support object types are: 
        # - a DataFrame
        # - A dict containing keys: "mime_type" and "data" for images
        # 
        # The Email function may need the notebook path. In this case the `get_notebook_func`
        # should be supplied by the kernel to return the notebook path.

        def clean_df(object):
            if isinstance(object, pd.DataFrame):
                self._last_result = object
                if 'count_star()' in object.columns:
                    object.rename(columns={'count_star()': 'count'}, inplace=True)

        # This is a hack for the email command, which will execute a notebook which needs exclusive
        # access to the database, so we defer to running the command until after we close the db.
        run_command_after_closing_db = None
        try:
            with dbmgr() as duck:
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

                    if not interactive and command in self.commands_needing_interaction:
                        self.print(f"Skipping command: {command}")
                        command = None

                    if command:    
                        method = getattr(self, command)
                        if 'notebook_path' in inspect.signature(method).parameters:
                            nb_path = get_notebook_func() if get_notebook_func else None
                            if 'notebook_path' not in self.parser_visitor._the_command_args:
                                self.parser_visitor._the_command_args['notebook_path'] = nb_path

                    if self._command_needs_db_closed(command, self.parser_visitor):
                        run_command_after_closing_db = command
                        command = None

                    if command:
                        result = getattr(self, command)(**self.parser_visitor._the_command_args)
                except lark.exceptions.LarkError:
                    # Let any parsing exceptions send the command down to the db
                    result = self._execute_duck(self._cmd)
                if isinstance(result, pd.DataFrame):
                    # print the row count
                    self.print("{} row{}".format(result.shape[0], "s" if result.shape[0] != 1 else ""))
                clean_df(result)
                if run_command_after_closing_db is None:
                    return (self.logger.get_output(), result)
        finally:
            self.duck = None
            if run_command_after_closing_db:
                result = getattr(self, run_command_after_closing_db)(**self.parser_visitor._the_command_args)
                # FIXME: use a namedtuple
                return (self.logger.get_output(), result)

    def _execute_duck(self, query, header=True):
        return self.duck.execute_df(query)

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
        """ count <table> - returns count of rows in a table """
        return self._execute_duck(f"select count(*) from {table_ref}")

    def help(self, help_choice):
        """ help - show this message 
        help schemas - overview of schemas
        help charts - help on generating charts
        help import - help on importing data
        help export - help on exporting data
        """
        if help_choice is None:
            for l in inspect.getdoc(self.help).splitlines():
                self.print(l)
            for f in sorted(inspect.getmembers(self.__class__, inspect.isfunction)):
                if f[0] in ['help','__init__']:
                    continue
                doc = inspect.getdoc(f[1])
                if doc:
                    self.print(doc)
            return
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
        msg = helps[help_choice]
        for l in msg.splitlines():
            self.print(l.strip())
    
    def drop_table(self, table_ref):
        """ drop <table> - removes the table from the database """
        val = self.get_input(f"Are you sure you want to drop the table '{table_ref}' (y/n)? ")
        if val == "y":
            return self._execute_duck(self._cmd)

    def drop_schema(self, schema_ref, cascade: bool):
        """ drop <schema> [cascade] - removes the entire schema from the database """
        val = input(f"Are you sure you want to drop the schema '{schema_ref}' (y/n)? ")
        if val == "y":
            return self.duck.drop_schema(schema_ref, cascade)

    def email_command(self, email_object, recipients, subject=None, notebook_path: str=None):
        """ email [notebook|<table>|chart <chart>| to '<recipients>' [subject 'msg subject'] - email a chart or notebook to the recipients """

        recipients = re.split(r"\s*,\s*", recipients)
        if email_object == "notebook":
            # Need to render and email the current notebook
            if notebook_path:
                notebook = os.path.basename(notebook_path)
                self.print(f"Emailing {notebook} to {recipients}")
                if subject is None:
                    subject = f"{notebook} notebook - Unify"
                self.email_helper.send_notebook(notebook_path, recipients, subject)
            else:
                self.print("Error, could not determine notebook name")
        elif email_object == "chart":
            if self.last_chart:                
                self.print(f"Emailing last chart to {recipients}")
                self.email_helper.send_chart(self.last_chart, recipients, subject)
            else:
                self.print("Error no recent chart available")
        else:
            if subject is None:
                subject = f"{email_object} contents - Unify"
            # Email the contents of a table
            df: pd.DataFrame = self._execute_duck(f"select * from {email_object}")
            if subject is None:
                subject = f"Unify - {email_object}"
            self.print(f"Emailing {email_object} to {recipients}")
            fname = email_object.replace(".", "_") + ".csv"
            self.email_helper.send_table(df, fname, recipients, subject)

    def export_table(self, adapter_ref, table_ref, file_ref, write_option=None):
        """ export <table> <adapter> <file> [append|overwrite] - export a table to a file """
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

    def peek_table(self, table_ref, line_count=20, build_stats=True, debug=False):
        # Get column weights and widths.
        schema, table_root = table_ref.split(".")
        store: UnifyDBStorageManager = UnifyDBStorageManager("meta", self.duck)
        cols = [ColumnIntel(name, attrs) for name, attrs in store.get_column_intels(schema, table_root)]
        if len(cols) == 0:
            if build_stats:
                self._analyze_columns(table_ref)
                return self.peek_table(table_ref, line_count=line_count, build_stats=False)
            else:
                self.print(f"Can't peek at {table_ref} because no column stats are available.")
                return

        cols = sorted(cols, key=lambda col: col.column_weight, reverse=True)
        # Take columns until our max width

        if debug:
            for c in cols:
                print(f"{c.name} - {c.column_weight} - {c.attrs}")

        use_cols = []
        total_width = 0
        date_used = False 

        def is_date(types):
            return "date" in types or "time" in types

        for col in cols:
            typest = col.attrs["type_str"]
            column_name = col.name
            display_name = col.name
            column_width = col.width
            if is_date(typest):
                if not date_used:
                    date_used = True
                    column_name = self.duck.get_short_date_cast(col.name)
                    display_name = col.name
                    column_width = 14
                else:
                    continue
            if typest == "string" and column_width > 50:
                column_name = f"substring({column_name}, 1, 50)"

            if typest == 'boolean':
                continue

            if column_width < len(display_name) and len(display_name) > 15:
                # column name will be longer than the values, so let's shorten it
                display_name = display_name[0:7] + "..." + display_name[-7:]

            if (total_width + max(column_width, len(display_name))) > 100:
                continue # keep adding smaller cols

            use_cols.append((column_name, display_name))
            total_width += max(column_width, len(display_name))

        col_list = ", ".join([f"{pair[0]} as \"{pair[1]}\"" for pair in use_cols])
        sql = f"select {col_list} from {table_ref} limit {line_count}"
        print(sql)
        return self._execute_duck(sql)

    def refresh_table(self, table_ref):
        """ refresh table <table> - updates the rows in a table from the source adapter """
        self.loader.refresh_table(table_ref)

    def reload_table(self, table_ref):
        """ reload table <table> - reloads the entire table from the source adapter """
        self._execute_duck(f"drop table {table_ref}")
        schema, table_root = table_ref.split(".")
        self.load_adapter_data(schema, table_root)

    def run_info(self, schedule_id):
        """ run info <notebook> - Shows details on the schedule for the indicated notebook """
        store: UnifyDBStorageManager = UnifyDBStorageManager("_system_", self.duck)
        schedule = store.get_object("schedules", schedule_id)
        if schedule:
            self.print("Schedule: ", schedule['run_at'], " repeat: ", schedule['repeater'])
            contents = schedule['contents']
            body = json.loads(contents)
            for cell in body['cells']:
                if 'source' in cell:
                    if isinstance(cell['source'], list):
                        for line in cell['source']:
                            self.print("| ", line)
                    else:
                        self.print("| ", cell['source'])
        else:
            self.print("No schedule found")

    def run_notebook_command(self, run_at_time: str, notebook_path: str, repeater: str=None):
        """ run [every day|week|month] at <date> <time> - Execute this notebook on a regular schedule """
        if notebook_path is None:
            self.print("Error, must supply a notebook name or full path")
            return
        contents = None
        if not os.path.exists(notebook_path):
            # Try to find the notebook in the Unify notebooks directory
            notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", notebook_path)

        if os.path.exists(notebook_path):
            # Jankily jam the whole notebook into the db so we can run it on the server
            contents = open(notebook_path, "r").read()
        else:
            raise RuntimeError(f"Cannot find notebook '{notebook_path}'")

        run_at_time = pd.to_datetime(run_at_time) # will assign the current date if no date
        schedule = {"notebook": notebook_path, "run_at": str(run_at_time), 
                    "repeater": repeater, "contents": contents}
        
        store: UnifyDBStorageManager = UnifyDBStorageManager("_system_", self.duck)
        # For now we are using the notebook name as the unique key. This means that a notebook
        # can have only one schedule. In the future we may allow to create multiple schedules.
        id = os.path.basename(notebook_path)
        store.put_object("schedules", id, schedule)
        notebook = os.path.basename(notebook_path)
        self.print(f"Scheduled to run notebook {notebook}")

    def run_schedule(self):
        """ run schedule - displays the current list of scheduled tasks """
        store: UnifyDBStorageManager = UnifyDBStorageManager("_system_", self.duck)
        items = store.list_objects("schedules")
        items = map(lambda row: [row[0], row[1]['notebook'], row[1]['run_at'], row[1]['repeater']], items)
        return pd.DataFrame(items, columns=["schedule_id", "notebook", "run_at", "repeat"])

    def delete_schedule(self, schedule_id):
        """ run delete <notebook> - Delete the schedule for the indicated notebook """
        store: UnifyDBStorageManager = UnifyDBStorageManager("_system_", self.duck)
        schedule = store.get_object("schedules", schedule_id)
        if schedule:
            store.delete_object("schedules", schedule_id)
            self.print(f"Deleted schedule {schedule_id} for notebook: ", schedule['notebook'])

    def set_variable(self, var_ref: str, var_expression: str):
        """ $<var> = <expr> - sets a variable. Use all caps for var to set a global variable. """
        is_global = var_ref.upper() == var_ref
        if not var_expression.lower().startswith("select "):
            # Need to evaluate the scalar expression
            val = self.duck.execute("select " + var_expression).fetchone()[0]
            self._save_variable(var_ref, val, is_global)
            self.print(val)
        else:
            val = self.duck.execute_df(var_expression)
            self._save_variable(var_ref, val, is_global)
            return val

    def _get_variable(self, name: str):
        if name.upper() == name:
            store: UnifyDBStorageManager =UnifyDBStorageManager(None, self.duck)
            return store.get_var(name)
        else:
            return self.session_vars[name]

    def _save_variable(self, name: str, value, is_global: bool):
        if is_global:
            # Save a scalar to our system table in meta
            store: UnifyDBStorageManager =UnifyDBStorageManager(None, self.duck)
            store.put_var(name, value)
        else:
            self.session_vars[name] = value

    def show_schemas(self):
        """ show schemas - list schemas in the datbase """
        return self._execute_duck(Queries.list_schemas)

    def show_tables(self, schema_ref=None):
        """ show tables [from <schema> [like '<expr>']] - list all tables or those from a schema """
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
        """ show columns [from <table> [like '<expr>']] - list all columns or those from a table """
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
        """ describe [<table>] - list all columns or those from a table """
        if table_ref is None:
            return self._execute_duck("describe")
        else:
            return self._execute_duck(f"describe {table_ref}")

    def create_statement(self):
        """ create table <table> ... """
        return self._execute_duck(self._cmd)

    def create_view_statement(self):
        """ create view <view> ... """
        return self._execute_duck(self._cmd)

    def insert_statement(self):
        """ insert into <table> ... """
        return self._execute_duck(self._cmd)

    def delete_statement(self):
        """ delete from <table> [where ...] """
        return self._execute_duck(self._cmd)

    def load_adapter_data(self, schema_name, table_name):
        if self.loader.materialize_table(schema_name, table_name):
            self.loader.create_views(schema_name, table_name)
            return True
        else:
            self.print("Loading table...")
            return False

    def select_query(self, fail_if_missing=False, **kwargs):
        """ select <columns> from <table> [where ...] [order by ...] [limit ...] [offset ...] """
        try:
            return self._execute_duck(self._cmd)

        except TableMissingException as e:
            if fail_if_missing:
                self.print(e)
                return
            schema, table_root = e.table.split(".")
            if self.load_adapter_data(schema, table_root):
                return self.select_query(fail_if_missing=True)

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
            store: UnifyDBStorageManager =UnifyDBStorageManager(None, self.duck)
            value = store.get_var(var_ref)
            if isinstance(value, pd.DataFrame):
                return value
            else:
                self.print(value)
        else:
            self.print(f"Error, unknown variable '{var_ref}'")

    def show_variables(self):
        """ show variables - list all defined variables"""
        rows = [(k, "[query result]" if isinstance(v, pd.DataFrame) else v) for k, v in self.session_vars.items()]
        store: UnifyDBStorageManager =UnifyDBStorageManager(None, self.duck)
        vars = [k[0][0:-len(store.VAR_NAME_SENTINAL)].upper() for k in store.list_vars()]
        rows.extend([(k, "[query result]") for k in vars])
        return pd.DataFrame(rows, columns=["variable", "value"])

    def clear_table(self, table_schema_ref=None):
        """ clear <table> - removes all rows from a table """
        self.loader.truncate_table(table_schema_ref)
        self.print("Table cleared: ", table_schema_ref)

    def old_create_chart_with_matplot(
        self, 
        chart_name=None, 
        chart_type=None, 
        chart_source=None, 
        chart_where=None,
        chart_params={}):
        """ create chart from <$var or table> as <chart_type> where x = <col> and y = <col> [opts] - see 'help charts' for info """
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

    def create_chart(
        self, 
        chart_name=None, 
        chart_type=None, 
        chart_source=None, 
        chart_params: dict={}):
        # Note of these renderers worked for both Jupyterlab and email
        # --> mimetype, notebook, html
        altair.renderers.enable('png')

        if "x" not in chart_params:
            self._last_result = pd.DataFrame({
                'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
                'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
            })
            chart_type = "bar_chart"
            chart_params["x"] = 'a'
            chart_params["y"] = 'b'           

        if chart_source:
            df = self._execute_duck(f"select * from {chart_source}")
        else:
            df = self._last_result

        if df is None or df.shape[0] == 0:
            raise RuntimeError("No recent query, or query returned no rows")

        title = chart_params.pop('title', '')

        chart_methods = {
            'bar_chart': 'mark_bar',
            'pie_chart': 'marc_arc',
            'hbar_chart': 'mark_bar',
            'line_chart': 'mark_line',
            'area_chart': 'mark_area'
        }
        # To make a horizontal bar chart we have to annotate the X axis value as the
        # "quantitative" value like: x="total:Q" rather than the "ordinal" value as "date:O".
        if chart_type not in chart_methods:
            raise RuntimeError(f"Uknown chart type '{chart_type}'")

        #chart_params["tooltip"] = {"content":"data"}
        
        chart = altair.Chart(df)
        chart = getattr(chart, chart_methods[chart_type])(tooltip=True). \
            encode(**chart_params). \
            properties(title=title, height=400, width=600)

        self.last_chart = chart
        return chart

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
                    if cmd.strip() == "":
                        continue
                    outputs, df = self.interpreter.run_command(cmd)
                    print("\n".join(outputs))
                    if isinstance(df, pd.DataFrame):
                        with pd.option_context('display.max_rows', None):
                            if df.shape[0] == 0:
                                continue
                            fmt_opts = {
                                "index": False,
                                "max_rows" : None,
                                "min_rows" : 10,
                                "max_colwidth": 50,
                                "header": True,
                                "float_format": '{:0.2f}'.format
                            }
                            if df.shape[0] > 40:
                                pydoc.pager(df.to_string(**fmt_opts))
                            else:
                                print(df.to_string(**fmt_opts))
                    elif df is not None:
                        print(df)
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

    interpreter = CommandInterpreter(silence_errors=silent)

    for i in range(len(sys.argv)):
        if sys.argv[i] == '-e':
            command = sys.argv[i+1]
            with pd.option_context('display.max_rows', None):
                lines, df = interpreter.run_command(command)
                print("\n".join(lines))
                if df is not None:
                    print(df)
            sys.exit(0)

    UnifyRepl(interpreter).loop()

