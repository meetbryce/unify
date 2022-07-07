import os
import time
from threading import Thread
import cmd2
import glob
import pydoc
import re
import sys
import traceback
import yaml
from lark import Lark, Visitor
from lark.visitors import v_args
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
# These imports are required to unified schemas across parquest files
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa

from timeit import default_timer as timer

# DuckDB
import duckdb

from rest_schema import RESTCol, RESTTable, RESTAPISpec, Connector
from schemata import Queries
from parsing_utils import find_node_return_children

# Pandas setup
#pd.set_option('display.max_columns', None)

def ptime(label, f):
    start=timer(); 
    f() 
    end=timer(); 
    print(label, ": ", end-start)

DATA_HOME = "./data"
os.makedirs(DATA_HOME, exist_ok=True)


class TableScan(Thread):
    """
        Thread class for querying a REST API, by pages, and storing the results as 
        parquet files in the table's data directory.
    """
    def __init__(self, tableMgr=None, tableLoader=None, duck=None, select: list = None):
        Thread.__init__(self)
        self.tableMgr = tableMgr
        self.tableLoader = tableLoader
        self.duck = duck
        self.select = select

    def cleanup_df_page(self, df, cols_to_drop=[]):
        # Pandas json_normalize does not have a clever way to normalize embedded lists
        # of records, so it just returns a list of dicts. These will cause deserialization
        # issues if different REST response pages have different results for these
        # embedded lists. To work around that for now we are just removing those columns
        # from our results.
        # TODO: Convert the structured element into a JSON field

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


    def run(self):
        print("Running table scan for: ", self.tableMgr.name)

        def flush_rows(tableMgr, df, page, flush_count):
            if page <= page_flush_count:
                # First set of pages, so create the table
                self.duck.register('df1', row_buffer_df)
                self.duck.execute(f"create table {tableMgr.name} as select * from df1")
                # FIXME: The following shouldn't be neccessary, but it seems like `Duck.append`
                # does not work properly with qualified table names, so we hack around it
                # by setting the search path
                self.duck.execute(f"set search_path='{tableMgr.rest_spec.name}'")
            else:
                self.duck.append(tableMgr.table_spec.name, row_buffer_df)

        page = 1
        page_flush_count = 5 # flush 5 REST calls worth of data to the db
        row_buffer_df = None
        table_cols = set()

        for json_page, size_return in self.tableMgr.table_spec.query_resource(self.tableLoader):
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
                flush_rows(self.tableMgr, row_buffer_df, page, page_flush_count)
                if page == page_flush_count:
                    table_cols = set(row_buffer_df.columns.tolist())
                row_buffer_df = row_buffer_df[0:0] # clear flushed rows, but keep columns

            page += 1

        if row_buffer_df.shape[0] > 0:
            flush_rows(self.tableMgr, row_buffer_df, page, page_flush_count)
            
        print("Finished table scan for: ", self.tableMgr.name)


class TableMgr:
    def __init__(self, schema, rest_spec, table_spec, auth = None, params={}):
        self.schema = schema
        self.name = schema + "." + table_spec.name
        self.rest_spec = rest_spec
        self.table_spec = table_spec
        self.synchronous_scanning = True
        self.setup()

    def setup(self):
        self.data_dir = os.path.join(DATA_HOME, self.name)
        os.makedirs(self.data_dir, exist_ok=True)
 
    def has_data(self):
        return len(os.listdir(self.data_dir)) > 0

    def truncate(self, duck):
        for f in os.listdir(self.data_dir):
            os.remove(os.path.join(self.data_dir, f))
        # Remove table from duck db
        duck.execute(f"drop table {self.name}")

    def load_table(self, duck, waitForScan=False, tableLoader=None):
        # Spawn a thread to query the table source
        self.scan = TableScan(self, tableLoader=tableLoader, duck=duck, select=self.table_spec.select_list)
        if self.synchronous_scanning:
            self.scan.run()
        else:
            self.scan.start()
            if waitForScan:
                self.scan.join()

class TableLoader:
    """Master loader class. This class can load the data for any table, either by 
       querying the table from DuckDB or else by calling the underlying REST API.

       Tables can be in 4 states:
       1. Unknown (schema and table name never registered)
       2. Never queried. Call the REST API to query the data and store it to parquet.
       3. Not loaded. Data exists on disk, needs to be loaded as a view into Duck.
       4. Table view loaded in Duck.
    """
    def __init__(self, duck):
        self.duck = duck
        self.connections = Connector.setup_connections('./connections.yaml')
        self.tables = {}

        # Connections defines the set of schemas we will create in the database.
        # For each connection/schema then we will define the tables as defined
        # in the REST spec for the target system.
        for conn in self.connections:
            self.duck.execute(f"create schema if not exists {conn.schema_name}")
            for t in conn.spec.list_tables():
                tmgr = TableMgr(conn.schema_name, conn.spec, t)
                self.tables[tmgr.name] = tmgr

    def materialize_table(self, table):
        tmgr = self.tables[table]
        tmgr.load_table(self.duck, tableLoader=self)
        return tmgr.has_data()

    def read_table_rows(self, table):
        if not self.table_exists_in_db(table):
            tmgr = self.tables[table]
            tmgr.load_table(self.duck, tableLoader=self, waitForScan=True)

        if self.table_exists_in_db(table):
            r = self.duck.execute(f"select * from {table}")
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
        self.tables[table].truncate(self.duck)

    def table_exists_in_db(self, table):
        try:
            self.duck.execute(f"select 1 from {table}")
            return True
        except:
            return False

class RunCommand(Visitor):
    _last_result: pd.DataFrame = None

    def __init__(self, wide_display=False, read_only=False):
        #super().__init__(multiline_commands=['select'], persistent_history_file='/tmp/hist')
        self.debug = True
        try:
            self.duck = duckdb.connect(os.path.join(DATA_HOME, "duckdata"), read_only=read_only)
        except RuntimeError:
            print("Database is locked. Is there a write process running?")
            sys.exit(1)

        self.parser = Lark(open("grammar.lark").read())
        self.__output = sys.stdout

        if wide_display:
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)

        self.setup()

    def setup(self):
        self.loader = TableLoader(self.duck)

    def _list_schemas(self, match_prefix=None):
        all = sorted(list(r[0] for r in self.duck.execute(Queries.list_schemas).fetchall()))
        if match_prefix:
            all = [s[len(match_prefix):] for s in all if s.startswith(match_prefix)]
        return all
    
    def _list_tables_filtered(self, schema, table=None):
        conn = self.loader.lookup_connection(schema)
        table = table or ''
        return sorted(list(t.name[len(table):] for t in conn.list_tables() if t.name.startswith(table)))

    def _run_command(self, cmd, output_buffer=None, use_pager=True):
        save_output = self.__output
        try:
            self._allow_pager = use_pager
            if output_buffer:
                self.__output = output_buffer
            self._cmd = cmd
            parse_tree = self.parser.parse(self._cmd)
            #print(parse_tree.pretty())
            self.visit(parse_tree)
        finally:
            self.__output = save_output

    def loop(self):
        session = PromptSession(history=FileHistory(os.path.expanduser("~/.pphistory")))
        suggester = AutoSuggestFromHistory()
        try:
            while True:
                try:
                    cmd = session.prompt("> ", auto_suggest=suggester)
                    self._run_command(cmd)
                except Exception as e:
                    if isinstance(e, EOFError):
                        raise
                    traceback.print_exc(file=self.__output)
        except EOFError:
            sys.exit(0)

    def show_tables(self, tree):
        # FIXME: merge tables known us but not yet to Duck
        #    for table in sorted(self.loader.tables.keys()):
        #        print(table, file=self.__output)
        schema_ref = find_node_return_children("schema_ref", tree)
        print("tables\n----------", file=self.__output)
        if schema_ref:
            query = Queries.list_tables_filtered.format(schema_ref[0])
        else:
            query = Queries.list_tables

        self._execute_duck(query, header=False)
        return tree

    def show_schemas(self, tree):
        self._execute_duck(Queries.list_schemas)
        return tree

    def find_qualified_table_ref(self, tree):
        table = None
        sch_ref = find_node_return_children("table_schema_ref", tree)
        if sch_ref:
            schema = sch_ref[0]
            table = sch_ref[1]
            table = schema + "." + table
        else:
            table_ref = find_node_return_children("table_ref", tree)
            if table_ref:
                table = table_ref[0]
        return table

    def show_columns(self, tree):
        table = self.find_qualified_table_ref(tree)
        if table:
            self._execute_duck(f"describe {table}")
        else:
            self._execute_duck("describe")
        return tree

    def describe(self, tree):
        if len(tree.children) == 0:
            self._execute_duck("describe")
        elif tree.children[0].data.value == 'table_ref':
            table = tree.children[0].children[0].value
            self._execute_duck(f"describe {table}")
        else:
            schema = tree.children[0].children[0].value
            table = tree.children[0].children[1].value
            self._execute_duck(f"describe {schema}.{table}")

    def create_statement(self, tree=None):
        self._execute_duck(self._cmd)
        return tree

    def insert_statement(self, tree=None):
        self._execute_duck(self._cmd)
        return tree

    def delete_statement(self, tree=None):
        self._execute_duck(self._cmd)
        return tree

    def select_query(self, tree=None, fail_if_missing=False):
        try:
            self._execute_duck(self._cmd)

        except RuntimeError as e:
            if fail_if_missing:
                print(e, file=self.__output)
                return
            m = re.search("Table with name (\S+) does not exist", str(e))
            if m:
                table = m.group(1)
                # Extract the schema from the original query
                m = re.search(f"(\w+)\.{table}", self._cmd)
                if m:
                    table = m.group(1) + "." + table
                    if self.loader.materialize_table(table):
                        return self.select_query(self._cmd, fail_if_missing=True)
                    else:
                        print("Loading table...", file=self.__output)
                else:
                    print(e, file=self.__output)
            else:
                print(e, file=self.__output)

    def clear(self, tree):
        schema = tree.children[0].children[0].value
        _table = tree.children[0].children[1].value
        table = schema + "." + _table
        self.loader.truncate_table(table)
        print("Table cleared: ", table, file=self.__output)

    def _execute_duck(self, query, header=True):
        r = self.duck.execute(query)
        with pd.option_context('display.max_rows', None):
            df = r.df()
            self._last_result = df
            fmt_opts = {
                "index": False,
                "max_rows" : None,
                "min_rows" : 10,
                "max_colwidth": 50,
                "header": header
            }
            if df.shape[0] > 40 and self._allow_pager:
                pydoc.pager(df.to_string(**fmt_opts))
            else:
                print(df.to_string(**fmt_opts), file=self.__output)

    def do_set(self, args):
        self._execute_duck("set " + args)
        
    def do_create(self, args):
        self._execute_duck("create " + args)

    def do_drop(self, args):
        self._execute_duck("drop " + args)

    def do_pragma(self, args):
        self._execute_duck("PRAGMA " + args)

    def do_delete(self, args):
        self._execute_duck("delete " + args)
        
if __name__ == '__main__':
    RunCommand(read_only=True).loop()    

