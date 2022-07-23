import base64
from email.parser import Parser
from inspect import isfunction
import io
import os
from threading import Thread
import pydoc
import re
import sys
import traceback
from lark import Lark, Visitor
from lark.visitors import v_args
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import urllib
from typing import Dict
# CHARTING
from matplotlib import ticker
import matplotlib.pyplot as plt

import pandas as pd
import pyarrow as pa

from timeit import default_timer as timer

# DuckDB
import duckdb

from rest_schema import Adapter, Connection, StorageManager
from schemata import LoadTableRequest, Queries
from parsing_utils import (
    find_subtree, 
    find_node_return_child, 
    find_node_return_children,
    collect_child_strings
)

DATA_HOME = "./data"
os.makedirs(DATA_HOME, exist_ok=True)

class DuckContext:
    """ A cheap hack around DuckDB only usable in a single process. We just open/close
        each time to manage. Context manager for accessing the DuckDB database """
    def __init__(self):
        pass

    def __enter__(self):
        self.duck = duckdb.connect(os.path.join(DATA_HOME, "duckdata"), read_only=False)
        return self.duck

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duck.close()
        

class TableScan(Thread):
    """
        Thread class for querying a REST API, by pages, and storing the results as 
        parquet files in the table's data directory.
    """
    def __init__(self, tableMgr=None, tableLoader=None, select: list = None):
        Thread.__init__(self)
        self.tableMgr = tableMgr
        self.tableLoader = tableLoader
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
        with DuckContext() as duck:
            self.perform_scan(duck)

    def perform_scan(self, duck):
        print("Running table scan for: ", self.tableMgr.name)

        def flush_rows(tableMgr, df, page, flush_count):
            if page <= page_flush_count:
                # First set of pages, so create the table
                duck.register('df1', row_buffer_df)
                duck.execute(f"create table {tableMgr.name} as select * from df1")
                # FIXME: The following shouldn't be neccessary, but it seems like `Duck.append`
                # does not work properly with qualified table names, so we hack around it
                # by setting the search path
                duck.execute(f"set search_path='{tableMgr.rest_spec.name}'")
            else:
                duck.append(tableMgr.table_spec.name, row_buffer_df)

        page = 1
        page_flush_count = 5 # flush 5 REST calls worth of data to the db
        row_buffer_df = None
        table_cols = set()

        for json_page, size_return in self.tableMgr.table_spec.query_resource(self.tableLoader):
            record_path = self.tableMgr.table_spec.result_body_path
            breakpoint()
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

    def has_data(self):
        return True

    def truncate(self, duck):
        # Remove table from duck db
        duck.execute(f"drop table {self.name}")

    def load_table(self, waitForScan=False, tableLoader=None):
        # Spawn a thread to query the table source
        self.scan = TableScan(self, tableLoader=tableLoader, select=self.table_spec.select_list)
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
       2. Never queried. Call the Adapter to query the data and store it to local db.
       3. Table exists in the local db.
    """
    def __init__(self):
        self.storage_manager = BasicStorageManager()
        self.connections = Connection.setup_connections(
            './connections.yaml', 
            storage_manager=self.storage_manager
        )
        self.tables = {}

        self.adapters: dict[str, Adapter] = dict(
            [(conn.schema_name, conn.adapter) for conn in self.connections if conn.adapter.supports_commands()]
        )
        # Connections defines the set of schemas we will create in the database.
        # For each connection/schema then we will define the tables as defined
        # in the REST spec for the target system.
        with DuckContext() as duck:
            for conn in self.connections:
                duck.execute(f"create schema if not exists {conn.schema_name}")
                for t in conn.adapter.list_tables():
                    tmgr = TableMgr(conn.schema_name, conn.adapter, t)
                    self.tables[tmgr.name] = tmgr

    def materialize_table(self, schema, table):
        with DuckContext() as duck:
            qual = schema + "." + table
            if qual in self.tables:
                tmgr = self.tables[table]
            else:
                table_spec = self.adapters[schema].lookupTable(table)
                tmgr = TableMgr(schema, self.adapters[schema], table_spec)
                self.tables[tmgr.name] = tmgr           
            tmgr.load_table(duck, tableLoader=self)
            return tmgr.has_data()

    def read_table_rows(self, table):
        with DuckContext() as duck:
            if not self.table_exists_in_db(table):
                tmgr = self.tables[table]
                tmgr.load_table(duck, tableLoader=self, waitForScan=True)

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
        except:
            return False

class BasicStorageManager(StorageManager):
    def __init__(self):
        self.collections = {}

    def get_col(self, name) -> dict:
        if name not in self.collections:
            self.collections[name] = {}
        return self.collections[name]

    def put_object(self, collection: str, id: str, values: dict) -> None:
        self.get_col(collection)[id] = values

    def get_object(self, collection: str, id: str) -> dict:
        return self.get_col(collection)[id]

    def delete_object(self, collection: str, id: str) -> bool:
        col = self.get_col(collection)
        if id in col:
            del col[id]
            return True
        else:
            return False
            
    def list_objects(self, collection: str) -> list[tuple]:
        return [(key, val) for key, val in self.get_col(collection).items()]

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

    def perform_new_visit(self, parse_tree):
        self._the_command = None
        self._the_command_args = {}
        self.visit(parse_tree)
        return self._the_command

    def show_tables(self, tree):
        self._the_command = 'show_tables'
        self._the_command_args['schema_ref'] = find_node_return_child("schema_ref", tree)
        return tree

    def show_schemas(self, tree):
        self._the_command = 'show_schemas'
        return tree

    def show_columns(self, tree):
        """" Always returns 'table_ref' either qualified or unqualified by the schema name."""
        self._the_command = 'show_columns'
        table = find_node_return_children("table_schema_ref", tree)
        if table:
            table = ".".join(table)
        else:
            table = find_node_return_child("table_ref", tree)
        self._the_command_args['table_ref'] = table
        filter = collect_child_strings("column_filter", tree)
        if filter:
            self._the_command_args['column_filter'] = filter.strip()
        return tree

    def describe(self, tree):
        self._the_command = 'describe'
        if len(tree.children) == 0:
            self._the_command_args['table_ref'] = None
        elif tree.children[0].data.value == 'table_ref':
            table = tree.children[0].children[0].value
            self._the_command_args['table_ref'] = table
        else:
            schema = tree.children[0].children[0].value
            table = tree.children[0].children[1].value
            self._the_command_args['table_ref'] = f"{schema}.{table}"
        return tree

    def select_query(self, tree):
        self._the_command = 'select_query'
    
    def create_statement(self, tree):
        self._the_command = 'create_statement'

    def create_view_statement(self, tree):
        self._the_command = 'create_view_statement'

    def delete_statement(self, tree):
        self._the_command = 'delete_statement'

    def drop_schema(self, tree):
        self._the_command = "drop_schema"
        self._the_command_args["schema_ref"] = find_node_return_child("schema_ref", tree)

    def insert_statement(self, tree):
        self._the_command = 'insert_statement'

    def clear_table(self, tree):
        self._the_command = 'clear_table'
        self._the_command_args['table_schema_ref'] = find_node_return_children("table_schema_ref", tree)
        if self._the_command_args['table_schema_ref']:
            self._the_command_args['table_schema_ref'] = ".".join(self._the_command_args['table_schema_ref'])
        return tree

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
            print(child)
            key = key or find_node_return_child("chart_param", child)
            value = value or find_node_return_child("param_value", child)
            if value and value[0] == '"':
                value = value[1:-1]
            if value and value[-1] == '"':
                value = value[:-1]
            if key and value:
                params[key] = value
                key = value = None
        self._the_command_args['chart_params'] = params

class RunCommand:
    _last_result: pd.DataFrame = None

    def __init__(self, wide_display=False, read_only=False):
        self.debug = True
        try:
            #self.duck = duckdb.connect(os.path.join(DATA_HOME, "duckdata"), read_only=read_only)
            pass
        except RuntimeError:
            print("Database is locked. Is there a write process running?")
            sys.exit(1)

        self.parser = Lark(open("grammar.lark").read())
        self.__output = sys.stdout
        self.parser_visitor = ParserVisitor()
        self.loader = TableLoader()
        self.adapters: dict[str, Adapter] = self.loader.adapters

        if wide_display:
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)

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

    def pre_handle_command(self, code, output_buffer: io.TextIOBase):
        m = re.match(r"\s*([\w_0-9]+)\s+(.*$)", code)
        if m:
            first_word = m.group(1)
            rest_of_command = m.group(2)
            if first_word in self.adapters:
                handler: Adapter = self.adapters[first_word]
                handler.run_command(rest_of_command, output_buffer)
                return True


    def _run_command(self, cmd, output_buffer=None, use_pager=True):
        save_output = self.__output    
        try:
            # Allow Adapters to process their special commands
            pre_result = self.pre_handle_command(cmd, output_buffer)
            if isinstance(pre_result, LoadTableRequest):
                self.load_adapter_data(pre_result.schema_name, pre_result.table_name)
                return {"response_type": "stream"}
            elif pre_result:
                return {"response_type": "stream"}

            with DuckContext() as duck:
                self.duck = duck
                self._allow_pager = use_pager
                if output_buffer:
                    self.__output = output_buffer
                self._cmd = cmd
                parse_tree = self.parser.parse(self._cmd)
                command = self.parser_visitor.perform_new_visit(parse_tree)
                if command:
                    result = getattr(self, command)(**self.parser_visitor._the_command_args)
                    if result:
                        return result
                    else:
                        return {"response_type": "stream"}
        finally:
            self.__output = save_output
            self.duck = None

    def loop(self):
        session = PromptSession(history=FileHistory(os.path.expanduser("~/.pphistory")))
        suggester = AutoSuggestFromHistory()
        try:
            while True:
                try:
                    cmd = session.prompt("> ", auto_suggest=suggester)
                    self._run_command(cmd)
                except RuntimeError as e:
                    print(e)
                except Exception as e:
                    if isinstance(e, EOFError):
                        raise
                    traceback.print_exc(file=self.__output)
        except EOFError:
            sys.exit(0)

    def _execute_duck(self, query, header=True):
        r = self.duck.execute(query)
        with pd.option_context('display.max_rows', None):
            df = r.df()
            if df.shape[0] == 0:
                return
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

    ################
    ## Commands 
    ################
    def show_schemas(self):
        self._execute_duck(Queries.list_schemas)

    def drop_schema(self, schema_ref):
        print("Are you sure you want to drop the schema '{schema_ref}' (y/n)?", end=' ')
        val = input()
        if val == "y":
            self._execute_duck(self._cmd)
            
    def show_tables(self, schema_ref=None):
        # FIXME: merge tables known us but not yet to Duck
        #    for table in sorted(self.loader.tables.keys()):
        #        print(table, file=self.__output)
        print("tables\n--------------", file=self.__output)
        if schema_ref:
            if schema_ref in self.adapters:
                for tableDef in self.adapters[schema_ref].list_tables():
                    print(tableDef.name)

            query = Queries.list_tables_filtered.format(schema_ref)
        else:
            query = Queries.list_tables

        self._execute_duck(query, header=False)

    def show_columns(self, table_ref, column_filter=None):
        if table_ref:
            if column_filter:
                parts = table_ref.split(".")
                column_filter = re.sub(r"\*", "%", column_filter)
                self._execute_duck(Queries.list_columns.format(parts[0], parts[1], column_filter))
            else:
                self._execute_duck(f"describe {table_ref}")
        else:
            self._execute_duck("describe")

    def describe(self, table_ref):
        if table_ref is None:
            self._execute_duck("describe")
        else:
            self._execute_duck(f"describe {table_ref}")

    def create_statement(self):
        self._execute_duck(self._cmd)

    def create_view_statement(self):
        self._execute_duck(self._cmd)

    def insert_statement(self):
        self._execute_duck(self._cmd)

    def delete_statement(self):
        self._execute_duck(self._cmd)

    def load_adapter_data(self, schema_name, table_name):
        if self.loader.materialize_table(schema_name, table_name):
            return True
        else:
            print("Loading table...", file=self.__output)
            return False

    def select_query(self, fail_if_missing=False):
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
                    if self.load_adapter_data(m.group(1), table):
                        return self.select_query(fail_if_missing=True)
                else:
                    print(e, file=self.__output)
            else:
                print(e, file=self.__output)

    def clear_table(self, table_schema_ref=None):
        self.loader.truncate_table(table_schema_ref)
        print("Table cleared: ", table_schema_ref, file=self.__output)

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
        self.__output.write(urllib.parse.quote(
            base64.b64encode(imgdata.getvalue())))
        return {"response_type": "display_data"}


if __name__ == '__main__':
    RunCommand(read_only=True).loop()    

