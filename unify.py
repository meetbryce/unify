import os
import time
from threading import Thread
import cmd2
import glob
import pydoc
import re
import yaml

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

from rest_schema import RESTCol, RESTTable, RESTAPISpec, Connection

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

    def cleanup_df_page(self, df):
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

            # Caller can optionally specific a set of columns to keep. We keep any column with
            # either an exact match, or a parent property match ("name_??").
            if self.select:
                if not (f.name in self.select or \
                    any(sname for sname in self.select if re.match(sname + r"_.*", f.name))):
                    #print("Dropping column missing from 'select' clause: ", f.name)
                    df.drop(columns=f.name, inplace=True)


    def run(self):
        print("Running table scan for: ", self.tableMgr.name)

        page = 1
        for json_page, count in self.tableMgr.table_spec.query_resource(self.tableLoader):
            df = pd.json_normalize(json_page, sep='_')

            if df.size == 0:
                continue

            self.cleanup_df_page(df)

            fname = self.tableMgr.name + "_" + str(time.time()) + "_" + str(page)
            df.to_parquet(os.path.join(self.tableMgr.data_dir, fname))

            if page == 1:
                self.duck.register('df1', df)
                self.duck.execute(f"create table {self.tableMgr.name} as select * from df")
                # FIXME: The following shouldn't be neccessary, but it seems like `Duck.append`
                # does not work properly with qualified table names, so we hack around it
                # by setting the search path
                self.duck.execute(f"set search_path='{self.tableMgr.rest_spec.name}'")
            else:
                self.duck.append(self.tableMgr.table_spec.name, df)
            page += 1

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
        if self.has_data():
            # Loads the table data as a view into Duck from the parquet files
            #duck.execute(f"create view {self.name} as select * from read_parquet('{self.data_dir}/*')")
            return True
        else:
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
        self.connections = Connection.load_config('./connections.yaml')
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


    def truncate_table(self, table):
        self.tables[table].truncate(self.duck)

    def table_exists_in_db(self, table):
        try:
            self.duck.execute(f"select 1 from {table}")
            return True
        except:
            return False

class ReplApp(cmd2.Cmd):
    def __init__(self):
        super().__init__(multiline_commands=['select'], persistent_history_file='/tmp/hist')
        self.debug = True
        self.duck = duckdb.connect(os.path.join(DATA_HOME, "duckdata")) # memory store by default for now
        self.setup()

    def setup(self):
        self.loader = TableLoader(self.duck)


    def do_clear(self, args):
        table = args.args
        self.loader.truncate_table(table)
        print("Table cleared: ", table)

    def _execute_duck(self, query):
        r = self.duck.execute(query)
        with pd.option_context('display.max_rows', None):
            df = r.df()
            fmt_opts = {
                "index": False,
                "max_rows" : None,
                "min_rows" : 10,
                "max_cols": 0,
                "max_colwidth": 50,
                "show_dimensions": 'truncate',
                "line_width": 80                
            }
            if df.shape[0] > 40:
                pydoc.pager(df.to_string(**fmt_opts))
            else:
                print(df.to_string(**fmt_opts))


    def do_show(self, args):
        command = "show " + args
        if args == 'schemas':
            command = "select schema_name from information_schema.schemata"
        # FIXME: "show tables" should show tables that aren't loaded yet
        self._execute_duck(command)

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
        
    def do_describe(self, args):
        return self.do_select(args, command='describe')

    def do_select(self, query_remainder, fail_if_missing=False, command='select'):
        query = command + " " + query_remainder
        try:
            self._execute_duck(query)

        except RuntimeError as e:
            if fail_if_missing:
                print(e)
                return
            m = re.search("Table with name (\S+) does not exist", str(e))
            if m:
                table = m.group(1)
                # Extract the schema from the original query
                m = re.search(f"(\w+)\.{table}", query)
                if m:
                    table = m.group(1) + "." + table
                    if self.loader.materialize_table(table):
                        return self.do_select(query_remainder, fail_if_missing=True)
                    else:
                        print("Loading table...")
                else:
                    print(e)
            else:
                print(e)

def main():
    import sys
    c = ReplApp()
    sys.exit(c.cmdloop())

if __name__ == '__main__':
    main()

