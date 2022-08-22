import logging
import os
import re
import time

import pandas as pd
import pyarrow as pa
import duckdb
#import clickhouse_connect
from clickhouse_driver import Client
import clickhouse_driver

logger = logging.getLogger(__name__)
class MyFilter:
    def filter(self, record):
        record.msg = "[clickhouse] " + record.msg
        return True

logger.addFilter(MyFilter())

class TableMissingException(RuntimeError):
    def __init__(self, table: str):
        self.table = table

class DBWrapper:
    def execute(self, query: str, args: list = []) -> pd.DataFrame:
        return None

    def _substitute_args(self, query: str, args: tuple):
        # FIXME: Need to properly escape args for single quotes
        args = list(args)
        def repl(match):
            val = args.pop(0)
            if isinstance(val, str):
                return "'{}'".format(val)
            else:
                return val
        return re.sub("\\?", repl, query)

    def current_date_expr(self):
        return "current_date"

    def create_memory_table(self, table: str, df: pd.DataFrame):
        """ Create an in-memory table from the given dataframe. Used for processing lightweight results."""
        pass

    def table_exists(self, table) -> bool:
        pass

    def write_dataframe_as_table(self, value: pd.DataFrame, schema: str, table_root: str):
        pass

    def append_dataframe_to_table(self, value: pd.DataFrame, schema: str, table_root: str):
        pass

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        breakpoint()
        rows = self.execute_df("describe " + table)
        rows
        return rows["column_name"].values.tolist()

    def delete_rows(self, table, filter_values: dict, where_clause: str=None):
        pass

DATA_HOME = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_HOME, exist_ok=True)

class DuckDBWrapper(DBWrapper):
    DUCK_CONN = None
    REF_COUNTER = 0

    """ A cheap hack around DuckDB only usable in a single process. We just open/close
        each time to manage. Context manager for accessing the DuckDB database """

    def __init__(self):
        pass

    def execute(self, query: str, args=[]):
        try:
            return DuckDBWrapper.DUCK_CONN.execute(query, args)
        except RuntimeError as e:
            m = re.search(r"Table with name (\S+) does not exist", str(e))
            if m:
                raise TableMissingException(m.group(1))
            else:
                raise

    def execute_df(self, query: str, args=[]) -> pd.DataFrame:
        return self.execute(query, args).df()

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute_df("describe " + table)
        return rows["column_name"].values.tolist()

    def delete_rows(self, table, filter_values: dict, where_clause: str=None):
        query = f"delete from {table} where " + ",".join([f"{key} = ?" for key in filter_values.keys()])
        query = self._substitute_args(query, filter_values.values())
        self.execute(query)

    def create_schema(self, schema):
        return self.execute(f"create schema if not exists {schema}")

    def create_table(self, table: str, columns: dict):
        new_cols = {}
        for name, type in columns.items():
            if name.startswith("*"):
                name = name[1:]
                type += " PRIMARY KEY"
            elif name.startswith('__'):
                continue
            new_cols[name] = type

        self.execute(
            f"create table if not exists {table} (" + ",".join(["{} {}".format(n, t) for n, t in new_cols.items()]) + ")"
        )

    def create_memory_table(self, table_root: str, df: pd.DataFrame):
        """ Create an in-memory table from the given dataframe. Used for processing lightweight results."""
        if "." in table_root:
            raise RuntimeError("Memory tables cannot specify a schema")
        DuckDBWrapper.DUCK_CONN.register(table_root, df)
        return table_root

    def drop_memory_table(self, table_root: str):
        DuckDBWrapper.DUCK_CONN.unregister(table_root)

    def write_dataframe_as_table(self, value: pd.DataFrame, schema: str, table_root: str):
        DuckDBWrapper.DUCK_CONN.register('df1', value)
        # create the table AND flush current row_buffer values to the db            
        DuckDBWrapper.DUCK_CONN.execute(f"create or replace table {schema}.{table_root} as select * from df1")
        DuckDBWrapper.DUCK_CONN.unregister("df1")

    def append_dataframe_to_table(self, value: pd.DataFrame, schema: str, table_root: str):
        # FIXME: we should probably use a context manager at the caller to ensure
        # we set and unset the search_path properly
        DuckDBWrapper.DUCK_CONN.execute(f"set search_path='{schema}'")
        DuckDBWrapper.DUCK_CONN.append(table_root, value)

    def table_exists(self, table):
        try:
            DuckDBWrapper.DUCK_CONN.execute(f"describe {table}")
            return True
        except Exception as e:
            if 'Catalog Error' in str(e):
                return False
            else:
                raise

    def replace_table(self, source_table: str, dest_table: str):
        self.execute(f"""
        BEGIN;
        DROP TABLE IF EXISTS {dest_table};
        ALTER TABLE {source_table} RENAME TO {dest_table};
        COMMIT;
        """)

    def close(self):
        pass

    def is_closed(self) -> bool:
        return DuckDBWrapper.DUCK_CONN is None

    def __enter__(self):
        DuckDBWrapper.REF_COUNTER += 1
        if DuckDBWrapper.DUCK_CONN is None:
            DuckDBWrapper.DUCK_CONN = duckdb.connect(os.path.join(DATA_HOME, "duckdata"), read_only=False)
            DuckDBWrapper.DUCK_CONN.execute("PRAGMA log_query_path='/tmp/duckdb_log'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        DuckDBWrapper.REF_COUNTER -= 1
        if DuckDBWrapper.REF_COUNTER == 0 and self.__class__.DUCK_CONN is not None:
            try:
                DuckDBWrapper.DUCK_CONN.execute("COMMIT")
            except:
                pass
            DuckDBWrapper.DUCK_CONN.close()
            DuckDBWrapper.DUCK_CONN = None

class DBAPIResultFacade:
    def __init__(self, result):
        self.result = result

    def fetchall(self):
        return list(self.result)

    def fetchone(self):
        res = None
        for row in self.result:
            return row

    def fetchmany(self, n):
        rows = []
        for row in self.result:
            rows.append(row)
            if len(rows) == n:
                return rows
        return rows

class ClickhouseWrapper(DBWrapper):
    SHARED_CLIENT = None

    def __init__(self):
        self.client = None

    def __enter__(self):
        if 'DATABASE_HOST' not in os.environ:
            raise RuntimeError("DATABASE_HOST not set")
        if 'DATABASE_USER' not in os.environ:
            raise RuntimeError("DATABASE_USER not set")
        if 'DATABASE_PASSWORD' not in os.environ:
            raise RuntimeError("DATABASE_PASSWORD not set")

        settings = {'allow_experimental_object_type': 1, 'allow_experimental_lightweight_delete': 1}
        self.client: Client = Client(
            host=os.environ['DATABASE_HOST'], 
            user=os.environ['DATABASE_USER'],
            password=os.environ['DATABASE_PASSWORD'],
            settings=settings
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # let context manager going out of scope release the client
        self.client.disconnect()

    def current_date_expr(self):
        return "today()"

    def table_exists(self, table) -> bool:
        return self.client.execute(f"EXISTS {table}")[0] == 1

    def execute(self, query: str, args=[]):
        if query.strip().lower().startswith("insert"):
            return self._execute_insert(query, args)
        if args:
            query = self._substitute_args(query, args)

        try:
            return DBAPIResultFacade(self.client.execute(query))
        except clickhouse_driver.errors.ServerException as e:
            m = re.search(r"Table (\S+) doesn't exist.", str(e))
            if m:
                raise TableMissingException(m.group(1))
            else:
                raise


    def _execute_insert(self, query, args=[]):
        # Clickhouse client has a weird syntax for inserts, where you leave
        # a dangling 'VALUES' clause in the SQL, and then provide arguments
        # a list of dictionaries matching the insert columns

        match = re.search(r"into ([\w\._]+)\s*\(([^\)]+)\)", query, re.IGNORECASE)
        if match:
            table = match.group(1)
            col_names = re.split(r'\s*,\s*', match.group(2))
            query = re.sub(r'\s*values\s*.*$', ' ', query, flags=re.IGNORECASE) # strip values clause
            query += " VALUES"
            args = [dict(zip(col_names, args))]

            return self.client.execute(query, args)
        else:
            raise RuntimeError("Cannot parse insert query, did you specify the column list?: " + query)

    def execute_get_json(self, query: str, args=[]):
        if args:
            query = self._substitute_args(query, args)
        return DBAPIResultFacade(self.client.execute(query))
        #result, columns = self.client.execute(query, with_column_types=True)
        #df=pd.DataFrame(result,columns=[tuple[0] for tuple in columns])
        #return df.to_json(orient='records')

    def execute_df(self, query: str) -> pd.DataFrame:
        try:
            return self.client.query_dataframe(query)
        except clickhouse_driver.errors.ServerException as e:
            m = re.search(r"Table (\S+) doesn't exist.", str(e))
            if m:
                raise TableMissingException(m.group(1))
            else:
                raise

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute_df("describe " + table)
        return rows["name"].values.tolist()

    def delete_rows(self, table, filter_values: dict=None, where_clause: str=None):
        if filter_values:
            query = f"alter table {table} delete where " + ",".join([f"{key} = ?" for key in filter_values.keys()])
            query = self._substitute_args(query, filter_values.values())
        elif where_clause:
            query = f"alter table {table} delete where {where_clause}"
        res = self.execute(query).fetchall()
        # Ugh. Seems deletes have some delay to show up...
        time.sleep(0.1)

    def create_schema(self, schema):
        query = f"CREATE DATABASE IF NOT EXISTS {schema}"
        return self.client.execute(query)

    def create_table(self, table: str, columns: dict):
        new_cols = {}
        primary_key = ''
        ordering = ''
        for name, type in columns.items():
            if name.startswith("*"):
                name = name[1:]
                primary_key = f"PRIMARY KEY ({name})"
            elif name == '__order':
                ordering = "ORDER BY ({})".format(",".join(type))
                primary_key = ""
                continue # skip this column
            if type == 'JSON':
                type = 'VARCHAR' # JSON type isnt working well yet
            new_cols[name] = type

        table_ddl = \
            f"create table if not exists {table} (" + ",".join(["{} {}".format(n, t) for n, t in new_cols.items()]) + ")" + \
                f" ENGINE = MergeTree() {primary_key} {ordering}"
        self.execute(table_ddl)

    def replace_table(self, source_table: str, dest_table: str):
        self.execute(f"EXCHANGE TABLES {source_table} AND {dest_table}")
        self.execute(f"DROP TABLE {source_table}")

    def create_memory_table(self, table_root: str, df: pd.DataFrame):
        """ Create an in-memory table from the given dataframe. Used for processing lightweight results."""
        if "." in table_root:
            raise RuntimeError("Memory tables cannot specify a schema")
        self.write_dataframe_as_table(df, "default", table_root, table_engine="Memory")
        return "default." + table_root

    def drop_memory_table(self, table_root: str):
        self.execute(f"DROP TABLE IF EXISTS default.{table_root}")

    def _infer_df_columns(self, df: pd.DataFrame):
        schema = pa.Schema.from_pandas(df)
        col_specs = {}
        for col in schema.names:
            f = schema.field(col)
            if pa.types.is_boolean(f.type):
                col_specs[col] = "UInt8"
            elif pa.types.is_integer(f.type):
                col_specs[col] = "Int64"
            elif pa.types.is_floating(f.type):
                col_specs[col] = "Float64"
            elif pa.types.is_string(f.type):
                col_specs[col] = "varchar"
            elif pa.types.is_date(f.type):
                col_specs[col] = "Date"
            elif pa.types.is_timestamp(f.type):
                col_specs[col] = "DateTime"
            elif pa.types.is_null(f.type):
                # No data is present to guess the type, so just use string
                col_specs[col] = "String"
            else:
                raise RuntimeError(f"Unknown type for dataframe column {col}: ", f.type)
        return col_specs, schema.names[0]

    def write_dataframe_as_table(self, value: pd.DataFrame, schema: str, table_root: str, table_engine: str="MergeTree"):
        col_specs, primary_key = self._infer_df_columns(value)
        if primary_key:
            primary_key = f"PRIMARY KEY {primary_key}"
        if table_engine == "Memory":
            primary_key = ""

        self.client.execute(f"drop table if exists {schema}.{table_root}")

        sql = f"create table {schema}.{table_root} (" + \
            ", ".join([f"{col} {ctype}" for col, ctype in col_specs.items()]) + \
                f") Engine={table_engine}() {primary_key}"
        logger.debug(sql)
        self.client.execute(sql)

        logger.debug("Writing dataframe to table")
        self.client.insert_dataframe(
            f"INSERT INTO {schema}.{table_root} VALUES", 
            value, 
            settings={'use_numpy': True}
        )

    def append_dataframe_to_table(self, value: pd.DataFrame, schema: str, table_root: str):
        self.client.insert_dataframe(
            f"INSERT INTO {schema}.{table_root} VALUES", 
            value, 
            settings={'use_numpy': True}
        )
