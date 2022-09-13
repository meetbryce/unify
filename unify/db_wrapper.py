from datetime import datetime, timedelta
from functools import lru_cache
import logging
import json
import os
import pickle
import re
import time

import pandas as pd
import pyarrow as pa
import duckdb
#import clickhouse_connect
from clickhouse_driver import Client
import clickhouse_driver

from .schemata import Queries
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class MyFilter:
    def filter(self, record):
        record.msg = "[clickhouse] " + record.msg
        return True

logger.addFilter(MyFilter())

class TableMissingException(RuntimeError):
    def __init__(self, table: str):
        super().__init__("Table {} does not exist".format(table))
        self.table = table

class QuerySyntaxException(RuntimeError):
    pass

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

    def get_table_root(self, table):
        if "." in table:
            return table.split(".")[-1]
        else:
            return table

    def get_short_date_cast(self, column):
        return f"strftime(CAST(\"{column}\" AS TIMESTAMP), '%m/%d/%y %H:%M')"

    def drop_schema(self, schema, cascade: bool=False):
        sql = f"drop schema {schema}"
        if cascade:
            sql += " cascade"
        return self.execute(sql)

    def dialect(self):
        return "postgres"

DATA_HOME = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_HOME, exist_ok=True)

class DuckDBWrapper(DBWrapper):
    DUCK_CONN = None
    REF_COUNTER = 0

    """ A cheap hack around DuckDB only usable in a single process. We just open/close
        each time to manage. Context manager for accessing the DuckDB database """

    def __init__(self):
        print(f"Connecting to local DuckDB database")

    def dialect(self):
        return "duckdb"

    def execute(self, query: str, args=[]):
        try:
            return DuckDBWrapper.DUCK_CONN.execute(query, args)
        except RuntimeError as e:
            m = re.search(r"Table with name (\S+) does not exist", str(e))
            if m:
                table_root = m.group(1)
                m = re.search("(\w+)\."+table_root, str(e))
                if m:
                    schema = m.group(1)
                else:
                    schema = ""
                raise TableMissingException(schema + "." + table_root)
            else:
                raise

    def execute_df(self, query: str, args=[]) -> pd.DataFrame:
        return self.execute(query, args).df()

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute_df("describe " + table)
        return rows["column_name"].values.tolist()

    def delete_rows(self, table, filter_values: dict=None, where_clause: str=None):
        if filter_values:
            query = f"delete from {table} where " + " and ".join([f"{key} = ?" for key in filter_values.keys()])
            query = self._substitute_args(query, filter_values.values())
        else:
            query = f"delete from {table} where {where_clause}"
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
        # Duck doesn't like the target name to be qualified
        dest_table_root = self.get_table_root(dest_table)

        self.execute(f"""
        BEGIN;
        DROP TABLE IF EXISTS {dest_table};
        ALTER TABLE {source_table} RENAME TO {dest_table_root};
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

from clickhouse_driver.columns.numpy.datetimecolumn import NumpyDateTimeColumnBase

def patched_apply_timezones_before_write(self, items):
    if isinstance(items, pd.DatetimeIndex):
        ts = items
    else:
        timezone = self.timezone if self.timezone else self.local_timezone
        try:
            ts = pd.to_datetime(items, utc=True).tz_localize(timezone)
        except (TypeError, ValueError):
            ts = pd.to_datetime(items, utc=True).tz_convert(timezone)

    ts = ts.tz_convert('UTC')
    return ts.tz_localize(None).to_numpy(self.datetime_dtype)

def monkeypatch_clickhouse_driver():
    NumpyDateTimeColumnBase.apply_timezones_before_write = patched_apply_timezones_before_write

monkeypatch_clickhouse_driver()

class ClickhouseWrapper(DBWrapper):
    SHARED_CLIENT = None

    def __init__(self):
        self.client = None
        print(f"Connecting to clickhouse database at: {os.environ['DATABASE_HOST']}")

    def dialect(self):
        return "clickhouse"

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

        logger.debug(query)
        try:
            return DBAPIResultFacade(self.client.execute(query))
        except clickhouse_driver.errors.ServerException as e:
            m = re.search(r"Table (\S+) doesn't exist.", str(e))
            if m:
                raise TableMissingException(m.group(1))
            else:
                logger.critical(str(e) + "\n" + f"While executing: {query}")
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

            logger.debug(query)
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
            if e.code == 60:
                m = re.search(r"Table (\S+) doesn't exist.", str(e))
                if m:
                    raise TableMissingException(m.group(1))
            elif e.code == 62:
                m = re.search(r"Syntax error[^.]+.", e.message)
                if m:
                    raise QuerySyntaxException(m.group(0))
            m = re.search(r"(^.*)Stack trace:", e.message)
            if m:
                e.message = m.group(1)
            raise e

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute_df("describe " + table)
        return rows["name"].values.tolist()

    def get_short_date_cast(self, column):
        return f"formatDateTime(CAST(\"{column}\" AS TIMESTAMP), '%m/%d/%y %H:%M')"

    def delete_rows(self, table, filter_values: dict=None, where_clause: str=None):
        if filter_values:
            query = f"alter table {table} delete where " + " and ".join([f"{key} = ?" for key in filter_values.keys()])
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

    def drop_schema(self, schema, cascade: bool=False):
        sql = f"drop database {schema}"
        if cascade:
            sql += " cascade"
        return self.execute(sql)

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
        if value.shape[0] > 0:
            self.client.insert_dataframe(
                f"INSERT INTO {schema}.{table_root} VALUES", 
                value, 
                settings={'use_numpy': True}
            )

    def append_dataframe_to_table(self, value: pd.DataFrame, schema: str, table_root: str):
        # There is a problem where a REST API returns a boolean column, but the first page 
        # of results is all nulls. In that case the type inference will have failed and we
        # will have defaulted to type the column as a string. We need to detect this case
        # and either coerce the bool column or fix the column type. For now we are doing
        # the former.

        # Use pyarrow for convenience, but type info probably already exists on the dataframe
        df_schema = pa.Schema.from_pandas(value)
        for col in df_schema.names:
            f = df_schema.field(col)
            if pa.types.is_boolean(f.type):
                # See if the table column is a string
                r = self.execute(Queries.list_columns.format(schema, table_root, col)).result
                if len(r) > 0 and r[0][1].lower() == "string":
                    # Corece bool values to string
                    value[col] = value[col].astype(str)
                    logger.critical("Coercing bool column {} to string".format(col))

        self.client.insert_dataframe(
            f"INSERT INTO {schema}.{table_root} VALUES", 
            value, 
            settings={'use_numpy': True}
        )

class UnifyDBStorageManager(StorageManager):
    """
        Stores adapter metadata in DuckDB. Creates a "meta" schema, and creates
        tables named <adapter schema>_<collection name>.
    """
    DUCK_TABLE_SCHEMA = "(id VARCHAR PRIMARY KEY, blob JSON)"
    TABLE_SCHEMA = {'*id': 'VARCHAR', 'blob': 'JSON'}
    LOG_TABLE_SCHEMA = {'id': 'VARCHAR', 'created': 'DATETIME', 'blob': 'JSON', '__order': ['id','created']}
    VAR_NAME_SENTINAL = '__var__'
    COLUMN_INTEL_TABLE_SCHEMA = {'schema_':'VARCHAR','table_':'VARCHAR','column_':'VARCHAR', 'attrs':'JSON',
                                '__order': ['schema_', 'table_', 'column_']}

    def __init__(self, adapter_schema: str, duck):
        self.adapter_schema = adapter_schema
        self.duck = duck
        self.duck.create_schema("meta")

    def ensure_col_table(self, duck, name) -> dict:
        table = self.adapter_schema + "_" + name
        duck.create_table("meta." + table, self.TABLE_SCHEMA)
        return table

    def ensure_log_table(self, duck, name) -> dict:
        table = self.adapter_schema + "_" + name
        duck.create_table("meta." + table, self.LOG_TABLE_SCHEMA)
        return table

    def create_var_storage_table(self, duck):
        table = "system__vars"
        duck.create_table("meta." + table, {'*name': 'VARCHAR', 'value': 'BLOB'})
        return table

    def put_var(self, name, value):
        table = self.create_var_storage_table(self.duck)
        self.duck.delete_rows("meta." + table, {"name": name})
        if isinstance(value, pd.DataFrame):
            name = name.lower() + self.VAR_NAME_SENTINAL
            self.duck.write_dataframe_as_table(value, "meta", name)
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
            return self.duck.execute_df(f"select * from meta.{name}")
            #raise RuntimeError(f"No value stored for global variable '{name}'")

    def put_object(self, collection: str, id: str, values: dict) -> None:
        table = self.ensure_col_table(self.duck, collection)
        self.duck.delete_rows("meta." + table, {"id": id})
        self.duck.execute(f"insert into meta.{table} (id, blob) values (?, ?)", [id, json.dumps(values)])

    def log_object(self, collection: str, id: str, values: dict) -> None:
        table = self.ensure_log_table(self.duck, collection)
        created = datetime.utcnow()
        self.duck.execute(f"insert into meta.{table} (id,created,blob) values (?, ?, ?)", [id, created, json.dumps(values)])

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

    @lru_cache(maxsize=500)
    def ensure_column_intel_table(self):
        table = "meta.column_intels"
        self.duck.create_table(table, self.COLUMN_INTEL_TABLE_SCHEMA)
        return table

    def insert_column_intel(self, schema: str, table_root: str, column: str, attrs: dict):
        meta_table = self.ensure_column_intel_table()
        self.duck.execute(f"insert into {meta_table} (schema_,table_,column_,attrs) values (?, ?, ?, ?)", 
            [schema, table_root, column, json.dumps(attrs)])

    def get_column_intels(self, schema: str, table_root: str):
        meta_table = self.ensure_column_intel_table()
        return [
            (col, json.loads(val)) for col, val in \
                self.duck.execute(
                    f"select column_, attrs from {meta_table} where schema_ = ? and table_ = ?",
                    [schema, table_root]
                ).fetchall()
        ]

    def delete_all_column_intel(self, schema, subject_table):
        meta_table = self.ensure_column_intel_table()
        self.duck.delete_rows(meta_table, {'schema_':schema, 'table_':subject_table})

    def delete_object(self, collection: str, id: str) -> bool:
        table = "meta." + self.ensure_col_table(self.duck, collection)
        r = self.duck.delete_rows(table, {'id': id})
        print("Delete resulted in: ", r)

    def delete_log_objects(self, collection: str, id: str):
        self.delete_object(collection, id)

    def list_objects(self, collection: str) -> list[tuple]:
        table = self.ensure_col_table(self.duck, collection)
        return [
            (key, json.loads(val)) for key, val in \
                self.duck.execute(f"select id, blob from meta.{table}").fetchall()
        ]

