from __future__ import annotations
import contextlib
from datetime import datetime, timedelta
from functools import lru_cache
import logging
import json
import os
import pickle
import re
import time
import typing
from typing import Union, Any
import uuid
from venv import create

import pandas as pd
import pyarrow as pa
import duckdb
from clickhouse_driver import Client
import clickhouse_driver
import sqlglot
import enum
from sqlalchemy import Enum
from sqlalchemy import Column, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import Session
from sqlalchemy import UniqueConstraint
from clickhouse_sqlalchemy import engines as clickhouse_engines
from signaling import Signal

from .schemata import Queries
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class MyFilter:
    def filter(self, record):
        record.msg = "[clickhouse] " + record.msg
        return True

logger.addFilter(MyFilter())
UNIFY_META_SCHEMA = 'unify_schema'

class TableMissingException(RuntimeError):
    def __init__(self, table: str):
        super().__init__("Table {} does not exist".format(table))
        self.table = table

class QuerySyntaxException(RuntimeError):
    pass

class TableHandle:
    """ Manages a schema qualified or unqualified table reference, and supports the
        'user' side table string vs the 'real' db side table string.

        'table_opts' property allows us to pass additional table metadata when operating on 
        a table., This info will get stored in the Unify information_schema.
    """
    def __init__(self, table_root: str, schema: str=None, table_opts={}):
        if schema is not None:
            self._schema = schema
            self._table = table_root
            if "." in table_root:
                raise RuntimeError(f"Cannot provide qualified table {table_root} and schema {schema}")
        else:
            if "." not in table_root:
                raise RuntimeError(f"Unqualified table {table_root} provided but no schema")
            self._schema, self._table = table_root.split(".")
        self._table_opts = table_opts

    def table_root(self):
        return self._table

    def schema(self):
        return self._schema

    def real_table_root(self):
        return self._table

    def real_schema(self):
        return self._schema

    def real_table(self):
        return self.real_schema() + "." + self.real_table_root()

    def table_opts(self):
        return self._table_opts

    def __str__(self) -> str:
        return self.schema() + "." + self.table_root()

    def __repr__(self) -> str:
        return "TableHandle(" + str(self) + ")"

# 
# The DBManager class tree implements the interface to our warehouse database.
#
# Unlike the typical DBAPI pattern, we generally don't rely on taking raw SQL
# statements as our underlying engines (DuckDB, Clickhouse) support very different
# syntaxes for everything but the most core ANSI SQL statements. So instead we
# provide explicit interfaces for CREATE TABLE, DROP TABLE, SHOW TABLES, etc...
#
# This also works because the Unify engine is already parsing the user's SQL
# statements and is interpreting/adjusting many of them.
#
# DBManager also defines an interface for reading and writing DataFrames into the 
# database.
#
# **Information schema**
#
# The DBManager will create an "user space" information_schema database in the user's
# database, and will keep this schema populated with meta information about the
# user's tables and schemas. These tables are created by a set of SQLAlchemy models.
# The DBManager implements a `signals` system where entity creation/deletion produce
# signals that we observe and use to keep the meta information up to date. This system
# is mostly implemented in the DBManager base class, but underlying engines need
# to generate signals at the right times.
#

class DBManager(contextlib.AbstractContextManager):
    # 
    # Schemata signals
    #
    def __init__(self) -> None:
        super().__init__()
        self.signals = {
            "schema_create": Signal(args=['schema']),
            "schema_drop": Signal(args=['schema']),
            "table_create": Signal(args=['table']),
            "table_drop": Signal(args=['table'])
        }
        self.signals["schema_create"].connect(self._on_schema_create)
        self.signals["schema_drop"].connect(self._on_schema_drop)
        self.signals["table_create"].connect(self._on_table_create)
        self.signals["table_drop"].connect(self._on_table_drop)
        self.engine: Unknown = None

    def _on_schema_create(self, schema):
        session = Session(bind=self.engine)
        session.query(Schemata).filter(Schemata.name == schema).delete()
        session.add(Schemata(name=schema, type="schema"))
        session.commit()

    def _on_schema_drop(self, schema):
        session = Session(bind=self.engine)
        session.query(Schemata).filter(Schemata.name == schema).delete()
        session.commit()

    def _on_table_create(self, **kwargs):
        table = kwargs['table']
        session = Session(bind=self.engine)
        session.query(SchemataTable).filter(
            SchemataTable.table_name == table.table_root(),
            SchemataTable.table_schema == table.schema()
        ).delete()
        session.commit()
        t = SchemataTable(table_name=table.table_root(), table_schema=table.schema())
        opts = table.table_opts()
        for key in ['description', 'source', 'connection']:
            if key in opts:
                setattr(t, key, opts[key])
        session.add(t)
        session.commit()
        #self.engine.execute("COMMIT")
        session.expire_all()

    def _on_table_drop(self, **kwargs):
        table = kwargs['table']
        session = Session(bind=self.engine)
        session.query(SchemataTable).filter(
            SchemataTable.table_name == table.table_root(),
            SchemataTable.table_schema == table.schema()
        ).delete()
        session.commit()

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

    # 
    # context manager
    #
    def __enter__(self) -> DBManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def execute(self, query: str, args: Union[list, tuple] = []) -> pd.DataFrame:
        return None

    def current_date_expr(self):
        return "current_date"

    def create_schema(self, schema: str) -> Any:
        pass

    def create_table(self, table: TableHandle, columns: dict):
        pass

    def create_memory_table(self, table: str, df: pd.DataFrame):
        """ Create an in-memory table from the given dataframe. Used for processing lightweight results."""
        pass

    def rewrite_query(self, query: sqlglot.expressions.Expression) -> str:
        # Allows the db adapter to rewrite an incoming sql query
        pass

    def table_exists(self, table: TableHandle) -> bool:
        pass

    def write_dataframe_as_table(self, value: pd.DataFrame, table: TableHandle):
        pass

    def append_dataframe_to_table(self, value: pd.DataFrame, table: TableHandle):
        pass

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute_df("describe " + table)
        return rows["column_name"].values.tolist()

    def list_columns(self, table: TableHandle, match: str=None) -> pd.DataFrame:
        return self.execute_df(f"describe {table}")

    def delete_rows(self, table: TableHandle, filter_values: dict, where_clause: str=None):
        pass

    def get_table_root(self, table):
        if "." in table:
            return table.split(".")[-1]
        else:
            return table

    def get_short_date_cast(self, column):
        return f"strftime(CAST(\"{column}\" AS TIMESTAMP), '%m/%d/%y %H:%M')"

    def drop_schema(self, schema, cascade: bool=False):
        sql = f"DROP SCHEMA IF EXISTS {schema}"
        if cascade:
            sql += " cascade"
        res = self.execute(sql)
        self.signals["schema_drop"].emit(schema=schema)
        return res

    def drop_table(self, table: TableHandle):
        res = self.execute(f"drop table {table}")
        self.signals["table_drop"].emit(table=table)
        return res

    def dialect(self):
        return "postgres"

    def extract_missing_table(self, query, e):
        for table_ref in sqlglot.parse_one(query).find_all(sqlglot.exp.Table):
            if table_ref.name in query:
                return table_ref.sql('clickhouse')
        return '<>'

    def list_schemas(self):
        return self.execute(Queries.list_schemas)

    def list_tables(self, schema=None) -> pd.DataFrame:
        session = Session(bind=self.engine)
        records = [
            {
                'table_name': table.table_name,
                'table_schema': table.table_schema,
                'connection': table.connection,
                'description': table.description,
                'source': table.source,
                'created': table.created
            } for table in session.query(SchemataTable).filter(SchemataTable.table_schema == schema)
        ]
        return pd.DataFrame(records)

    @classmethod
    def get_sqlalchemy_table_args(cls, primary_key=None, schema=None):
        return {"schema": schema}

DATA_HOME = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_HOME, exist_ok=True)

class DuckDBWrapper(DBManager):
    DUCK_CONN: duckdb.DuckDBPyConnection = None
    REF_COUNTER = 0

    """ A cheap hack around DuckDB only usable in a single process. We just open/close
        each time to manage. Context manager for accessing the DuckDB database """

    def __init__(self):
        super().__init__()
        print(f"Connecting to local DuckDB database")

    def __enter__(self) -> DBManager:
        DuckDBWrapper.REF_COUNTER += 1
        if DuckDBWrapper.DUCK_CONN is None:
            db_path = os.path.join(DATA_HOME, "duckdata")
            use_same_connection = True

            if use_same_connection:
                self.engine = create_engine("duckdb:///" + db_path)
            else:
                self.engine = create_engine("duckdb:///" + "/tmp/duckmeta")
            self.engine.execute("CREATE SCHEMA IF NOT EXISTS " + UNIFY_META_SCHEMA)
            Base.metadata.create_all(self.engine)

            if use_same_connection:
                conn = self.engine.connect()
                DuckDBWrapper.DUCK_CONN = conn._dbapi_connection.dbapi_connection.c
            else:
                DuckDBWrapper.DUCK_CONN = duckdb.connect(db_path, read_only=False)

            DuckDBWrapper.DUCK_CONN.execute("PRAGMA log_query_path='/tmp/duckdb_log'")
            # create sqla model tables in the target schema
            DuckDBWrapper.DUCK_CONN.execute(f"create schema if not exists {UNIFY_META_SCHEMA}")
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

    def dialect(self):
        return "duckdb"

    def execute(self, query: str, args=[]):
        try:
            return DuckDBWrapper.DUCK_CONN.execute(query, args).df()
        except RuntimeError as e:
            if re.search(r"Table.+ does not exist", str(e)):
                raise TableMissingException(self.extract_missing_table(query, e))
            else:
                raise

    def execute_df(self, query: str, args=[]) -> pd.DataFrame:
        return self.execute(query, args)  # type: ignore

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute_df("describe " + table)
        return rows["column_name"].values.tolist()

    def delete_rows(self, table: TableHandle, filter_values: dict=None, where_clause: str=None):
        if filter_values:
            query = f"delete from {table} where " + " and ".join([f"{key} = ?" for key in filter_values.keys()])
            query = self._substitute_args(query, filter_values.values())
        else:
            query = f"delete from {table} where {where_clause}"
        self.execute(query)

    def create_schema(self, schema) -> duckdb.DuckDBPyConnection:
        res: duckdb.DuckDBPyConnection = self.execute(f"create schema if not exists {schema}")
        self.signals["schema_create"].emit(schema=schema)
        return res

    def create_table(self, table: TableHandle, columns: dict):
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
        self.signals["table_create"].emit(table=table)


    def create_memory_table(self, table_root: str, df: pd.DataFrame):
        """ Create an in-memory table from the given dataframe. Used for processing lightweight results."""
        if "." in table_root:
            raise RuntimeError("Memory tables cannot specify a schema")
        DuckDBWrapper.DUCK_CONN.register(table_root, df)
        return table_root

    def drop_memory_table(self, table_root: str):
        DuckDBWrapper.DUCK_CONN.unregister(table_root)

    def write_dataframe_as_table(self, value: pd.DataFrame, table: TableHandle):
        DuckDBWrapper.DUCK_CONN.register('df1', value)
        # create the table AND flush current row_buffer values to the db            
        DuckDBWrapper.DUCK_CONN.execute(f"create or replace table {table} as select * from df1")
        DuckDBWrapper.DUCK_CONN.unregister("df1")
        self.signals["table_create"].emit(table=table)

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

    def replace_table(self, source_table: TableHandle, dest_table: TableHandle):
        # Duck doesn't like the target name to be qualified
        self.execute(f"""
        BEGIN;
        DROP TABLE IF EXISTS {dest_table};
        ALTER TABLE {source_table} RENAME TO {dest_table.real_table_root()};
        COMMIT;
        """)

    def close(self):
        pass

    def is_closed(self) -> bool:
        return DuckDBWrapper.DUCK_CONN is None

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

PROCTECTED_SCHEMAS = ['information_schema','default']

class CHTableHandle(TableHandle):
    SCHEMA_SEP = '____'

    def __init__(self, table_handle: TableHandle, tenant_id, table_opts={}):
        super().__init__(table_handle.table_root(), table_handle.schema(), table_opts)
        self._tenant_schema = f"tenant_{tenant_id}"

    def real_table_root(self):        
        if super().schema() in PROCTECTED_SCHEMAS:
            return super().real_table_root()
        return self.schema() + CHTableHandle.SCHEMA_SEP + self.table_root()

    def real_schema(self):
        if super().schema() in PROCTECTED_SCHEMAS:
            return super().schema()
        return self._tenant_schema

    def __str__(self) -> str:
        return self.real_schema() + "." + self.real_table_root()

class CHTableMissing(TableMissingException):
    def __init__(self, msg):
        table = msg.split(".")[1].replace(CHTableHandle.SCHEMA_SEP, ".")
        super().__init__(table)

class ClickhouseWrapper(DBManager):
    # FIXME: Create a multi-tenant version of our DB manager
    SHARED_CLIENT = None
    SHARED_ENGINE = None
    SINGLE_TENANT_ID = None

    def __init__(self):
        super().__init__()
        self.client: Union[str,None] = None
        self.tenant_id: Union[str,None] = None
        self.tenant_db: Union[str,None] = None
        self.engine: Union[str,None] = None
        print(f"Connecting to clickhouse database at: {os.environ['DATABASE_HOST']}")

    def dialect(self):
        return "clickhouse"

    @staticmethod
    def _connect_to_db():
        if 'DATABASE_HOST' not in os.environ:
            raise RuntimeError("DATABASE_HOST not set")
        if 'DATABASE_USER' not in os.environ:
            raise RuntimeError("DATABASE_USER not set")
        if 'DATABASE_PASSWORD' not in os.environ:
            raise RuntimeError("DATABASE_PASSWORD not set")

        settings = {'allow_experimental_object_type': 1, 'allow_experimental_lightweight_delete': 1}
        ClickhouseWrapper.SHARED_CLIENT = Client(
            host=os.environ['DATABASE_HOST'], 
            user=os.environ['DATABASE_USER'],
            password=os.environ['DATABASE_PASSWORD'],
            settings=settings
        )
        ClickhouseWrapper.SINGLE_TENANT_ID = os.environ['DATABASE_USER']
         
        # Make sure the unify_schema database is created
        ClickhouseWrapper.SHARED_CLIENT.execute(f"CREATE DATABASE IF NOT EXISTS {UNIFY_META_SCHEMA}")
        # FIXME: Remove 'meta' schema and replace with use of unify_schema
        ClickhouseWrapper.SHARED_CLIENT.execute(f"CREATE DATABASE IF NOT EXISTS meta")

        # And make sure that the current tenant's dedicated schema exists
        tenant_db = f"tenant_{ClickhouseWrapper.SINGLE_TENANT_ID}"
        ClickhouseWrapper.SHARED_CLIENT.execute(f"CREATE DATABASE IF NOT EXISTS {tenant_db}")
        engine = get_sqla_engine()
        # Map of SQLA models to write to the tenant schema
        ClickhouseWrapper.SHARED_ENGINE = engine.execution_options(
    	    schema_translate_map={UNIFY_META_SCHEMA: tenant_db, None: tenant_db}
	    )
        # create sqla model tables in the target schema
        Base.metadata.create_all(ClickhouseWrapper.SHARED_ENGINE)


    def __enter__(self):
        if self.client is None:
            if ClickhouseWrapper.SHARED_CLIENT is None:
                ClickhouseWrapper._connect_to_db()
            self.client = ClickhouseWrapper.SHARED_CLIENT
            self.tenant_id = ClickhouseWrapper.SINGLE_TENANT_ID
            self.tenant_db = f"tenant_{self.tenant_id}"
            self.engine = ClickhouseWrapper.SHARED_ENGINE
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @classmethod
    def get_sqlalchemy_table_args(cls, primary_key=None, schema=None):
        return (
            clickhouse_engines.MergeTree(primary_key=primary_key),
            {"schema": schema}
        )

    def rewrite_query(self, query: typing.Union[sqlglot.expressions.Expression,str]=None) -> str:
        # Rewrite table references to use prefixes instead of schema names
        if isinstance(query, str):
            try:
                query = sqlglot.parse_one(query)
            except Exception as e:
                breakpoint()
                msg = f"sqlglot parsing failed: {e}"
                print(msg)
                return f"/* {msg} */ {query}"

        def transformer(node):
            if isinstance(node, sqlglot.exp.Table):
                parts = str(node).split(".")
                new_table = self.tenant_db + "." + parts[0] + CHTableHandle.SCHEMA_SEP + parts[1]
                return sqlglot.parse_one(new_table)
            return node

        newsql = query.transform(transformer).sql('clickhouse')
        print(f"REWROTE '{query}' as '{newsql}'")
        return newsql

    def current_date_expr(self):
        return "today()"

    def table_exists(self, table: TableHandle) -> bool:
        real_table = CHTableHandle(table, tenant_id=self.tenant_id)
        return self.client.execute(f"EXISTS {real_table}")[0] == 1

    def execute_raw(self, query: str, args=[]):
        return self.execute(query, args=args, native=True)

    def execute(self, query: str, args=[], native=False):
        if not native:
            query = self.rewrite_query(query)
        if query.strip().lower().startswith("insert"):
            return self._execute_insert(query, args)
        if args:
            query = self._substitute_args(query, args)

        logger.debug(query)
        try:
            return self.client.query_dataframe(query)
        except clickhouse_driver.errors.ServerException as e:
            m = re.search(r"Table (\S+) doesn't exist.", str(e))
            if m:
                # convert back to user land version
                raise CHTableMissing(m.group(1))
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

    def execute_df(self, query: str, native=False) -> pd.DataFrame:
        if not native:
            query = self.rewrite_query(query)
        try:
            return self.client.query_dataframe(query)
        except clickhouse_driver.errors.ServerException as e:
            if e.code == 60:
                m = re.search(r"Table (\S+) doesn't exist.", str(e))
                if m:
                    raise CHTableMissing(m.group(1))
            elif e.code == 62:
                m = re.search(r"Syntax error[^.]+.", e.message)
                if m:
                    raise QuerySyntaxException(m.group(0))
            elif "<Empty trace>" in e.message:
                e.message += " (while executing: " + query + ")"
                raise e
            m = re.search(r"(^.*)Stack trace:", e.message)
            if m:
                e.message = m.group(1)
            raise e

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute_df("describe " + str(CHTableHandle(table, tenant_id=self.tenant_id)), native=True)
        return rows["name"].values.tolist()

    def get_short_date_cast(self, column):
        return f"formatDateTime(CAST(\"{column}\" AS TIMESTAMP), '%m/%d/%y %H:%M')"

    def delete_rows(self, table: TableHandle, filter_values: dict=None, where_clause: str=None):
        table = CHTableHandle(table, tenant_id=self.tenant_id)
        if filter_values:
            query = f"alter table {table} delete where " + " and ".join([f"{key} = ?" for key in filter_values.keys()])
            query = self._substitute_args(query, filter_values.values())
        elif where_clause:
            query = f"alter table {table} delete where {where_clause}"
        res = self.execute(query, native=True)
        # Ugh. Seems deletes have some delay to show up...
        time.sleep(0.1)

    def create_schema(self, schema):
        # Clickhouse only supports one level of database.table nesting. To support multi-tenant therefore
        # we create a single database for the tenant and put all Unify "schemas" and tables in that
        # database. Table names are prefixed by the Unify schema name, and we will rewrite queries
        # to use the right table prefixes.

        # Thus our 'create schema' just registers the schema in the schemata, which in effect
        # "creates the schema" as far as the caller is concerned
        self.signals["schema_create"].emit(schema=schema)

    def list_schemas(self):
        session = Session(bind=self.engine)
        recs = [schema.name for schema in session.query(Schemata).filter(Schemata.type == 'schema')]
        recs += ['information_schema']
        return pd.DataFrame(recs, columns=["schema_name"])

    def list_tables(self, schema=None) -> pd.DataFrame:
        return super().list_tables(schema)

    def list_columns(self, table: TableHandle, match: str=None) -> pd.DataFrame:
        table = CHTableHandle(table, tenant_id=self.tenant_id)
        # FIXME: Handle 'match' - may need our tenant-specific information_schema table
        return self.execute_df(f"describe {table}", native=True)

    def create_table(self, table: TableHandle, columns: dict):
        real_table = CHTableHandle(table, tenant_id=self.tenant_id)

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
            f"create table if not exists {real_table} (" + ",".join(["{} {}".format(n, t) for n, t in new_cols.items()]) + ")" + \
                f" ENGINE = MergeTree() {primary_key} {ordering}"
        self.execute(table_ddl, native=True)
        self.signals["table_create"].emit(table=real_table)

    def replace_table(self, source_table: TableHandle, dest_table: TableHandle):
        real_src = CHTableHandle(source_table, tenant_id=self.tenant_id)
        real_dest = CHTableHandle(dest_table, tenant_id=self.tenant_id)

        self.execute_raw(f"EXCHANGE TABLES {real_src} AND {real_dest}")
        self.execute_raw(f"DROP TABLE {real_src}")

    def create_memory_table(self, table_root: str, df: pd.DataFrame):
        """ Create an in-memory table from the given dataframe. Used for processing lightweight results."""
        if "." in table_root:
            raise RuntimeError("Memory tables cannot specify a schema")
        table = TableHandle(table_root, "default")
        self.write_dataframe_as_table(df, table, table_engine="Memory")
        self.signals["table_create"].emit(table=table)
        return str(table)

    def drop_memory_table(self, table_root: str):
        self.execute(f"DROP TABLE IF EXISTS default.{table_root}")

    def drop_schema(self, schema, cascade: bool=False):        
        if cascade:
            # delete tables in the schema
            pass
        self.signals["schema_drop"].emit(schema=schema)

    def _on_schema_drop(self, schema):
        super()._on_schema_drop(schema)
        # FIXME: figure out the Clickhouse sync option
        time.sleep(0.1)
    

    def drop_table(self, table: TableHandle):
        chtable = CHTableHandle(table, tenant_id=self.tenant_id)
        self.execute(f"drop table {chtable}", native=True)
        self.signals["table_drop"].emit(table=table)

    def _on_table_drop(self, **kwargs):
        super()._on_table_drop(**kwargs)
        # FIXME: figure out the Clickhouse sync option
        time.sleep(0.1)

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

    def write_dataframe_as_table(self, value: pd.DataFrame, table: TableHandle, table_engine: str="MergeTree"):
        # FIXME: Use sqlglot to contruct this create statement
        # FIXME: replace spaces in column names
        table = CHTableHandle(table, tenant_id=self.tenant_id, table_opts = table.table_opts())
        col_specs, primary_key = self._infer_df_columns(value)
        if primary_key:
            primary_key = f"PRIMARY KEY \"{primary_key}\""
        if table_engine == "Memory":
            primary_key = ""

        self.client.execute(f"drop table if exists {table}")

        sql = f"create table {table} (" + \
            ", ".join([f"\"{col}\" {ctype}" for col, ctype in col_specs.items()]) + \
                f") Engine={table_engine}() {primary_key}"
        logger.debug(sql)
        self.client.execute(sql)
        self.signals["table_create"].emit(table=table)

        logger.debug("Writing dataframe to table")
        if value.shape[0] > 0:
            self.client.insert_dataframe(
                f"INSERT INTO {table} VALUES", 
                value, 
                settings={'use_numpy': True}
            )

    def append_dataframe_to_table(self, value: pd.DataFrame, table: TableHandle):
        # There is a problem where a REST API returns a boolean column, but the first page 
        # of results is all nulls. In that case the type inference will have failed and we
        # will have defaulted to type the column as a string. We need to detect this case
        # and either coerce the bool column or fix the column type. For now we are doing
        # the former.

        # Use pyarrow for convenience, but type info probably already exists on the dataframe
        real_table = CHTableHandle(table, tenant_id=self.tenant_id)

        # FIXME: When are append a DF to an existing table it is possible that the types won't
        # match. So we need to query the types from the table and make sure to coerce the DF
        # to those types. For now we have only handled a special case for booleans.
        df_schema = pa.Schema.from_pandas(value)
        for col in df_schema.names:
            f = df_schema.field(col)
            if pa.types.is_boolean(f.type):
                # See if the table column is a string
                db_type = self.execute(
                    Queries.list_columns.format(real_table.real_schema(), real_table.real_table_root(), col),
                    native=True
                )['data_type'].iloc[0]
                if db_type.lower() == "string" or db_type.lower().startswith("varchar"):
                    # Corece bool values to string
                    value[col] = value[col].astype(str)
                    logger.critical("Coercing bool column {} to string".format(col))

        self.client.insert_dataframe(
            f"INSERT INTO {real_table} VALUES", 
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
        self.duck : DBManager = duck
        self.duck.create_schema("meta")

    def ensure_col_table(self, duck, name) -> str:
        table: str = self.adapter_schema + "_" + name
        duck.create_table(TableHandle(table, "meta"), self.TABLE_SCHEMA)
        return table

    def ensure_log_table(self, duck, name) -> dict:
        table = self.adapter_schema + "_" + name
        duck.create_table(TableHandle("meta." + table), self.LOG_TABLE_SCHEMA)
        return table

    def create_var_storage_table(self, duck):
        table = "system__vars"
        duck.create_table(TableHandle("meta." + table), {'*name': 'VARCHAR', 'value': 'BLOB'})
        return table

    def put_var(self, name, value):
        table = self.create_var_storage_table(self.duck)
        self.duck.delete_rows(TableHandle("meta." + table), {"name": name})
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
        self.duck.delete_rows(TableHandle("meta." + table), {"id": id})
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
        )
        return [json.loads(row) for row in rows['blob']]

    @lru_cache(maxsize=500)
    def ensure_column_intel_table(self):
        table = TableHandle("meta.column_intels")
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
        r = self.duck.delete_rows(TableHandle(table), {'id': id})
        print("Delete resulted in: ", r)

    def delete_log_objects(self, collection: str, id: str):
        self.delete_object(collection, id)

    def list_objects(self, collection: str) -> list[tuple]:
        table = self.ensure_col_table(self.duck, collection)
        return [
            (row['id'], json.loads(row['blob'])) for row in \
                self.duck.execute_df(f"select id, blob from meta.{table}").to_records()
        ]


## ORM classes for unify_schema 

Base = declarative_base()

def uniq_id():
    return str(uuid.uuid4())

DBMGR_CLASS: DBManager = ClickhouseWrapper if os.environ['DATABASE_BACKEND'] == 'clickhouse' else DuckDBWrapper

class Schemata(Base):  # type: ignore
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "schemata"

    id = Column(String, default=uniq_id, primary_key=True)
    name = Column(String)
    type = Column(String, default="schema")
    type_or_spec = Column(String)
    created = Column(DateTime, default=datetime.utcnow())
    description = Column(String)
    
    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='id', schema=UNIFY_META_SCHEMA)

class SchemataTable(Base): #type ignore
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "tables"

    id = Column(String, default=uniq_id, primary_key=True)
    table_name = Column(String)
    table_schema = Column(String)
    connection = Column(String)
    refresh_schedule = Column(String)
    description = Column(String)
    source = Column(String)
    created = Column(DateTime, default=datetime.utcnow())
    
    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='id', schema=UNIFY_META_SCHEMA)
    def __repr__(self) -> str:
        return f"TableSchemata({self.table_schema}.{self.table_name})"

class ConnectionScan(Base):
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "connectionscans"

    id = Column(String, default=uniq_id, primary_key=True)
    created = Column(DateTime, default=datetime.utcnow())
    table_name = Column(String)
    table_schema = Column(String)
    connection = Column(String)
    values = Column(String)

    def set_values(self, vals: dict):
        self.values = json.dumps(vals)

    def get_values(self):
        if self.values is not None:
            return json.loads(self.values)
        else:
            return {}

    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='id', schema=UNIFY_META_SCHEMA)
    def __repr__(self) -> str:
        return f"ConnectionScan({self.table_schema}.{self.table_name})"

def get_sqla_engine():
    uri = 'clickhouse://' + \
        os.environ['DATABASE_USER'] + ':' +\
        os.environ['DATABASE_PASSWORD'] + '@' + \
        os.environ['DATABASE_HOST'] + '/default'
    return create_engine(uri)

def create_orm_tables(engine):
    engine = get_sqla_engine()
    Base.metadata.create_all(engine)
    return engine
