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

from __future__ import annotations
import base64
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
import threading
from typing import Union, Any
import uuid
from venv import create

import pandas as pd
import pyarrow as pa
import duckdb
from clickhouse_driver import Client
import clickhouse_driver
import sqlglot
import sqlglot.expressions
from sqlalchemy import Enum
from sqlalchemy import Column, String, DateTime, PickleType, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import Session
from sqlalchemy.types import TypeDecorator, VARCHAR
from clickhouse_sqlalchemy import engines as clickhouse_engines
from signaling import Signal

from .schemata import Queries
from .storage_manager import StorageManager

logger: logging.Logger = logging.getLogger('unify')

UNIFY_META_SCHEMA = 'unify_schema'

# Patch SQLA Cursor class which doesn't work will with SQL Inserts on Clickhouse
from .sqla_patch import patch_sqlalchemy_cursor
patch_sqlalchemy_cursor()

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

    def user_name(self) -> str:
        return self.schema() + "." + self.table_root()

    def __str__(self) -> str:
        return self.schema() + "." + self.table_root()

    def __repr__(self) -> str:
        return "TableHandle(" + str(self) + ")"

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def set_opt(self, key: str, value: str):
        self._table_opts[key] = value

class DBSignals:
    SCHEMA_CREATE = "schema_create"
    SCHEMA_DROP = "schema_drop"
    TABLE_CREATE = "table_create"
    TABLE_DROP = "table_drop"
    TABLE_LOADED = "table_loaded"
    TABLE_RENAME = "table_rename"

class DBManager(contextlib.AbstractContextManager):
    SIG_DICT = {}

    # 
    # Schemata signals
    #
    def __init__(self) -> None:
        super().__init__()
        self.signals : typing.Dict[str, Signal] = {}
        self.DATA_HOME = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.DATA_HOME, exist_ok=True)

    def _setup_signals(self):
        self.signals = DBManager.SIG_DICT
        if len(self.signals) == 0:
            self.signals.update({
                DBSignals.SCHEMA_CREATE: Signal(args=['dbmgr', 'schema']),
                DBSignals.SCHEMA_DROP: Signal(args=['dbmgr', 'schema']),
                DBSignals.TABLE_CREATE: Signal(args=['dbmgr', 'table']),
                DBSignals.TABLE_DROP: Signal(args=['dbmgr', 'table']),
                DBSignals.TABLE_RENAME: Signal(args=['dbmgr', 'old_table', 'new_table']),
                DBSignals.TABLE_LOADED: Signal(args=['dbmgr', 'table']), # emitted after a table is loaded or updated
            })
        self.signals[DBSignals.SCHEMA_CREATE].connect(self._on_schema_create)
        self.signals[DBSignals.SCHEMA_DROP].connect(self._on_schema_drop)
        self.signals[DBSignals.TABLE_CREATE].connect(self._on_table_create)
        self.signals[DBSignals.TABLE_DROP].connect(self._on_table_drop)
        self.signals[DBSignals.TABLE_RENAME].connect(self._on_table_rename)
        # Keep track of table references from the most recent query
        self.last_seen_tables = []

    def _remove_signals(self):
        self.signals[DBSignals.SCHEMA_CREATE].disconnect(self._on_schema_create)
        self.signals[DBSignals.SCHEMA_DROP].disconnect(self._on_schema_drop)
        self.signals[DBSignals.TABLE_CREATE].disconnect(self._on_table_create)
        self.signals[DBSignals.TABLE_DROP].disconnect(self._on_table_drop)
        self.signals[DBSignals.TABLE_RENAME].disconnect(self._on_table_rename)

    def register_for_signal(self, signal, callback):
        if signal not in self.signals:
            raise RuntimeError(f"Unknown signal '{signal}'")
        self.signals[signal].connect(callback)

    def send_signal(self, signal=None, **kwargs):
        if signal not in self.signals:
            raise RuntimeError(f"Unknown signal '{signal}'")
        kwargs['dbmgr'] = self       
        self.signals[signal].emit(**kwargs)

    def _send_signal(self, signal, **kwargs):
        kwargs['dbmgr'] = self
        self.signals[signal].emit(**kwargs)

    def _on_schema_create(self, dbmgr, schema):
        session = Session(bind=self.engine)
        session.query(Schemata).filter(Schemata.name == schema).delete()
        session.add(Schemata(name=schema, type="schema"))
        session.commit()

    def _on_schema_drop(self, dbmgr, schema):
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
        session.expire_all()

    def _on_table_rename(self, **kwargs):
        old_table = kwargs['old_table']
        new_table = kwargs['new_table']
        session = Session(bind=self.engine)
        session.query(SchemataTable).filter(
            SchemataTable.table_name == old_table.table_root(),
            SchemataTable.table_schema == old_table.schema()
        ).update({
            SchemataTable.table_name: new_table.table_root(),
            SchemataTable.table_schema: new_table.schema()
        })
        session.commit()
        
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
    # query parsing
    #

    def parse_query_intel(self, query: str, dialect: str, capture_dml: list=[]) -> dict:
        tables: list[TableHandle] = []

        def transformer(node):
            if isinstance(node, sqlglot.exp.Table):
                tables.append(TableHandle(str(node)))
            return node

        try:
            parsed_query = sqlglot.parse_one(query, read=dialect)
            # Look for creates or drops executed as raw sql
            capture_dml.extend(self._find_table_op(parsed_query, sqlglot.exp.Create, DBSignals.TABLE_CREATE))
            capture_dml.extend(self._find_table_op(parsed_query, sqlglot.exp.Drop, DBSignals.TABLE_DROP))

            parsed_query.transform(transformer).sql()
            return {"tables": tables}
        except Exception as e:
            return {}

    def _find_table_op(self, query: sqlglot.Expression, operation: sqlglot.expressions._Expression, signal: str):
        for table_op in query.find_all(operation):
            kind = table_op.args['kind'] # view or table
            for table_ref in table_op.find_all(sqlglot.exp.Table):
                return [{'signal': signal, 'table': table_ref.sql(), 'kind': kind}]
        return []

    # 
    # context manager
    #
    def __enter__(self) -> DBManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def execute(self, query: str, args: Union[list, tuple] = [], context=None) -> pd.DataFrame:
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

    def rename_table(self, table: TableHandle, new_name: str):
        pass

    def table_exists(self, table: TableHandle) -> bool:
        pass

    def write_dataframe_as_table(self, value: pd.DataFrame, table: TableHandle):
        pass

    def append_dataframe_to_table(self, value: pd.DataFrame, table: TableHandle):
        pass

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute("describe " + table)
        return rows["column_name"].values.tolist()

    def list_columns(self, table: TableHandle, match: str=None) -> pd.DataFrame:
        """ Returns a dataframe describing each column in a table. """
        return self.execute(f"describe {table}")

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
        self._send_signal(signal=DBSignals.SCHEMA_DROP, schema=schema)
        return res

    def drop_table(self, table: TableHandle):
        res = self.execute(f"drop table if exists {table}")
        self._send_signal(signal=DBSignals.TABLE_DROP, table=table)
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
        """ Returns a dataframe describing the tables inside the given schema.
            Columns will include: table_name, table_schema, connection, decription, source, created
        """
        session = Session(bind=self.engine)
        query = session.query(SchemataTable)
        if schema:
            query = query.filter(SchemataTable.table_schema == schema)
        records = [
            {
                'table_name': table.table_name,
                'table_schema': table.table_schema,
                'connection': table.connection,
                'description': table.description,
                'source': table.source,
                'created': table.created
            } for table in query.order_by('table_schema', 'table_name')
        ]
        df = pd.DataFrame(records)
        if df.shape[0] > 0:
            df = df[['table_schema', 'table_name', 'connection', 'created']]
        return df

    @classmethod
    def get_sqlalchemy_table_args(cls, primary_key=None, schema=None):
        return {"schema": schema}

class DuckDBWrapper(DBManager):
    DUCK_CONN: duckdb.DuckDBPyConnection = None
    DUCK_ENGINE = None
    REF_COUNTER = 0
    ALERTED = False

    """ A cheap hack around DuckDB only usable in a single process. We just open/close
        each time to manage. Context manager for accessing the DuckDB database """

    def __init__(self):
        super().__init__()
        if not DuckDBWrapper.ALERTED:
            logger.debug(f"Connecting to local DuckDB database")
            DuckDBWrapper.ALERTED = True

    def __enter__(self) -> DBManager:
        DuckDBWrapper.REF_COUNTER += 1
        if DuckDBWrapper.DUCK_CONN is None:
            db_path = os.path.join(self.DATA_HOME, "duckdata")

            DuckDBWrapper.DUCK_ENGINE = create_engine("duckdb:///" + db_path)
            DuckDBWrapper.DUCK_ENGINE.execute("CREATE SCHEMA IF NOT EXISTS " + UNIFY_META_SCHEMA)
            Base.metadata.create_all(DuckDBWrapper.DUCK_ENGINE)

            conn = DuckDBWrapper.DUCK_ENGINE.connect()
            DuckDBWrapper.DUCK_CONN = conn._dbapi_connection.dbapi_connection.c

            DuckDBWrapper.DUCK_CONN.execute("PRAGMA log_query_path='/tmp/duckdb_log'")
            # create sqla model tables in the target schema
            DuckDBWrapper.DUCK_CONN.execute(f"create schema if not exists {UNIFY_META_SCHEMA}")

        self.engine = DuckDBWrapper.DUCK_ENGINE
        self._setup_signals()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        DuckDBWrapper.REF_COUNTER -= 1
        self._remove_signals()
        if DuckDBWrapper.REF_COUNTER == 0 and self.__class__.DUCK_CONN is not None:
            try:
                DuckDBWrapper.DUCK_CONN.execute("COMMIT")
            except:
                pass
            # We are sharing the conn with SQLA, so let the Engine close it
            DuckDBWrapper.DUCK_ENGINE.dispose()
            DuckDBWrapper.DUCK_CONN = None

    def dialect(self):
        return "duckdb"

    def execute(self, query: str, args=[], context=None):
        capture_dmls: list = []
        intel = self.parse_query_intel(query, dialect='duckdb', capture_dml=capture_dmls)
        if 'tables' in intel and context:
            context.set_recent_tables(intel['tables'])

        try:
            df = DuckDBWrapper.DUCK_CONN.execute(query, args).df()

            for dml in capture_dmls:
                self._send_signal(dml['signal'], table=TableHandle(dml['table']))

            return df
        except RuntimeError as e:
            if re.search(r"Table.+ does not exist", str(e)):
                raise TableMissingException(self.extract_missing_table(query, e))
            else:
                raise

    def get_table_columns(self, table: TableHandle):
        # Returns the column names for the table in their insert order
        rows = self.execute("describe " + str(table))
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
        self._send_signal(signal=DBSignals.SCHEMA_CREATE, schema=schema)
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

        sql = f"create table if not exists {table} (" + ",".join(["{} {}".format(n, t) for n, t in new_cols.items()]) + ")"
        self.execute(sql)
        # This is not needed, as 'execute' will parse the create and send the signal for us
        #self._send_signal(signal=DBSignals.TABLE_CREATE, table=table)


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
        self._send_signal(signal=DBSignals.TABLE_CREATE, table=table)

    def append_dataframe_to_table(self, value: pd.DataFrame, table: TableHandle):
        # FIXME: we should probably use a context manager at the caller to ensure
        # we set and unset the search_path properly
        DuckDBWrapper.DUCK_CONN.execute(f"set search_path='{table.schema()}'")
        DuckDBWrapper.DUCK_CONN.append(table.table_root(), value)

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

def patched_clickhouse_write_items(self, items, buf):
    items = [x if x else '' for x in items]
    buf.write_strings(items or '', encoding=self.encoding)

def monkeypatch_clickhouse_driver():
    NumpyDateTimeColumnBase.apply_timezones_before_write = patched_apply_timezones_before_write
    clickhouse_driver.columns.stringcolumn.String.write_items = patched_clickhouse_write_items

monkeypatch_clickhouse_driver()

# patch LegacyResultCursor with this logic:
# if hasattr(result, 'inserted_primary_key_rows'):
#                         primary_key = result.inserted_primary_key_rows[0]
#                     else:
#                         primary_key = result.inserted_primary_key

PROCTECTED_SCHEMAS = ['information_schema']

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
        self.lock = threading.RLock()

    def dialect(self):
        return "clickhouse"

    @staticmethod
    def get_sqla_engine():
        uri = 'clickhouse+native://' + \
            os.environ['DATABASE_USER'] + ':' +\
            os.environ['DATABASE_PASSWORD'] + '@' + \
            os.environ['DATABASE_HOST'] + '/default'
        return create_engine(uri)

    @staticmethod
    def _connect_to_db():
        logger.debug(f"Connecting to clickhouse database at: {os.environ['DATABASE_HOST']}")
        if 'DATABASE_HOST' not in os.environ:
            raise RuntimeError("DATABASE_HOST not set")
        if 'DATABASE_USER' not in os.environ:
            raise RuntimeError("DATABASE_USER not set")
        if 'DATABASE_PASSWORD' not in os.environ:
            raise RuntimeError("DATABASE_PASSWORD not set")

        ClickhouseWrapper.SINGLE_TENANT_ID = os.environ['DATABASE_USER']
        tenant_db = f"tenant_{ClickhouseWrapper.SINGLE_TENANT_ID}"
        settings = {'allow_experimental_object_type': 1, 'allow_experimental_lightweight_delete': 1}
        # FIXME: Pass the settings in when creating the SQLAlchemy engine

        ClickhouseWrapper.SHARED_ENGINE = ClickhouseWrapper.get_sqla_engine().execution_options(
            # Map of SQLA model schemas to point to the tenant schema
    	    schema_translate_map={UNIFY_META_SCHEMA: tenant_db, None: tenant_db}
	    )
        conn = ClickhouseWrapper.SHARED_ENGINE.connect()
        ClickhouseWrapper.SHARED_CLIENT = conn._dbapi_connection.dbapi_connection.transport

        # And make sure that the current tenant's dedicated schema exists
        ClickhouseWrapper.SHARED_CLIENT.execute(f"CREATE DATABASE IF NOT EXISTS {tenant_db}")

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
        self._setup_signals()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_signals()
        pass

    @classmethod
    def get_sqlalchemy_table_args(cls, primary_key=None, schema=None):
        return (
            clickhouse_engines.MergeTree(primary_key=primary_key),
            {"schema": schema}
        )

    def rewrite_query(self, query: typing.Union[sqlglot.expressions.Expression,str]=None, capture_dml: list=[]) -> str:
        # Rewrite table references to use prefixes instead of schema names
        # FIXME: fix "show create table ..."
        if isinstance(query, str):
            try:
                query = sqlglot.parse_one(query, read='clickhouse')            
            except Exception as e:
                msg = f"sqlglot parsing failed: {e}"
                return f"/* {msg} */ {query}"

        # Look for creates or drops executed as raw sql
        capture_dml.extend(self._find_table_op(query, sqlglot.exp.Create, DBSignals.TABLE_CREATE))
        capture_dml.extend(self._find_table_op(query, sqlglot.exp.Drop, DBSignals.TABLE_DROP))

        cte_tables = [str(alias) for alias in query.find_all(sqlglot.exp.TableAlias)]

        def transformer(node):
            if isinstance(node, (sqlglot.exp.Table, sqlglot.exp.Column)):
                if re.search(r"\(.*\)", str(node)):
                    # ignore function calls
                    return node
                if isinstance(node, sqlglot.exp.Table):
                    if "." not in str(node):
                        # Coerce unqualified table references into 'default' schema
                        node = sqlglot.parse_one(f"default.{str(node)}")
                    self.last_seen_tables.append(TableHandle(str(node)))
                parts = str(node).split(".")
                if len(parts) == 2:
                    if parts[0] not in cte_tables:
                        new_table = self.tenant_db + "." + parts[0] + CHTableHandle.SCHEMA_SEP + parts[1]
                        return sqlglot.parse_one(new_table)
            return node

        newsql = query.transform(transformer).sql('clickhouse')
        if re.match(r"create\s+table", newsql, re.IGNORECASE) and not re.match("engine", newsql, re.IGNORECASE):
            newsql += " ENGINE = MergeTree()"
        return newsql

    def rename_table(self, table: TableHandle, new_name: str):
        if "." in new_name:
            raise QuerySyntaxException("Cannot specify schema when renaming table")
        old_table = CHTableHandle(table, self.tenant_id)
        new_table = CHTableHandle(TableHandle(new_name, table.schema()), self.tenant_id)
        df = self.execute(f"RENAME TABLE {old_table} TO {new_table}", native=True)
        self.send_signal(DBSignals.TABLE_RENAME, old_table=old_table, new_table=new_table)


    def current_date_expr(self):
        return "today()"

    def table_exists(self, table: TableHandle) -> bool:
        real_table = CHTableHandle(table, tenant_id=self.tenant_id)
        return self.client.execute(f"EXISTS {real_table}")[0] == 1

    def execute_raw(self, query: str, args=[]):
        return self.execute(query, args=args, native=True)

    def execute(self, query: str, args=[], native=False, context=None) -> pd.DataFrame:
        self.last_seen_tables = []
        capture_dmls = []
        if not native:
            query = self.rewrite_query(query, capture_dml=capture_dmls)
            if context:
                context.set_recent_tables(self.last_seen_tables)
        if query.strip().lower().startswith("insert"):
            return self._execute_insert(query, args)
        if args:
            query = self._substitute_args(query, args)

        try:
            df = self.client.query_dataframe(query)
            # The Clickhouse client doesn't seem to return actually date-typed columns for date functions, even
            # through the query returns date-typed values. So let's try to coerce the column type.
            if df.shape[0] > 0:
                for col in df.columns:
                    if df[col].iloc[0].__class__.__name__.startswith('date'):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='ignore')
                        except:
                            pass

            for dml in capture_dmls:
                self._send_signal(dml['signal'], table=TableHandle(dml['table']))
            return df
        except clickhouse_driver.errors.ServerException as e:
            if e.code == 60:
                m = re.search(r"Table (\S+) doesn't exist.", str(e))
                if m:
                    raise CHTableMissing(m.group(1))
            elif e.code == 62:
                m = re.search(r"Syntax error[^.]+.", e.message)
                if m:
                    raise QuerySyntaxException(m.group(0))
            elif e.code == 47:
                m = re.search(r"DB::Exception.*(Stack)?", e.message)
                if m:
                    raise QuerySyntaxException(m.group(0))
            elif "<Empty trace>" in e.message:
                e.message += " (while executing: " + query + ")"
                raise e
            m = re.search(r"(^.*)Stack trace:", e.message)
            if m:
                e.message = m.group(1)
            raise e


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

    def get_table_columns(self, table):
        # Returns the column names for the table in their insert order
        rows = self.execute("describe " + str(CHTableHandle(table, tenant_id=self.tenant_id)), native=True)
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
        self._send_signal(signal=DBSignals.SCHEMA_CREATE, schema=schema)


    def list_schemas(self):
        session = Session(bind=self.engine)
        recs = [schema.name for schema in session.query(Schemata).filter(Schemata.type == 'schema')]
        recs += ['information_schema']
        return pd.DataFrame(recs, columns=["schema_name"])

    def list_tables(self, schema=None) -> pd.DataFrame:
        if schema == 'information_schema':
            # Special case because we don't reflect our custom IS tables ourselves, so fall back
            # to the system.
            tprefix = "information_schema" + CHTableHandle.SCHEMA_SEP
            q = f"select * from information_schema.tables where table_schema = '{self.tenant_db}' and " + \
                f"table_name like '{tprefix}%'"
            df = self.execute(q, native=True)
            df['table_schema'] = 'information_schema'
            df['table_name'] = df['table_name'].apply(lambda x: x[len(tprefix):])
            return df[['table_schema', 'table_name']]
        else:
            return super().list_tables(schema)

    def list_columns(self, table: TableHandle, match: str=None) -> pd.DataFrame:
        table = CHTableHandle(table, tenant_id=self.tenant_id)
        # FIXME: Handle 'match' - may need our tenant-specific information_schema table
        df = self.execute(f"describe {table}", native=True)
        df.rename(columns={'name': 'column_name', 'type': 'column_type'}, inplace=True)
        if match is not None:
            match = match.replace('*', '.*')
            match = match.replace('%', '.*')
            match = '^' + match + '$'
            return df.loc[df.column_name.str.contains(match, case=False)].sort_values('column_name')
        else:
            return df.sort_values('column_name')

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
        self._send_signal(signal=DBSignals.TABLE_CREATE, table=real_table)

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
        self._send_signal(signal=DBSignals.TABLE_CREATE, table=table)
        return str(table)

    def drop_memory_table(self, table_root: str):
        self.execute(f"DROP TABLE IF EXISTS default.{table_root}")

    def drop_schema(self, schema, cascade: bool=False):        
        if cascade:
            try:
                for table in super().list_tables(schema)['table_name']:
                    self.drop_table(TableHandle(table, schema))
            except KeyError:
                pass
        self._send_signal(signal=DBSignals.SCHEMA_DROP, schema=schema)

    def _on_schema_drop(self, dbmgr, schema):
        super()._on_schema_drop(dbmgr, schema)
        # FIXME: figure out the Clickhouse sync option
        time.sleep(0.1)
    
    def drop_table(self, table: TableHandle):
        chtable = CHTableHandle(table, tenant_id=self.tenant_id)
        self.execute(f"drop table if exists {chtable}", native=True)
        self._send_signal(signal=DBSignals.TABLE_DROP, table=table)

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
        self._send_signal(signal=DBSignals.TABLE_CREATE, table=table)

        logger.debug("Writing dataframe to table")
        if value.shape[0] > 0:
            self.client.insert_dataframe(
                f"INSERT INTO {table} VALUES", 
                value, 
                settings={'use_numpy': True}
            )

    def append_dataframe_to_table(self, value: pd.DataFrame, table: TableHandle):
        with self.lock:
            self._append_dataframe_to_table(value, table)

    def _append_dataframe_to_table(self, value: pd.DataFrame, table: TableHandle):
        # There is a problem where a REST API returns a boolean column, but the first page 
        # of results is all nulls. In that case the type inference will have failed and we
        # will have defaulted to type the column as a string. We need to detect this case
        # and either coerce the bool column or fix the column type. For now we are doing
        # the former.
        self.lock.acquire()

        # Use pyarrow for convenience, but type info probably already exists on the dataframe
        real_table = CHTableHandle(table, tenant_id=self.tenant_id)

        # FIXME: When are append a DF to an existing table it is possible that the types won't
        # match. So we need to query the types from the table and make sure to coerce the DF
        # to those types. For now we have only handled a special case for booleans.
        df_schema = pa.Schema.from_pandas(value)
        for col in df_schema.names:
            f = df_schema.field(col)
            if pa.types.is_boolean(f.type) or pa.types.is_timestamp(f.type):
                # See if the table column is a string
                db_type = self.execute(
                    Queries.list_columns.format(real_table.real_schema(), real_table.real_table_root(), col),
                    native=True
                )['data_type'].iloc[0]
                if db_type.lower() == "string" or db_type.lower().startswith("varchar"):
                    # Corece bool values to string
                    value[col] = value[col].astype(str)
                    logger.debug("Coercing bool/date column {} to string".format(col))

        self.client.insert_dataframe(
            f"INSERT INTO {real_table} VALUES", 
            value, 
            settings={'use_numpy': True}
        )

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
    description = Column(String, nullable=True)
    
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

# from here: https://docs.sqlalchemy.org/en/14/core/custom_types.html#marshal-json-strings
class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string.

    Usage::

        JSONEncodedDict(255)

    """
    impl = VARCHAR
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value

class ConnectionScan(Base):
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "connectionscans"

    id = Column(String, default=uniq_id, primary_key=True)
    created = Column(DateTime, default=datetime.utcnow())
    table_name = Column(String)
    table_schema = Column(String)
    connection = Column(String)
    values = Column(JSONEncodedDict)

    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='id', schema=UNIFY_META_SCHEMA)
    def __repr__(self) -> str:
        return f"ConnectionScan({self.table_schema}.{self.table_name})"

class ColumnInfo(Base):
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "columns"
    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='id', schema=UNIFY_META_SCHEMA)

    id = Column(String, default=uniq_id, primary_key=True)
    created = Column(DateTime, default=datetime.utcnow())
    table_name = Column(String)
    table_schema = Column(String)
    name = Column(String)
    attrs: dict = Column(JSONEncodedDict)

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

class RunSchedule(Base):
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "run_schedules"
    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='id', schema=UNIFY_META_SCHEMA)

    id = Column(String, primary_key=True)
    notebook_path = Column(String)
    run_at = Column(DateTime)
    repeater = Column(String)
    contents = Column(String)

class Base64Encoded(TypeDecorator):
    impl = VARCHAR
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = base64.b64encode(pickle.dumps(value)).decode("ascii")

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = pickle.loads(base64.b64decode(value.encode("ascii")))
        return value

class SavedVar(Base):
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "variables"
    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='name', schema=UNIFY_META_SCHEMA)

    name = Column(String, primary_key=True)
    value = Column(Base64Encoded)

class AdapterMetadata(Base):
    __tablename__ = "information_schema" + CHTableHandle.SCHEMA_SEP + "adapter_metadata"
    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key=['id','collection'], schema=UNIFY_META_SCHEMA)

    id = Column(String, primary_key=True)
    collection = Column(String, primary_key=True)
    values = Column(JSONEncodedDict)

# 
# ACTIVE DB CONNECTION
#
# Verify DB settings
if 'DATABASE_BACKEND' not in os.environ or os.environ['DATABASE_BACKEND'] not in ['duckdb','clickhouse']:
    raise RuntimeError("Must set DATABASE_BACKEND to 'duckdb' or 'clickhouse'")

dbmgr: DBManager = ClickhouseWrapper if os.environ['DATABASE_BACKEND'] == 'clickhouse' else DuckDBWrapper

