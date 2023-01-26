import logging
import os
import typing
import re
import shutil
import subprocess
from datetime import datetime

import pandas as pd
import psycopg

from .connectors import (
    Connector, 
    ConnectorQueryResult, 
    OutputLogger, 
    ReloadStrategy, 
    StorageManager, 
    TableDef, 
    TableUpdater
)
from .db_wrapper import DBManager, TableHandle, CHTableHandle, DBSignals
 
#######
# A Replicating Postgres connector which uses COPY to quickly replicate Postgres tables.
# On extract, use "COPY" to dump the Postgres table to a local CSV file.
# 
# To load the file we:
#   - create the table with the right schema in Clickhouse, mapping PG types to CH types
#   - determine the OrderBy column for the Clickhouse MergeTree
#   - use "INSERT INTO table SELECT FROM <file>" quickly load the CSV data
#
# Note that this works weel (and quickly) as long as the Clickhouse server is local.
# To support remote servers we have to stream the file over the Clickhouse client.
#####

PostgresConnector = typing.NewType("PostgresConnector", None)

class PostgresTableSpec(TableDef):
    # Represents a Postgres table on the remote server

    def __init__(self, table: str, connector):
        super().__init__(table)
        self.connector: PostgresConnector = connector
    
    def get_table_updater(self, updates_since: datetime) -> TableUpdater:
        # Our broad table update strategy looks like this:
        #
        # 1. For small tables (< 100,000 rows), we'll just do a full reload and swap the new
        # table into place.
        # 2. For larger tables we will attempt to do an incremental update by querying for new or
        # modified records.
        # 3. If we can't find an "update" column then we'll fall back to the full reload.

        return ReloadStrategy(self)

        # See if we can find a column to use for incremental updates
        db: DBManager = self.connector.db
        table = TableHandle(self.schema_name, self.connector.name)
        cols = db.list_columns(table)

        # We have a few strategies:
        # 1. Look for a timestamp column with "update" in the name, and assume that represents the update
        # time of the row.
        # 2. Look for a timestamp column with "create" in the name, and assume that represents the creation
        # time.
        # 3. Look for an int column whose range is close to the row count of the table. This implies an 
        # incrementing column that we can use.
        for index, col in cols.iterrows():
            col_name = col['column_name']
            if col['column_type'].lower().startswith("int"):
                vals = db.execute(f"select min({col_name}), max({col_name}), count(*) as count from {table}").to_records(index=False)
                print(vals)

        # Once we have the update column, then we will go query the updated records from the remote server
        # mapping, then yield those in batches to the normal table loader.

    def query_resource(self, tableLoader, logger: logging.Logger):
        # We will implement the replication entirely within the database, rather than the
        # usual pattern of returning chunks of rows to insert.
        for count in self.connector.load_table(self.name):
            yield ConnectorQueryResult(json=pd.DataFrame(), size_return=[], rows_written=count)

class PostgresConnector(Connector):
    def __init__(self, spec, storage: StorageManager, schema_name: str):
        super().__init__(spec['name'], storage)
        self.auth = spec.get('auth', {}).get('params').copy()

        # connection params will live in self.auth
        self.logger: OutputLogger = None
        self.tables = None
        self.schema_name = schema_name
        if storage:
            self.db: DBManager = storage.get_local_db()
            self.tenant_db = self.db.tenant_db

    def get_config_parameters(self):
        return {
            "db_host": "Database host name",
            "db_user": "Database user",
            "db_password": "User password",
            "db_database": "Database name"
        }

    def validate(self) -> bool:
        self.conn = psycopg.connect(
            f"postgresql://{self.auth['db_user']}:{self.auth['db_password']}@{self.auth['db_host']}/{self.auth['db_database']}"
        )
        return True

    def list_tables(self):
        # select tables names through the remote PG connection and return TableDefs for each one
        with self.conn.cursor() as cursor:
            cursor.execute("select table_name, table_schema from information_schema.tables where table_schema='public'")
            return [PostgresTableSpec(t[0], self) for t in cursor.fetchall()]

    def _get_remote_table_columns(self, table_root):
        # Find the table columns and types so we can copy the schema
        # Borrowed from: https://stackoverflow.com/questions/20194806/how-to-get-a-list-column-names-and-datatypes-of-a-table-in-postgresql
        schema = 'public'
        sql = f"""
            SELECT
                    a.attname as "column_name",
                    pg_catalog.format_type(a.atttypid, a.atttypmod) as "data_type", not a.attnotnull as is_nullable
                FROM
                    pg_catalog.pg_attribute a
                WHERE
                    a.attnum > 0
                    AND NOT a.attisdropped
                    AND a.attrelid = (
                        SELECT c.oid
                        FROM pg_catalog.pg_class c
                            LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = '{table_root}' and n.nspname = '{schema}'
                            AND pg_catalog.pg_table_is_visible(c.oid)
                    );
        """        
        # FIXME: support schema other than 'public'
        sql2 = f"""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_catalog='{self.auth["db_database"]}' and table_name='{table_root}';
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            return pd.DataFrame(cursor.fetchall(), columns=['column_name', 'data_type', 'is_nullable'])

    def _download_pg_table(self, table: str):
        temp_file = '/tmp/download.csv'
        # TODO: yield empty results to run the progress bar
        count = 0
        with open(temp_file, "wb") as outf:
            with self.conn.cursor() as cursor:
                with cursor.copy(f"COPY {table} TO STDOUT CSV HEADER") as copy:
                    for row in copy:
                        outf.write(row)
                        count +=1 
                        if count % 1000 == 0:
                            yield count

    def _find_sort_column(self, cols_df: pd.DataFrame, table: str):
        # Heuristic to determine an Ordering (partition) column
        # FIXME: We should use the primary key, else find a timestamp column. For now
        # we just look for an int, timestamp, or str column which is NOT NULLABLE.
        sort_col = None
        df = cols_df
        int_cols = df[(df.data_type.str.contains('int')) & (df.is_nullable==False)]
        if int_cols.shape[0] >= 1:
            sort_col = int_cols.iloc[0].column_name
        else:
            ts_cols = df[(df.data_type.str.startswith('timestamp')) & (df.is_nullable=='NO')]
            if ts_cols.shape[0] >= 1:
                sort_col = ts_cols.iloc[0].column_name
            else:
                str_cols = df[(df.data_type.str.startswith('character')) & (df.is_nullable=='NO')]
                if str_cols.shape[0] >= 1:
                    sort_col = str_cols.iloc[0].column_name

        if sort_col is None:
            raise RuntimeError("Cannot find a suitable ordering column for table " + table)
        return sort_col

    def format_select(self, col_name, col_type):
        if self.is_pg_date_type(col_type):
            return f"parseDateTimeBestEffortOrNull({col_name})"
        else:
            return col_name
        
    def load_table(self, table_root):
        # The logic here is tricky:
        # 1. Read the PG table schema and create a matching Clickhouse table, with pg types mapped to CH types
        # 2. Download the PG table in CSV format
        # 3. Now load the CSV file, but when we load we need to tell the CSV parser to treat dates (with tz's) 
        # as strings, but then apply the parseDateTimeBestEffort() function in the SELECT part of the query to
        # convert them for insert. Thus the `parser_cols` is the col name+type for the CSV parser, but the
        # `select_cols` is the same list but with the cast functions applied.
        cols_df = self._get_remote_table_columns(table_root)
        sort_col = self._find_sort_column(cols_df, table_root)
        columns = {tuple[0]: tuple[1] for tuple in cols_df.to_records(index=False)}

        select_cols = [self.format_select(col_name, col_type) for col_name, col_type in columns.items()]

        table_cols = [f"{tuple[0]} {self.pg_to_clickhouse_type(tuple[1], string_dates=False, nullable=tuple[2])}" for tuple in cols_df.to_records(index=False)]
        parser_cols = [f"{col_name} {self.pg_to_clickhouse_type(col_type, string_dates=True)}" for col_name, col_type in columns.items()]

        # Download the table data in CSV
        csv_file = '/tmp/download.csv'
        for count in self._download_pg_table(table_root):
            yield count

        table = f"{self.tenant_db}.{self.schema_name}____{table_root}"
        sql = f"""
            CREATE TABLE {table} ({", ".join(table_cols)}) ENGINE = MergeTree() ORDER BY ({sort_col})
        """
        self.db.execute_raw(sql)
        # Shell out to clickhouse-client to load the CSV file
        # FIXME: Stream to the clickhouse-driver client

        select_clause = ", ".join(select_cols)
        input_clause = ", ".join(parser_cols)

        self._clickclient(
            f"INSERT INTO {table} SELECT {select_clause} FROM input('{input_clause}') FORMAT CSVWithNames SETTINGS date_time_input_format='best_effort';", 
            csv_file
        )

        user_table = TableHandle(table_root, self.schema_name)
        real_table = CHTableHandle(user_table, tenant_id=self.db.tenant_id)
        self.db._send_signal(signal=DBSignals.TABLE_CREATE, table=real_table)

    def _clickclient(self, sql, cat_file: str):
        cclient_path = shutil.which("clickhouse-client")
        if cclient_path is None:
            raise RuntimeError("Cannot find clickhouse-client in PATH")
        
        host = os.environ['DATABASE_HOST']
        database = self.tenant_db
        user = os.environ['DATABASE_USER']
        password = os.environ['DATABASE_PASSWORD']

        # FIXME: parse --progress
        cmd = f"cat {cat_file} | clickhouse-client -h {host} -d {database} -u {user} --password '{password}' --query \"{sql}\""
        ret = subprocess.run(cmd, 
                                shell=True, capture_output=True)
        if ret.returncode != 0:
            raise RuntimeError(f"Cliclhouse import command failed {ret}: {sql}")
        return ret.stdout.decode('utf-8').strip()


    def drop_table(self, table_root: str):
        pass

    def rename_table(self, table_root: str, new_name: str):
        #implement
        pass

    def is_pg_date_type(self, pgtype):
        return pgtype.startswith("date") or pgtype.startswith("timestamp")

    def pg_to_ch_root_type(self, pgtype: str, string_dates=False):
        if pgtype.endswith("_enum"):
            return "String"
        if pgtype.startswith("boolean"):
            return "Bool"
        if pgtype.startswith("character") or pgtype.startswith("jsonb") or pgtype == "text":
            return "String"
        if pgtype.startswith("time "):
            return "String"
        if string_dates and (pgtype.startswith("date") or pgtype.startswith("timestamp")):
            return "String"
        if pgtype.startswith("date"):
            return "DateTime"
        if pgtype.startswith("timestamp"):
            return "DateTime64(3)"
        if pgtype.startswith("int") or pgtype.startswith("bigint"):
            return "Int64"
        if pgtype.startswith("smallint"):
            return "Int32"
        if pgtype.startswith("numeric") or pgtype.startswith("real") or pgtype.startswith("double"):
            return "Float64"
        if pgtype == 'tstzrange':
            return "String"
        raise RuntimeError("Unknown postgres type: " + pgtype)

    def pg_to_clickhouse_type(self, pgtype: str, string_dates=False, nullable=True):
        if pgtype.endswith("[]"):
            return "String" 
            # figure out how to parse CSV arrays. Need to sub '[' for '{' and then use JSONExtract(col,'Array(Int)')
            # "Array(" + ch_root_type(pgtype) + ")"
            # 
        else:
            roott = self.pg_to_ch_root_type(pgtype, string_dates=string_dates)
            if not nullable or roott.startswith("Array"):
                return roott
            else:
                return "Nullable(" + roott + ")"
        
