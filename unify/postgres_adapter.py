import logging
import typing
import re
from datetime import datetime

import pandas as pd

from .adapters import (
    Adapter, 
    AdapterQueryResult, 
    OutputLogger, 
    ReloadStrategy, 
    StorageManager, 
    TableDef, 
    TableUpdater
)
from .db_wrapper import DBManager, TableHandle, CHTableHandle, DBSignals
 
#######
# A Replicating Postgres adapter which relies on Clickhouse/DuckDB's native PG protocol support to connect
# to a target postgres server and then replicate tables into the local warehouse.
#  
# -- Replicating large tables
#
# Using Clickhouse's built-in PG adapter didn't seem to work for replicating a large Postgres table.
# So we need a more reliable method:
#
# 1. Connect to Postgres
# 2. Read the table schema (columns and types) into a schema file
#    psql --pset=format=unaligned -c '\d <table>'
# 3. Use psql to run "\copy <table>" to export the table in csv format. This may take a LONG time.
# 4. Now construct the import statement for Clickhouse:
# Extract the sort key from the PRIMARY KEY or a datetime field (created or updated)
# Create the column name,type list by reading the schema file and mapping PG -> Clickhouse types.
# Run the `create table as select * from file(<file>,CSV,<structure>) to import the data
########

PostgresAdapter = typing.NewType("PostgresAdapter", None)

class PostgresTableSpec(TableDef):
    # Represents a Postgres table on the remote server

    def __init__(self, table: str, adapter):
        super().__init__(table)
        self.adapter: PostgresAdapter = adapter
    
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
        db: DBManager = self.adapter.db
        table = TableHandle(self.schema_name, self.adapter.name)
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
        self.adapter.load_table(self.name)
        yield AdapterQueryResult(json=pd.DataFrame(), size_return=[])

class PostgresAdapter(Adapter):
    def __init__(self, spec, storage: StorageManager, schema_name: str):
        super().__init__(spec['name'], storage)
        self.auth = spec.get('auth', {}).get('params').copy()

        # connection params will live in self.auth
        self.logger: OutputLogger = None
        self.tables = None
        self.db: DBManager = storage.get_local_db()
        self.tenant_db = self.db.tenant_db
        self.schema_name = schema_name

    def validate(self) -> bool:
        self.mapped_remote_db = self.db.tenant_id + "_" + re.sub(r"[\.-]", "_", self.auth['db_host']) + "_" + self.auth['db_database']
        self.mapped_catalog = self.mapped_remote_db + "_catalog"
        self.base_url = self.auth['db_host']

        # Create the remote PG connected local db
        pw = "" if self.auth['db_password'] is None else self.auth['db_password']
        sql = f"""
        CREATE DATABASE IF NOT EXISTS {self.mapped_remote_db} 
        ENGINE = PostgreSQL(
                    '{self.auth["db_host"]}',
                    '{self.auth["db_database"]}',
                    '{self.auth["db_user"]}',
                    '{pw}'
                )
        """
        sql2 = f"""
        CREATE DATABASE IF NOT EXISTS {self.mapped_catalog} 
        ENGINE = PostgreSQL(
                    '{self.auth["db_host"]}',
                    '{self.auth["db_database"]}',
                    '{self.auth["db_user"]}',
                    '{pw}',
                    'information_schema'
                )
        """
        self.db.execute_raw(sql)
        self.db.execute_raw(sql2)
        return True

    def list_tables(self):
        # select tables names through the remote PG connection and return TableDefs for each one
        tables = self.db.execute_raw(f"SHOW TABLES FROM {self.mapped_remote_db}").to_records()
        return [PostgresTableSpec(t[1], self) for t in tables]

    def _get_remote_table_columns(self, table_root):
        # Find the table columns and types so we can pick a sort order
        sql = f"""
            SELECT column_name, data_type, is_nullable 
            FROM {self.mapped_catalog}.columns 
            WHERE table_catalog='{self.auth["db_database"]}' and table_name='{table_root}';
        """
        return self.db.execute_raw(sql)

    def load_table(self, table_root):
        # FIXME: We should use the primary key, else find a timestamp column. For now
        # we just look for an int, timestamp, or str column which is NOT NULLABLE.
        df = self._get_remote_table_columns(table_root)
        sort_col = None
        int_cols = df[(df.data_type.str.startswith('int')) & (df.is_nullable=='NO')]
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
            raise RuntimeError("Cannot find a suitable ordering column for table " + table_root)        

        sql = f"""
           CREATE TABLE IF NOT EXISTS {self.tenant_db}.{self.schema_name}____{table_root}  
             Engine=MergeTree() ORDER BY ({sort_col})
            AS SELECT * FROM {self.mapped_remote_db}.{table_root}
        """
        print(sql)
        self.db.execute_raw(sql)
        user_table = TableHandle(table_root, self.schema_name)
        real_table = CHTableHandle(user_table, tenant_id=self.db.tenant_id)
        self.db._send_signal(signal=DBSignals.TABLE_CREATE, table=real_table)

        # FIXME: check for errors

    def drop_table(self, table_root: str):
        pass

    def rename_table(self, table_root: str, new_name: str):
        #implement
        pass

    # TODO: Support exporting data
