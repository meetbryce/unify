import pytest
from datetime import datetime

from unify.connectors import Connection
from unify.postgres_connector import PostgresConnector
from unify.loading import TableLoader, LoaderJob
from unify.db_wrapper import TableMissingException, TableHandle, DuckDBWrapper
from unify.sqla_storage_manager import UnifyDBStorageManager
from unify import dbmgr

@pytest.fixture
def db():
    with dbmgr() as db:
        yield db

@pytest.fixture
def connection(db):
    config = [{"postgres": 
                {"connector": "postgres",
                 "options": {"db_host": "localhost", "db_database":"dvdrental", "db_user":"scottp", "db_password":""}
                }
            }]
    
    connections = Connection.setup_connections(
        conn_list=config, 
        storage_mgr_maker=lambda schema: UnifyDBStorageManager(schema, db)
    )
    return connections[0]

@pytest.mark.skip("Need to rewrite PG connector to not use Clickhouse primitives")
def test_postgres_connector(db, connection):
    loader = TableLoader(given_connections=[connection])

    try:
        loader.truncate_table("postgres.actor")
    except TableMissingException:
        pass
    loader.materialize_table(LoaderJob.load_table_job("postgres", "actor"))

    # Test our update strategy
    pg_connector: PostgresConnector = connection.connector
    assert isinstance(pg_connector, PostgresConnector)

    table_def = [t for t in pg_connector.list_tables() if t.name == 'actor'][0]
    table_def.get_table_updater(datetime.utcnow())
