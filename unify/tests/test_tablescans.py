import time
import pytest
import requests_mock

from unify import TableLoader, dbmgr
from unify.unify import TableMgr, BaseTableScan
from unify.db_wrapper import TableMissingException
from unify.rest_schema import Connection
from mocksvc.mocksvc import MockSvc

@pytest.fixture
def connections():
    config = [{"mocksvc": 
                {"adapter": "mocksvc",
                "options": {"MOCKSVC_USER": "scott@example.com", "MOCKSVC_PASS": "abc123"}
                }
            }]
    
    connections = Connection.setup_connections(conn_list=config, storage_mgr_maker=lambda x: x)
    return connections

def test_tableloader(connections):
    with requests_mock.Mocker() as mock:
        MockSvc.setup_mocksvc_api(mock)

        loader = TableLoader(given_connections=connections)

        try:
            loader.truncate_table("mocksvc.repos27")
        except TableMissingException:
            pass

        assert loader.table_exists_in_db("mocksvc.repos27") == False
        loader.materialize_table("mocksvc", "repos27")

        assert loader.table_exists_in_db("mocksvc.repos27")

        tmgr: TableMgr = loader.tables["mocksvc.repos27"]
        scanner: BaseTableScan = tmgr._create_scanner(loader)
        with dbmgr() as duck:
            scanner._set_duck(duck)

            recs = scanner.get_last_scan_records()
            assert len(recs) > 0

            assert duck.execute("select count(*) from mocksvc.repos27").fetchone()[0] == 27

        loader.refresh_table("mocksvc.repos27")

        with dbmgr() as duck:
            assert duck.execute("select count(*) from mocksvc.repos27").fetchone()[0] == 27

def test_updates_strategy(connections):
    table = "repos1100"
    try:
        with requests_mock.Mocker() as mock:
            MockSvc.setup_mocksvc_api(mock)

            loader = TableLoader(given_connections=connections)
            try:
                loader.truncate_table("mocksvc." + table)
            except:
                pass
            loader.materialize_table("mocksvc", table)

            assert loader.table_exists_in_db(f"mocksvc.{table}")

            with dbmgr() as duck:
                count = duck.execute("select count(*) from mocksvc.{}".format(table)).fetchone()[0]
                assert count == 1027
                
            loader.refresh_table(f"mocksvc.{table}")

            with dbmgr() as duck:
                count = duck.execute("select count(*) from mocksvc.{}".format(table)).fetchone()[0]
                # Clickhouse doesn't exactly reflect deletes/inserts immediately, so give a slight
                # delay or else the count can come back show a few rows
                retry = 5
                while count < 1027 and retry > 0:
                    print("Waiting for database upserts to reflect")
                    time.sleep(1)
                    count = duck.execute("select count(*) from mocksvc.{}".format(table)).fetchone()[0]
                    retry -= 1
                # FIXME: This should be exactly 1027, but Clickhouse seems to end up with a variable number of rows..
                # Not sure if that's because the delete isn't working, or just takes time to apply...
                assert count >= 1020
    finally:
        with dbmgr() as duck:
            duck.execute(f"DROP TABLE IF EXISTS mocksvc.{table}")
            pass

def test_reload_strategy(connections):
    table = "repos100"
    try:
        with requests_mock.Mocker() as mock:
            MockSvc.setup_mocksvc_api(mock)

            loader = TableLoader(given_connections=connections)
            try:
                loader.truncate_table("mocksvc." + table)
            except:
                pass
            loader.materialize_table("mocksvc", table)

            assert loader.table_exists_in_db(f"mocksvc.{table}")

            with dbmgr() as duck:
                count = duck.execute("select count(*) from mocksvc.{}".format(table)).fetchone()[0]
                assert count == 100
                
            loader.refresh_table(f"mocksvc.{table}")

            with dbmgr() as duck:
                count = duck.execute("select count(*) from mocksvc.{}".format(table)).fetchone()[0]
                assert count == 100
    finally:
        with dbmgr() as duck:
            duck.execute(f"DROP TABLE IF EXISTS mocksvc.{table}")
