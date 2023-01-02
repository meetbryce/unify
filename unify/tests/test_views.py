# Test creation of views defined in Adapter specs
from distutils.cmd import Command
import pytest
import requests_mock

from mocksvc.mocksvc import MockSvc
from unify.loading import TableLoader, dbmgr
from unify.db_wrapper import dbmgr
from unify.adapters import Connection
from unify import CommandInterpreter

@pytest.fixture
def connections():
    config = [{"mocksvc": 
                {"adapter": "mocksvc",
                "options": {"username": "scott@example.com", "password": "abc123"}
                }
            }]
    
    connections = Connection.setup_connections(conn_list=config, storage_mgr_maker=lambda x: x)
    return connections

@pytest.fixture
def db():
    yield dbmgr()

def test_tableloader(connections, db):
    with requests_mock.Mocker(real_http=True) as mock:
        MockSvc.setup_mocksvc_api(mock)

        loader = TableLoader(given_connections=connections)

        conn = connections[0]
        assert conn.adapter.name == 'mocksvc'

        adapter = conn.adapter
        views = adapter.list_views()
        assert len(views) >= 2

        assert 'repos_view' in views[0].name
        assert views[0].help is not None

        query = views[0].query
        assert " repos27" in query

        schema = conn.schema_name
        query = loader.qualify_tables_in_view_query(query, views[0].from_list, schema, db.dialect())
        print(query)
        assert (" " + schema + "." + "repos27") in query
        
        query = views[1].query
        query = loader.qualify_tables_in_view_query(query, views[1].from_list, schema, db.dialect())
        print(query)
        assert (" " + schema + "." + "repos1100") in query

