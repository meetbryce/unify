import sys
import logging
import pytest
import requests
import requests_mock

from mocksvc.mocksvc import MockSvc
from unify.adapters import Connection, Adapter
from unify.rest_adapter import RESTAdapter

logger = logging.getLogger(__name__)

@pytest.fixture
def connection():
    config = [{"mocksvc": 
                {"adapter": "mocksvc",
                "options": {"username": "scott@example.com", "password": "abc123"}
                }
            }]
    
    connections = Connection.setup_connections(conn_list=config, storage_mgr_maker=lambda x: x)
    return connections[0]

def test_mocksvc_config(connection):
    assert isinstance(connection.adapter, RESTAdapter)
    assert connection.adapter.name == "mocksvc"
    assert connection.adapter.base_url == "https://mocksvc.com"

    # Verify basic auth options set properly
    assert connection.adapter.auth['params']['username'] == "scott@example.com"
    assert connection.adapter.auth['params']['password'] == "abc123"

def test_mocksvc_requests_mock():
    with requests_mock.Mocker() as mock:
        MockSvc.setup_mocksvc_api(mock)

        resp = requests.get("https://mocksvc.com/api/ping")
        assert resp.status_code == 200
        assert resp.text == "pong"

        auth = ("scott@example.com", "abc123")
        resp = requests.get("https://mocksvc.com/api/repos_27", auth=auth)
        assert resp.status_code == 200

        assert len(resp.json()) == 27

        resp = requests.get("https://mocksvc.com/api/repos_100", auth=auth)
        assert resp.status_code == 200
        assert len(resp.json()) == 100

        resp = requests.get("https://mocksvc.com/api/repos_1100", auth=auth)
        assert resp.status_code == 200
        assert len(resp.json()) == 100

        for page in range(1, 12):
            params = {"page":page, "count":100}
            resp = requests.get("https://mocksvc.com/api/repos_1100", auth=auth, params=params)
            assert resp.status_code == 200
            assert len(resp.json()) == (100 if page < 11 else 27)

def test_calling_rest_api(connection):
    with requests_mock.Mocker() as mock:
        MockSvc.setup_mocksvc_api(mock)

        table_spec = connection.adapter.lookupTable("repos100")
        total_records = 0
        for qres in table_spec.query_resource(None, logger):
            total_records += len(qres.json)
            qres.size_return.append(len(qres.json))
        assert total_records == 100

        table_spec = connection.adapter.lookupTable("repos27")
        total_records = 0
        for qres in table_spec.query_resource(None, logger):
            total_records += len(qres.json)
            qres.size_return.append(len(qres.json))
        assert total_records == 27

        table_spec = connection.adapter.lookupTable("repos1100")
        total_records = 0
        for qres in table_spec.query_resource(None, logger):
            total_records += len(qres.json)
            qres.size_return.append(len(qres.json))
        assert total_records == 1027
