import pytest
import requests
import requests_mock

from mocksvc.mocksvc import MockSvc
from rest_schema import Connector, RESTAPISpec

@pytest.fixture
def connection():
    config = [{"mocksvc": 
                {"spec": "mocksvc",
                "options": {"MOCKSVC_USER": "scott@example.com", "MOCKSVC_PASS": "abc123"}
                }
            }]
    
    connections = Connector.setup_connections(conn_list=config)
    return connections[0]

def test_mocksvc_config(connection):
    assert isinstance(connection.spec, RESTAPISpec)
    assert connection.spec.name == "mocksvc"
    assert connection.spec.base_url == "https://mocksvc.com"

    # Verify basic auth options set properly
    assert connection.spec.auth['uservar'] == "scott@example.com"
    assert connection.spec.auth['tokenvar'] == "abc123"

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

        table_spec = connection.spec.lookupTable("repos100")
        total_records = 0
        for json, size_return in table_spec.query_resource(None):
            total_records += len(json)
            size_return.append(len(json))
        assert total_records == 100

        table_spec = connection.spec.lookupTable("repos27")
        total_records = 0
        for json, size_return in table_spec.query_resource(None):
            total_records += len(json)
            size_return.append(len(json))
        assert total_records == 27

        table_spec = connection.spec.lookupTable("repos1100")
        total_records = 0
        for json, size_return in table_spec.query_resource(None):
            total_records += len(json)
            size_return.append(len(json))
        assert total_records == 1027
