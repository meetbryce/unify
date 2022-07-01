import requests
import requests_mock

from mocksvc.mocksvc import MockSvc
from rest_schema import Connector, RESTAPISpec

def test_mocksvc_config():
    config = [{"mocksvc": 
                {"spec": "mocksvc",
                "options": {"MOCKSVC_USER": "scott@example.com", "MOCKSVC_PASS": "abc123"}
                }
            }]
    
    connections = Connector.setup_connections(conn_list=config)
    assert len(connections) == 1
    assert isinstance(connections[0].spec, RESTAPISpec)
    assert connections[0].spec.name == "mocksvc"
    assert connections[0].spec.base_url == "https://mocksvc.com"

    # Verify basic auth options set properly
    assert connections[0].spec.auth['uservar'] == "scott@example.com"
    assert connections[0].spec.auth['tokenvar'] == "abc123"

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
            assert len(resp.json()) == 100
