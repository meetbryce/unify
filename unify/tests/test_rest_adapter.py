import base64
import io
import os
import yaml
import requests

from unify.adapters import Adapter, Connection, RESTView, OutputLogger
from unify.rest_adapter import RESTAdapter, RESTTable

def test_apispec_class():
    config = {
        "name":"github", 
        "base_url":"https://api.github.com", 
        "description":"GitHub API",
        "help": "This is the GitHub API"
    }
    spec = RESTAdapter(config, None, schema_name="github")
    assert spec.name == "github"
    assert spec.base_url == "https://api.github.com"
    assert spec.help == config["help"]

    tables = [
        {"name":"repos", "resource_path": "/repos"},
        {"name":"users", "resource_path": "/users"}
    ]

    views = [
        {"name":"repo_view", "from":"repos", "query":"select name, date"}
    ]

    rest_tables = [RESTTable(spec, t) for t in tables]
    assert rest_tables[0].name == "repos"
    assert rest_tables[0].query_path == "/repos"
    assert rest_tables[0].spec == spec

    config['tables'] = tables
    config['views'] = views

    spec = RESTAdapter(config, None, schema_name="github")
    assert len(spec.tables) == 2
    assert spec.tables[0].name == "repos"
    assert rest_tables[0].query_path == "/repos"

    assert spec.supports_commands()
    output = OutputLogger()
    spec.run_command("help", output)
    assert "\n".join(output.get_output()) == config["help"]

    assert len(spec.list_views()) == 1
    v = spec.list_views()[0]
    assert isinstance(v, RESTView)
    assert v.name == 'repo_view'
    assert v.from_list == 'repos'

def test_connector():
    fpath = os.path.join(os.path.dirname(__file__), "connections.yaml")
    connections = Connection.setup_connections(connections_path=fpath, storage_mgr_maker=lambda x: x)
    assert len(connections) > 0

    assert connections[0].adapter is not None
    assert isinstance(connections[0].adapter, Adapter)
    # load yaml file from directory relative to current file
    config = yaml.load(open(fpath), Loader=yaml.FullLoader)
    conn_config = next(conn for conn in config if next(iter(conn.keys())) == "github")
    conn_config = next(iter(conn_config.values()))
    assert "options" in conn_config

def test_rest_basic_auth():
    spec = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "../../rest_specs/github_spec.yaml")))
    schema = "github"

    username = "scottpersinger@gmail.com"
    password = "secret"
    good_params = {"username": username, "password": password}
    adapter = RESTAdapter(spec, storage=None, schema_name=schema)
    adapter.resolve_auth(schema, good_params)

    session = requests.Session()
    request = requests.Request('GET', 'https://google.com')
    adapter._setup_request_auth(session)

    prepped = session.prepare_request(request)

    assert 'Authorization' in prepped.headers
    # Encode username and password with base64
    encoded = base64.b64encode(f"{username}:{password}".encode('utf-8')).decode('utf-8')
    assert prepped.headers['Authorization'] == f"Basic {encoded}"

def test_rest_headers_auth():
    spec = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "../../rest_specs/hubspot_spec.yaml")))
    schema = "hubspot"

    bearer_token = "hubspot123"
    good_params = {"bearer_token": bearer_token}
    adapter = RESTAdapter(spec, storage=None, schema_name=schema)
    adapter.resolve_auth(schema, good_params)

    session = requests.Session()
    request = requests.Request('GET', 'https://google.com')
    adapter._setup_request_auth(session)

    prepped = session.prepare_request(request)

    assert 'Authorization' in prepped.headers
    # Encode username and password with base64
    assert prepped.headers['Authorization'] == f"Bearer {bearer_token}"
