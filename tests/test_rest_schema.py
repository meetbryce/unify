import os
import yaml

from rest_schema import RESTAPISpec, RESTTable, RESTCol, Connector

def test_apispec_class():
    config = {"name":"github", "base_url":"https://api.github.com", "description":"GitHub API"}
    spec = RESTAPISpec(config)
    assert spec.name == "github"
    assert spec.base_url == "https://api.github.com"

    tables = [
        {"name":"repos", "resource_path": "/repos"},
        {"name":"users", "resource_path": "/users"}
    ]

    rest_tables = [RESTTable(spec, t) for t in tables]
    assert rest_tables[0].name == "repos"
    assert rest_tables[0].query_path == "/repos"
    assert rest_tables[0].spec == spec

    config['tables'] = tables

    spec = RESTAPISpec(config)
    assert len(spec.tables) == 2
    assert spec.tables[0].name == "repos"
    assert rest_tables[0].query_path == "/repos"

def test_connector():
    fpath = os.path.join(os.path.dirname(__file__), "../connections.yaml")
    connections = Connector.setup_connections(fpath)
    assert len(connections) > 0

    assert connections[0].spec is not None
    assert isinstance(connections[0].spec, Connector)
    # load yaml file from directory relative to current file
    config = yaml.load(open(fpath), Loader=yaml.FullLoader)
    conn_config = next(conn for conn in config if next(iter(conn.keys())) == "github")
    conn_config = next(iter(conn_config.values()))
    assert "options" in conn_config





    
