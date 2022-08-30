import io
import os
import yaml

from unify.rest_schema import Adapter, RESTAdapter, RESTTable, Connection
from unify import OutputLogger

def test_apispec_class():
    config = {
        "name":"github", 
        "base_url":"https://api.github.com", 
        "description":"GitHub API",
        "help": "This is the GitHub API"
    }
    spec = RESTAdapter(config)
    assert spec.name == "github"
    assert spec.base_url == "https://api.github.com"
    assert spec.help == config["help"]

    tables = [
        {"name":"repos", "resource_path": "/repos"},
        {"name":"users", "resource_path": "/users"}
    ]

    rest_tables = [RESTTable(spec, t) for t in tables]
    assert rest_tables[0].name == "repos"
    assert rest_tables[0].query_path == "/repos"
    assert rest_tables[0].spec == spec

    config['tables'] = tables

    spec = RESTAdapter(config)
    assert len(spec.tables) == 2
    assert spec.tables[0].name == "repos"
    assert rest_tables[0].query_path == "/repos"

    assert spec.supports_commands()
    output = OutputLogger()
    spec.run_command("help", output)
    assert "\n".join(output.get_output()) == config["help"]


def test_connector():
    fpath = os.path.join(os.path.dirname(__file__), "connections.yaml")
    connections = Connection.setup_connections(fpath, storage_mgr_maker=lambda x: x)
    assert len(connections) > 0

    assert connections[0].adapter is not None
    assert isinstance(connections[0].adapter, Adapter)
    # load yaml file from directory relative to current file
    config = yaml.load(open(fpath), Loader=yaml.FullLoader)
    conn_config = next(conn for conn in config if next(iter(conn.keys())) == "github")
    conn_config = next(iter(conn_config.values()))
    assert "options" in conn_config





    
