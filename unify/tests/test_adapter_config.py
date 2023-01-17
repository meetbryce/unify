import importlib
import yaml
import os
import pytest

from unify.connectors import Connector, Connection, RESTView, OutputLogger
from unify.rest_connector import RESTConnector

def test_basic_auth_config():
    spec = yaml.load(importlib.resources.read_text("unify.rest_specs", "github_spec.yaml"), Loader=yaml.FullLoader)
    schema = "github"

    bad_parameters = {"foo":"bar"}
    connector2 = RESTConnector(spec, storage=None, schema_name=schema)
    connector2.resolve_auth(schema, bad_parameters)

    with pytest.raises(RuntimeError):
        connector2.validate()

    connector_parameters = {
        "username": "user1",
        "password": "token1"
    }
    connector = RESTConnector(spec, storage=None, schema_name=schema)
    connector.resolve_auth(schema, connector_parameters)
    assert 'params' in connector.auth
    assert connector.auth['params']['username'] == "user1"
    assert connector.auth['params']['password'] == "token1"

    assert connector.validate()

    
def test_headers_auth_config():
    spec = yaml.load(importlib.resources.read_text("unify.rest_specs", "hubspot_spec.yaml"), Loader=yaml.FullLoader)
    schema = "hubspot"

    bad_params = {"foo":"bar"}
    connector = RESTConnector(spec, storage=None, schema_name=schema)
    connector.resolve_auth(schema, bad_params)

    with pytest.raises(RuntimeError):
        connector.validate()

    token = "apptoken1"
    good_params = {"bearer_token": token}
    connector = RESTConnector(spec, storage=None, schema_name=schema)
    connector.resolve_auth(schema, good_params)

    assert 'params' in connector.auth
    assert connector.auth['params']['bearer_token'] == token
    assert connector.validate()
