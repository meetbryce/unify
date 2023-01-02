import yaml
import os
import pytest

from unify.adapters import Adapter, Connection, RESTView, OutputLogger
from unify.rest_adapter import RESTAdapter

def test_basic_auth_config():
    spec = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "../../rest_specs/github_spec.yaml")))
    schema = "github"

    bad_parameters = {"foo":"bar"}
    adapter2 = RESTAdapter(spec, storage=None, schema_name=schema)
    adapter2.resolve_auth(schema, bad_parameters)

    with pytest.raises(RuntimeError):
        adapter2.validate()

    adapter_parameters = {
        "username": "user1",
        "password": "token1"
    }
    adapter = RESTAdapter(spec, storage=None, schema_name=schema)
    adapter.resolve_auth(schema, adapter_parameters)
    assert 'params' in adapter.auth
    assert adapter.auth['params']['username'] == "user1"
    assert adapter.auth['params']['password'] == "token1"

    assert adapter.validate()

    
def test_headers_auth_config():
    spec = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "../../rest_specs/hubspot_spec.yaml")))
    schema = "hubspot"

    bad_params = {"foo":"bar"}
    adapter = RESTAdapter(spec, storage=None, schema_name=schema)
    adapter.resolve_auth(schema, bad_params)

    with pytest.raises(RuntimeError):
        adapter.validate()

    token = "apptoken1"
    good_params = {"bearer_token": token}
    adapter = RESTAdapter(spec, storage=None, schema_name=schema)
    adapter.resolve_auth(schema, good_params)

    assert 'params' in adapter.auth
    assert adapter.auth['params']['bearer_token'] == token
    assert adapter.validate()
