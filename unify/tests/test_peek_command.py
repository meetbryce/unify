import pytest
import requests_mock

from mocksvc.mocksvc import MockSvc

from unify import CommandInterpreter

def test_updates_strategy():
    table = "mocksvc.repos1100"
    c = CommandInterpreter()
    
    with requests_mock.Mocker() as mock:
        MockSvc.setup_mocksvc_api(mock)

        c.run_command(f"drop table if exists {table}")
        c.run_command(f"select * from {table}")

        lines, df = c.run_command(f"count {table}")
        assert df.to_records()[0][1] == 1027

        lines, df = c.run_command(f"peek at {table}")
        assert df.shape[0] > 10
        assert 'name' in df.columns
        assert 'id' in df.columns

        lines, df = c.run_command(f"peek at {table} 38")
        assert df.shape[0] == 38
