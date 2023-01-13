import os
import pytest
import requests_mock

from mocksvc.mocksvc import MockSvc

from unify import CommandInterpreter

def test_updates_strategy():
    os.environ.pop('UNIFY_SKIP_COLUMN_INTEL', None) # need column intel for peek

    table = "mocksvc.repos1100"
    c = CommandInterpreter()
    
    with requests_mock.Mocker(real_http=True) as mock:
        MockSvc.setup_mocksvc_api(mock)

        c.run_command(f"drop table if exists {table}", interactive=False)
        c.run_command(f"select * from {table}")

        context = c.run_command(f"count {table}")
        assert context.df.to_records()[0][1] == 1027

        context = c.run_command(f"peek at {table}")
        # assert context.df.shape[0] > 10
        # assert len(context.df.columns) > 2
        # #assert 'name' in df.columns
        # #assert 'id' in df.columns

        # context = c.run_command(f"peek at {table} 38")
        # assert context.df.shape[0] == 38

        # TODO: re-enable once we fix peek at tables