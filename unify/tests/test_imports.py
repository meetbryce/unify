import os
import pytest

import pandas as pd

from unify import CommandInterpreter, CommandContext

@pytest.fixture
def interp():
    return CommandInterpreter()

# Note that the LocalFilesAdapter setup assumes files appear in $UNIFY_HOME/files,
# and our conftest.py set UNIFY_HOME to be the tests directory.
#@pytest.mark.skip(reason="")
def test_csv_import(interp: CommandInterpreter):
    fname = os.path.join(os.path.dirname(__file__), "files", "project_list.csv")
    assert os.path.exists(fname)

    interp.run_command("drop table if exists files.project_list", interactive=False)
    context: CommandContext = interp.run_command(f"import {fname}")

    assert isinstance(context.result, pd.DataFrame)

    context = interp.run_command("show columns from files.project_list")
    assert isinstance(context.result, pd.DataFrame)
    assert context.result.shape[0] > 2

def test_parquet_import(interp: CommandInterpreter):
    fname = os.path.join(os.path.dirname(__file__), "files", "gh_repos.parquet")
    assert os.path.exists(fname)

    interp.run_command("drop table if exists files.gh_repos", interactive=False)
    context: CommandContext = interp.run_command(f"import {fname}")

    assert isinstance(context.result, pd.DataFrame)

    context = interp.run_command("select name from files.gh_repos")
    assert isinstance(context.result, pd.DataFrame)
    assert context.result.shape[0] > 140

    interp.run_command("drop table files.gh_repos", interactive=False)
