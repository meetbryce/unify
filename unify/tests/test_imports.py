import os
import shutil
import pytest

import pandas as pd

from unify import CommandInterpreter, CommandContext
from unify.search import Searcher

@pytest.fixture
def interp():
    return CommandInterpreter()

def clear_search(interp):
    interp.loader.searcher.clear_index()

# Note that the LocalFilesAdapter setup assumes files appear in $UNIFY_HOME/files,
# and our conftest.py set UNIFY_HOME to be the tests directory.
def test_csv_import(interp: CommandInterpreter):
    clear_search(interp)

    fname = "project_list.csv"
    assert os.path.exists(os.path.join(os.path.dirname(__file__), "files", fname))

    interp.run_command("drop table if exists files.project_list", interactive=False)
    context: CommandContext = interp.run_command("import project_list.csv")

    assert isinstance(context.result, pd.DataFrame)

    context = interp.run_command("show columns from files.project_list")
    assert isinstance(context.result, pd.DataFrame)
    assert context.result.shape[0] > 2

    # Check that our new table got indexed
    context = interp.run_command("search project_list")
    assert 'project_list' in "\n".join(context.logger.get_output())

    interp.run_command("drop table files.project_list", interactive=False)
    context = interp.run_command("search project_list")
    assert 'project_list' not in "\n".join(context.logger.get_output())

def test_parquest_import(interp: CommandInterpreter):
    clear_search(interp)

    fname = "gh_repos.parquet"
    assert os.path.exists(os.path.join(os.path.dirname(__file__), "files", fname))

    interp.run_command("drop table if exists files.gh_repos", interactive=False)
    context: CommandContext = interp.run_command("import gh_repos.parquet")

    assert isinstance(context.result, pd.DataFrame)

    context = interp.run_command("select name from files.gh_repos")
    assert isinstance(context.result, pd.DataFrame)
    assert context.result.shape[0] > 140

    interp.run_command("drop table files.gh_repos", interactive=False)
