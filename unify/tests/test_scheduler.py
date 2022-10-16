from unify.interpreter import CommandInterpreter
from unify import dbmgr
import pandas as pd

def test_scheduler_commands():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame

    nb1 = "Incident Stats.ipynb"
    nb2 = "Coder Stats.ipynb"
    context = interp.run_command(f"run '{nb1}' at 23:00")
    context = interp.run_command(f"run '{nb2}' every day starting at 08:00")

    context = interp.run_command("run schedule")
    found1 = found2 = False
    id1 = id2 = None
    for row in context.df.to_records(index=False):
        if nb1 in str(row):
            found1 = True
            id1 = row[0]
        if nb2 in str(row):
            found2 = True
            id2 = row[0]
    assert found1
    assert found2

    context = interp.run_command("run delete '{}'".format(id1))
    context = interp.run_command("run delete '{}'".format(id2))

def test_scheduler_backend():
    # We want to schedule a few notebooks to run on different schedules, then interrogate
    # the scheduler package and verify that the notebook jobs are scheduled at the right times.
    #
    # Note that we don't try to actually run the jobs nor muck with "wall time".
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame

    nb1 = "Incident Stats.ipynb"
    nb2 = "Coder Stats.ipynb"

    context = interp.run_command(f"run '{nb1}' at 23:00")
    context = interp.run_command(f"run '{nb2}' every day starting at 08:00")




