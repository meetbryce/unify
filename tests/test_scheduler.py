from unify import CommandInterpreter, dbmgr
import pandas as pd

def test_scheduler_commands():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame

    lines, df = interp.run_command("run 'notebook 1' at 23:00")
    lines, df = interp.run_command("run 'notebook 2' every day starting at 08:00")

    lines, df = interp.run_command("run schedule")
    found1 = found2 = False
    id1 = id2 = None
    for row in df.to_records(index=False):
        if 'notebook 1' in str(row):
            found1 = True
            id1 = row['schedule_id']
        if 'notebook 2' in str(row):
            found2 = True
            id2 = row['schedule_id']
    assert found1
    assert found2

    lines, df = interp.run_command("run delete {}".format(id1))
    lines, df = interp.run_command("run delete {}".format(id2))



