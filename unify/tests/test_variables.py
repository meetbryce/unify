import pytest
from unify import CommandInterpreter, dbmgr
import pandas as pd

def test_session_variables():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame

    lines, df = interp.run_command("$user = 'joe'")
    assert 'joe' in lines[0]

    lines, df = interp.run_command("show variables")
    assert 'user' in str(df)

    lines, df = interp.run_command("$limit = 100")
    assert '100' in lines[0]

    lines, df = interp.run_command("$limit")
    assert '100' in lines[0]

    lines, df = interp.run_command("show variables")
    assert 'user' in str(df) and 'limit' in str(df)

    interp2 = CommandInterpreter()
    lines, df = interp2.run_command("show variables")
    assert 'user' not in str(df)

@pytest.mark.skip(reason="")
def test_var_expressions():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame

    with dbmgr() as db:
        date_expr = db.current_date_expr()

    lines, df = interp.run_command(f"select cast({date_expr} as VARCHAR)")
    date_str = df.to_records(index=False)[0][0]

    lines, df = interp.run_command(f"$file_name = 'Date as of - ' || cast({date_expr} as varchar)")
    
    lines, df = interp.run_command("show variables")
    recs = df.to_records(index=False)
    assert recs[0][0] == 'file_name'
    assert recs[0][1] == ('Date as of - ' + date_str)

    interp.run_command(
        "$tables = select table_name, table_schema from information_schema.tables"
    )

    lines, df = interp.run_command("select * from $tables")
    assert df.shape[0] > 3
    assert list(df.columns) == ['table_name', 'table_schema']

def test_global_vars():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame
    raw_df: pd.DataFrame

    interp.run_command("$HOST = 'api.stripe.com'")
    interp.run_command("$PORT = 8080")

    lines, df = interp.run_command("$PORT")
    assert '8080' in lines[0]

    info_query = "select table_name, table_schema from information_schema.tables"
    interp.run_command(
        f"$TABLES = {info_query}"
    )

    # A second interpreter should still see the same global variables
    interp2 = CommandInterpreter()
    lines, df = interp2.run_command("$PORT")
    assert '8080' in lines[0]

    lines, df = interp2.run_command("$HOST")
    assert 'api.stripe.com' in lines[0]

    lines, df = interp2.run_command("$TABLES")
    lines2, raw_df = interp2.run_command(info_query)

    assert df.shape[0] == raw_df.shape[0]
    
