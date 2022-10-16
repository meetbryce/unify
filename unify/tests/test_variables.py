import pytest
from unify.interpreter import CommandInterpreter
from unify.db_wrapper import dbmgr
import pandas as pd

def test_session_variables():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame

    context = interp.run_command("$user = 'joe'")
    assert 'joe' in context.lines[0]

    context = interp.run_command("show variables")
    assert 'user' in str(context.df)

    context = interp.run_command("$limit = 100")
    assert '100' in context.lines[0]

    context = interp.run_command("$limit")
    assert '100' in context.lines[0]

    context = interp.run_command("show variables")
    assert 'user' in str(context.df) and 'limit' in str(context.df)

    interp2 = CommandInterpreter()
    context = interp2.run_command("show variables")
    assert 'user' not in str(context.df)

@pytest.mark.skip(reason="")
def test_var_expressions():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame

    with dbmgr() as db:
        date_expr = db.current_date_expr()

    context = interp.run_command(f"select cast({date_expr} as VARCHAR)")
    date_str = context.df.to_records(index=False)[0][0]

    context = interp.run_command(f"$file_name = 'Date as of - ' || cast({date_expr} as varchar)")
    
    context = interp.run_command("show variables")
    recs = context.df.to_records(index=False)
    assert recs[0][0] == 'file_name'
    assert recs[0][1] == ('Date as of - ' + date_str)

    interp.run_command(
        "$tables = select table_name, table_schema from information_schema.tables"
    )

    context = interp.run_command("select * from $tables")
    assert context.df.shape[0] > 3
    assert list(context.df.columns) == ['table_name', 'table_schema']

def test_global_vars():
    interp = CommandInterpreter()
    lines: list[str]
    df: pd.DataFrame
    raw_df: pd.DataFrame

    interp.run_command("$HOST = 'api.stripe.com'")
    interp.run_command("$PORT = 8080")

    context = interp.run_command("$PORT")
    assert '8080' in context.lines[0]

    info_query = "select table_name, table_schema from information_schema.tables"
    interp.run_command(
        f"$TABLES = {info_query}"
    )

    # A second interpreter should still see the same global variables
    interp2 = CommandInterpreter()
    context = interp2.run_command("$PORT")
    assert '8080' in context.lines[0]

    context = interp2.run_command("$HOST")
    assert 'api.stripe.com' in context.lines[0]

    context = interp2.run_command("$TABLES")
    c2 = interp2.run_command(info_query)

    assert (c2.df.shape[0] - context.df.shape[0]) <= 2
    
