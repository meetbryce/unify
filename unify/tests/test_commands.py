import pytest
from unify import CommandInterpreter

@pytest.fixture
def interp():
    return CommandInterpreter()

def test_lasthelp_command(interp):
    interp.run_command("select * from information_schema.schemata")

    context = interp.run_command("??")
    output = "\n".join(context.lines)

    assert "schemata" in output


