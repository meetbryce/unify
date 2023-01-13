import io
import os
import yaml

import pytest

from gsheets_unify_adapter.gsheets_adapter import GSheetsAdapter
from unify.storage_manager import MemoryStorageManager
from unify.adapters import Connection, OutputLogger

@pytest.fixture
def adapter():
    spec = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), 
        "../../rest_specs/gsheets_spec.yaml"
    )))
    conns_config = Connection.load_connections_config()
    adapter = GSheetsAdapter(spec, MemoryStorageManager(), "gsheets")
    opts = [c for c in conns_config if "gsheets" in c.keys()][0]
    adapter.resolve_auth("gsheets", opts["gsheets"]["options"])
    return adapter

def test_adapter_basic(adapter: GSheetsAdapter):
    def get_output(fun, **kwargs):
        buffer = io.StringIO()
        adapter.output = buffer
        fun(**kwargs)
        buffer.seek(0)
        return buffer.read()

    outlog = OutputLogger()
    adapter.logger = outlog

    assert adapter.name == 'gsheets'

    assert adapter.validate(silent=True)

    outlog.clear()
    adapter.list_tables()
    assert outlog.get_output() == []

    outlog.clear()
    files = adapter.list_files()
    assert len(files) > 10

    outlog.clear()
    adapter.search(search_query="transactions")
    assert len(outlog.get_output()) > 2
    for row in outlog.get_output():
        if not row.strip():
            continue
        assert "transactions" in row.lower()

    first_file = outlog.get_output()[0].split("\t")[0]
    outlog.clear()
    adapter.info(file_or_gsheet_id=first_file)
    assert 'has tabs' in "\n".join(outlog.get_output())

    outlog.clear()
    adapter.run_command(code="search 'transactions'", output_logger=outlog)
    assert len(outlog.get_output()) > 2

def test_gsheet_id_resolution(adapter: GSheetsAdapter):
    assert adapter.validate()

    assert adapter.resolve_sheet_id('Money transactions') is not None
