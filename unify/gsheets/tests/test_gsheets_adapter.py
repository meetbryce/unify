import io
import os
import yaml

import pytest

from gsheets.gsheets_adapter import GSheetsAdapter
from storage_manager import MemoryStorageManager

@pytest.fixture
def adapter():
    spec = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), 
        "../../rest_specs/gsheets_spec.yaml"
    )))
    return GSheetsAdapter(spec, MemoryStorageManager())

def test_adapter_basic(adapter: GSheetsAdapter):
    def get_output(fun, **kwargs):
        buffer = io.StringIO()
        adapter.output = buffer
        fun(**kwargs)
        buffer.seek(0)
        return buffer.read()

    assert adapter.name == 'gsheets'

    assert adapter.validate()

    assert adapter.list_tables() == []

    result = get_output(adapter.list_files).split("\n")
    assert len(result) > 10

    result = get_output(adapter.search, search_query="transactions").split("\n")
    assert len(result) > 2
    for row in result:
        if not row.strip():
            continue
        assert "transactions" in row.lower()

    first_file = result[0].split("\t")[0]
    result = get_output(adapter.info, file_or_gsheet_id=first_file)
    assert 'has tabs' in result

    buffer = io.StringIO()
    result = adapter.run_command(code="search 'transactions'", output_buffer=buffer)
    buffer.seek(0)
    result = buffer.read().split("\n")
    assert len(result) > 2

def test_gsheet_id_resolution(adapter: GSheetsAdapter):
    assert adapter.validate()

    assert adapter.resolve_sheet_id('Money transactions') is not None
