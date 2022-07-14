import sys
import pytest
from gsheets.gsheets_adapter import GsheetCommandParser

@pytest.fixture
def parser():
    return GsheetCommandParser()

def test_parser(parser: GsheetCommandParser):
    def verify_parse(rule, query, args = {}):
        assert parser.parse_and_run(query, sys.stderr) == rule
        for key in args.keys():
            assert key in parser._safe.args and parser._safe.args[key] == args[key]

    verify_parse("list_files", query="list files")
    verify_parse("search", query="search Employee list",
        args={"search_query": "Employee list"})

    verify_parse("info", query="info google.com/sheets/xyz123",
        args={"file_or_gsheet_id": "google.com/sheets/xyz123"})
