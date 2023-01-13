import sys
import pytest
from gsheets_unify_adapter.gsheets_adapter import GsheetCommandParser

@pytest.fixture
def parser():
    return GsheetCommandParser()

def test_parser(parser: GsheetCommandParser):
    def verify_parse(rule, query, args = {}):
        assert parser.parse_and_run(query, sys.stderr) == rule
        for key in args.keys():
            assert key in parser._safe.args and parser._safe.args[key] == args[key]

    verify_parse("list_files", query="list files")
    verify_parse("search", query="search 'Employee list'",
        args={"search_query": "Employee list"})

    verify_parse("info", query="info 'google.com/sheets/xyz123'",
        args={"file_or_gsheet_id": "google.com/sheets/xyz123"})

    verify_parse("peek", query="peek 'Latest transactions'",
        args={"file_or_gsheet_id": "Latest transactions"})

    verify_parse("peek", query="peek 'Latest transactions' 'Sheet 5'",
        args={"file_or_gsheet_id": "Latest transactions", "tab_name": "Sheet 5"})

    verify_parse("create_table", query="create table employees from 'sheet1' tab 'tab1'",
        args={"table_name": "employees", "file_or_gsheet_id": "sheet1", "tab_name": "tab1"})

    verify_parse("create_table", query="create table fin_101 from 'Latest finances'",
        args={"table_name": "fin_101", "file_or_gsheet_id": "Latest finances", "tab_name": None})

    verify_parse("create_view", query="create view fin_102 from 'Latest finances' tab 'sheet 5'",
        args={"table_name": "fin_102", "file_or_gsheet_id": "Latest finances", "tab_name": "sheet 5"})
