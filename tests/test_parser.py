import os
import lark
from lark.visitors import Visitor
import pytest

class TestVisitor(Visitor):
    def clear(self):
        self.visited = []

    def show_tables(self, tree):
        self.visited.append('show_tables')

    def show_schemas(self, tree):
        self.visited.append('show_schemas')

    def show_columns(self, tree):
        self.visited.append('show_columns')

    def describe(self, tree):
        self.visited.append('describe')

    def select_query(self, tree):
        self.visited.append('select_query')
    
@pytest.fixture
def visitor():
    return TestVisitor()

@pytest.fixture
def parser():
    return lark.Lark(open(os.path.join(os.path.dirname(__file__), "../grammar.lark")).read())

def test_parser(visitor, parser):
    def verify_parse(rule, query):
        visitor.clear()
        ast = parser.parse(query)
        visitor.visit(ast)
        assert visitor.visited == [rule]

    verify_parse("show_tables", query="show tables")
    verify_parse("show_tables", query="show tables from github")
    verify_parse("show_tables", query="show tables  from github_data")
    verify_parse("show_schemas", query="show schemas")
    verify_parse("show_columns", query="show columns")
    verify_parse("show_columns", query="show columns from table1")
    verify_parse("show_columns", query="show columns from sch1.table1")

    verify_parse("select_query", query="select * from table")
    verify_parse("describe", query="describe github")
    verify_parse("describe", query="describe github.orgs")

    # newlines in select are ok
    verify_parse("select_query", query="select * \nfrom table")
    verify_parse("select_query", query="select * \nfrom table\nlimit 10")

def test_autocomplete_parser(visitor, parser):
    # Test parser snippets use for auto-completion
    def verify_parse(rule, query):
        visitor.clear()
        ast = parser.parse(query)
        visitor.visit(ast)
        assert visitor.visited == [rule]

    verify_parse("show_tables", query="show tables  from ")



