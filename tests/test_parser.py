import os
import lark
from lark.visitors import Visitor
import pytest
from parsing_utils import find_node_return_children

class TestVisitor(Visitor):
    def clear(self):
        self.visited = []
        self.grammar_parts = {}

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
    
    def create_statement(self, tree):
        self.visited.append('create_statement')

    def delete_statement(self, tree):
        self.visited.append('delete_statement')

    def insert_statement(self, tree):
        self.visited.append('insert_statement')

    def create_chart(self, tree):
        self.visited.append('create_chart')
        print(tree)
        self.grammar_parts['chart_name'] = find_node_return_children('chart_name', tree)
        self.grammar_parts['chart_type'] = find_node_return_children('chart_type', tree)
        self.grammar_parts['chart_source'] = find_node_return_children('chart_source', tree)
        self.grammar_parts['chart_where'] = find_node_return_children('create_chart_where', tree)


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

    verify_parse("show_columns", query="show columns from sch1.table1")
    verify_parse("show_tables", query="show tables")
    verify_parse("show_tables", query="show tables from github")
    verify_parse("show_tables", query="show tables  from github_data")
    verify_parse("show_schemas", query="show schemas")
    verify_parse("show_columns", query="show columns")
    verify_parse("show_columns", query="show columns from")
    verify_parse("show_columns", query="show columns from table1")

    verify_parse("select_query", query="select * from table")
    verify_parse("describe", query="describe github")
    verify_parse("describe", query="describe github.orgs")

    # newlines in select are ok
    verify_parse("select_query", query="select * \nfrom table")
    verify_parse("select_query", query="select * \nfrom table\nlimit 10")

    verify_parse("create_statement", query="create table foo1 (id INT)")
    verify_parse("insert_statement", query="insert into foo1 (id) values (5)")
    verify_parse("delete_statement", query="delete from foo1 where id = 5")

    # create chart
    verify_parse("create_chart", query="create chart as bar_chart where x = col1 and y = col2")
    verify_parse("create_chart", 
        query="create chart chr1 from github.users as pie_chart where " +
                "x = col1 and y = col2 and x_axis_label = green")
    print(visitor.grammar_parts)

def test_autocomplete_parser(visitor, parser):
    # Test parser snippets use for auto-completion
    def verify_parse(rule, query):
        visitor.clear()
        ast = parser.parse(query)
        visitor.visit(ast)
        assert visitor.visited == [rule]

    verify_parse("show_tables", query="show tables  from ")



