import os
import lark
import pytest
from unify import ParserVisitor

@pytest.fixture
def visitor():
    return ParserVisitor()

@pytest.fixture
def parser():
    return lark.Lark(open(os.path.join(os.path.dirname(__file__), "../grammar.lark")).read())

def test_parser(visitor, parser):
    def verify_parse(rule, query, args = {}):
        ast = parser.parse(query)
        assert visitor.perform_new_visit(ast) == rule
        for key in args.keys():
            assert key in visitor._the_command_args and visitor._the_command_args[key] == args[key]

    verify_parse("show_columns", query="show columns from sch1.table1 like '%user%'",
                args={'table_ref':"sch1.table1", "column_filter" : "%user%"})
    verify_parse("show_columns", query="show columns from sch1.table1", 
                args={'table_ref':"sch1.table1"})
    verify_parse("show_tables", query="show tables")
    verify_parse("show_tables", query="show tables from github", 
                args={'schema_ref':"github"})
    verify_parse("show_tables", query="show tables  from github_data",
                args={'schema_ref':"github_data"})
    verify_parse("show_schemas", query="show schemas")
    verify_parse("show_columns", query="show columns")
    verify_parse("show_columns", query="show columns from")
    verify_parse("show_columns", query="show columns from table1",
                args={'table_ref':"table1"})

    verify_parse("select_query", query="select * from table")
    verify_parse("describe", query="describe github", args={'table_ref':"github"})
    verify_parse("describe", query="describe github.orgs", args={'table_ref':"github.orgs"})

    # newlines in select are ok
    verify_parse("select_query", query="select * \nfrom table")
    verify_parse("select_query", query="select * \nfrom table\nlimit 10")

    verify_parse("create_statement", query="create table foo1 (id INT)")
    verify_parse("create_view_statement", query="create view foo1 as select 5")
    verify_parse("insert_statement", query="insert into foo1 (id) values (5)")
    verify_parse("delete_statement", query="delete from foo1 where id = 5")
    verify_parse("drop_schema", query="drop schema myscheme1",
        args={"schema_ref": "myscheme1"})


    verify_parse("clear_table", query="clear table github.orgs", args={'table_schema_ref':"github.orgs"})

    # create chart
    verify_parse("create_chart", query="create chart as bar_chart where x = col1 and y = col2")
    verify_parse("create_chart", 
        query="create chart chr1 from github.users as pie_chart where " +
                "x = col1 and y = col2 and x_axis_label = green",
                args={'chart_source': ['github', 'users'], 'chart_name':"chr1", 'chart_type':"pie_chart"})

    verify_parse("create_chart", query="create chart as pie_chart where title = \"Awesome chart\"",
                args={"chart_params": {"title": "Awesome chart"}})

    verify_parse("create_chart", query="create chart as bar_chart where x = col1 and stacked = true",
                args={"chart_params": {"stacked": "true", "x": "col1"}})

def test_autocomplete_parser(visitor, parser):
    # Test parser snippets use for auto-completion
    def verify_parse(rule, query):
        ast = parser.parse(query)
        assert visitor.perform_new_visit(ast) == rule

    verify_parse("show_tables", query="show tables  from ")

