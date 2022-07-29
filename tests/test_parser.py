import os
import lark
import pytest
from unify import ParserVisitor

@pytest.fixture
def visitor():
    return ParserVisitor()

@pytest.fixture
def parser():
    return lark.Lark(
        open(os.path.join(os.path.dirname(__file__), "../grammar.lark")).read(),
        propagate_positions=True
    )

def verify_parse(visitor, parser, rule, query, args = {}):
    ast = parser.parse(query)
    assert visitor.perform_new_visit(ast, full_code=query) == rule
    for key in args.keys():
        assert key in visitor._the_command_args and visitor._the_command_args[key] == args[key]

def test_show_commands(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "show_columns", query="show columns from sch1.table1 like '%user%'",
                args={'table_ref':"sch1.table1", "column_filter" : "%user%"})
    verify_parse(v, p, "show_columns", query="show columns from sch1.table1", 
                args={'table_ref':"sch1.table1"})
    verify_parse(v, p, "show_tables", query="show tables")
    verify_parse(v, p, "show_tables", query="show tables from github", 
                args={'schema_ref':"github"})
    verify_parse(v, p, "show_tables", query="show tables  from github_data",
                args={'schema_ref':"github_data"})
    verify_parse(v, p, "show_schemas", query="show schemas")
    verify_parse(v, p, "show_columns", query="show columns")
    verify_parse(v, p, "show_columns", query="show columns from table1",
                args={'table_ref':"table1"})
    verify_parse(v, p, "describe", query="describe github", args={'table_ref':"github"})
    verify_parse(v, p, "describe", query="describe github.orgs", args={'table_ref':"github.orgs"})

def test_select(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "select_query", query="select * from table")
    verify_parse(v, p, "select_query", query="select * from sch1.table2")

    # newlines in select are ok
    verify_parse(v, p, "select_query", query="select * \nfrom table")
    verify_parse(v, p, "select_query", query="select * \nfrom table\nlimit 10")

    verify_parse(v, p, "select_query", query="select * \nfrom table\nlimit 10")
    complex_query = "select id, name, users.date from users, costs where \n" + \
        " id = 5 and name != 'scooter' and users.date is today order by users.date"
    verify_parse(v, p, "select_query", complex_query)

def test_other_statements(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "create_statement", query="create table foo1 (id INT)")
    verify_parse(v, p, "create_view_statement", query="create view foo1 as select 5")
    verify_parse(v, p, "insert_statement", query="insert into foo1 (id) values (5)")
    verify_parse(v, p, "delete_statement", query="delete from foo1 where id = 5")
    verify_parse(v, p, "drop_table", query="drop table gsheets.users",
        args={"table_ref": "gsheets.users"})
    verify_parse(v, p, "drop_schema", query="drop schema myscheme1",
        args={"schema_ref": "myscheme1"})


    verify_parse(v, p, "clear_table", query="clear table github.orgs", args={'table_schema_ref':"github.orgs"})

def test_chart_commands(visitor, parser):
    v = visitor
    p = parser
    # create chart
    verify_parse(v, p, "create_chart", query="create chart as bar_chart where x = col1 and y = col2")
    verify_parse(v, p, "create_chart", 
        query="create chart chr1 from github.users as pie_chart where " +
                "x = col1 and y = col2 and x_axis_label = green",
                args={'chart_source': ['github', 'users'], 'chart_name':"chr1", 'chart_type':"pie_chart"})

    verify_parse(v, p, "create_chart", query="create chart as pie_chart where title = \"Awesome chart\"",
                args={"chart_params": {"title": "Awesome chart"}})

    verify_parse(v, p, "create_chart", query="create chart as bar_chart where x = col1 and stacked = true",
                args={"chart_params": {"stacked": "true", "x": "col1"}})

def test_export_commands(visitor, parser):
    # grammar: select_for_writing: "select" ANY ">>" adapter_ref file_ref writing_args?
    v = visitor
    p = parser

    select = "select col1, col2 from users where name ='bar' limit 5"
    sql = f"{select} >> gsheets 'foo'"

    verify_parse(v, p, "select_for_writing", sql,
            args={"select_query":select, "adapter_ref": "gsheets", "file_ref": "foo"})


def test_autocomplete_parser(visitor, parser):
    # Test parser snippets use for auto-completion
    def verify_parse(rule, query):
        ast = parser.parse(query)
        assert visitor.perform_new_visit(ast, full_code=query) == rule

    verify_parse("show_tables", query="show tables  from ")

