import os
import lark
import pytest
import unify
from unify.interpreter import ParserVisitor

@pytest.fixture
def visitor():
    return ParserVisitor()

@pytest.fixture
def parser():
    return lark.Lark(
        open(os.path.join(os.path.dirname(unify.__file__), "grammar.lark")).read(),
        propagate_positions=True
    )

@pytest.fixture
def gsheets_url():
    return 'https://docs.google.com/spreadsheets/d/16YgB5XykiMBMXQfHk8lI9hYilJ2ctn6madVAJoKt12Q/edit#gid=1609197558'

def verify_parse(visitor, parser, rule, query, args = {}):
    ast = parser.parse(query)
    assert visitor.perform_new_visit(ast, full_code=query) == rule
    for key in args.keys():
        assert key in visitor._the_command_args
        assert visitor._the_command_args[key] == args[key]

def test_help_commands(visitor, parser):
    v = visitor
    p = parser
    verify_parse(v, p, "help", query="help")
    verify_parse(v, p, "help", query="help schemas", args={"help_choice": "schemas"})
    verify_parse(v, p, "help", query="help charts", args={"help_choice": "charts"})

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

    verify_parse(v, p, "show_variables", query="show variables")

    verify_parse(v, p, "show_files", query="show files")

    verify_parse(v, p, "show_files", query="show files from gsheets",
                args={'schema_ref':"gsheets"})

    verify_parse(v, p, "show_files", query="show files from gsheets like '%Money'",
                args={'schema_ref':"gsheets", "match_expr": "%Money"})

    verify_parse(v, p, "show_files", query="show files like '*.csv'",
                args={"match_expr": "*.csv"})

    verify_parse(v, p, "show_connections", query="show connections")

def test_open_command(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "open_command", query="open metabase", args={"open_target": "metabase"})

def test_system(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "system_command", query="system status", args={"command": "status"})
    verify_parse(v, p, "system_command", query="system stop daemon", args={"command": "stop daemon"})

def test_select(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "select_query", query="select * from table")
    verify_parse(v, p, "select_query", query="select * from sch1.table2")
    verify_parse(v, p, "select_query", query="select count(*) from sch1.table2")
    verify_parse(v, p, "select_query", query="select name, count(*) as count from sch1.table2")

    # newlines in select are ok
    verify_parse(v, p, "select_query", query="select * \nfrom table")
    verify_parse(v, p, "select_query", query="select * \nfrom table\nlimit 10")

    verify_parse(v, p, "select_query", query="select * \nfrom table\nlimit 10")
    complex_query = "select id, name, sch1.users.date from sch1.users, sch2.costs where \n" + \
        " id = 5 and name != 'scooter' and sch1.users.date is today order by sch1.users.date"
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
    verify_parse(v, p, "drop_table", query="drop table if exists gsheets.users",
        args={"table_ref": "gsheets.users"})
    verify_parse(v, p, "drop_schema", query="drop schema myscheme1",
        args={"schema_ref": "myscheme1"})
    verify_parse(v, p, "clear_table", query="clear table github.orgs", args={'table_schema_ref':"github.orgs"})
    verify_parse(v, p, "refresh_table", query="refresh table github.orgs", args={'table_ref':"github.orgs"})
    verify_parse(v, p, "reload_table", query="reload table github.orgs", args={'table_ref':"github.orgs"})
    verify_parse(v, p , "alter_table", query="alter table github.orgs rename to organizations", 
        args={'table_ref':"github.orgs", 'new_table': "organizations"})

def test_chart_commands(visitor, parser):
    v = visitor
    p = parser
    # create chart

    verify_parse(v, p, "create_chart", query="create chart")

    verify_parse(v, p, "create_chart", query="create chart as bar_chart where x = col1 and y = col2")
    verify_parse(v, p, "create_chart", 
        query="create chart chr1 from github.users as pie_chart where " +
                "x = col1 and y = col2 and x_axis_label = green",
                args={'chart_source': 'github.users', 'chart_name':"chr1", 'chart_type':"pie_chart"})

    verify_parse(v, p, "create_chart", query="create chart as pie_chart where title = 'Awesome chart'",
                args={"chart_params": {"title": "Awesome chart"}})

    verify_parse(v, p, "create_chart", query="create chart as bar_chart where x = col1 and stacked = true",
                args={"chart_params": {"stacked": "true", "x": "col1"}})

    verify_parse(v, p, "create_chart", "create chart from AWS__actualized as bar_chart where x = 'monthdate(end_date):O' and y = total and color = svc_name")

def test_export_commands(visitor, parser):
    # grammar: select_for_writing: "select" ANY ">>" adapter_ref file_ref writing_args?
    v = visitor
    p = parser

    select = "select col1, col2 from users where name ='bar' limit 5"
    sql = f"{select} >> gsheets 'foo'"

    verify_parse(v, p, "select_for_writing", sql,
            args={"select_query":select, "adapter_ref": "gsheets", "file_ref": "foo"})

    verify_parse(v, p, "export_table", "export github.coders to gsheets 'Coders'",
            args={"table_ref": "github.coders", "adapter_ref": "gsheets", "file_ref": "Coders"})

    verify_parse(v, p, "export_table", "export github.coders to gsheets 'Coders' overwrite",
            args={"write_option": "overwrite"})

    verify_parse(v, p, "export_table", "export github.coders to gsheets 'Coders' append",
            args={"write_option": "append"})

    verify_parse(v, p, "export_table", "export coders to gsheets ('Coders' || current_date || '.txt') append",
            args={"write_option": "append", "file_ref": "('Coders' || current_date || '.txt')"})

    verify_parse(v, p, "export_table", "export coders to coders.csv",
            args={"adapter_ref": None, "file_ref": "coders.csv"})

    verify_parse(v, p, "export_table", "export coders to 'subdir/coders.csv'",
            args={"adapter_ref": None, "file_ref": "subdir/coders.csv"})

def test_import_command(visitor, parser, gsheets_url):
    v = visitor
    p = parser

    verify_parse(v, p, "import_command", "import 'projects.csv'",
            args={"file_path": "projects.csv"})

    verify_parse(v, p, "import_command", f"import {gsheets_url}",
            args={"file_path": gsheets_url})

    verify_parse(v, p, "import_command", "import 'projects.csv' skip 10",
            args={"file_path": "projects.csv", "options": ['skip', '10']})
    

def test_autocomplete_parser(visitor, parser):
    # Test parser snippets use for auto-completion
    def verify_parse(rule, query):
        ast = parser.parse(query)
        assert visitor.perform_new_visit(ast, full_code=query) == rule

    verify_parse("show_tables", query="show tables  from ")

def test_variables(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "set_variable", "$foo = 100",
        args={"var_ref": "foo", "var_expression": "100"})

    verify_parse(v, p, "show_variable", "$foo",
        args={"var_ref": "foo"})

    verify_parse(v, p, "set_variable", "$foo='hello world'",
        args={"var_ref": "foo", "var_expression": "'hello world'"})

    verify_parse(v, p, "show_variable", "  $foo  ",
        args={"var_ref": "foo"})

    verify_parse(v, p, "set_variable", "  $USER = 'john ' || 'adams'",
        args={"var_ref": "USER", "var_expression": "'john ' || 'adams'"})

    q = "select id, name, date from hubspot.orders limit 10"
    verify_parse(v, p, "set_variable", f"$orders = {q}",
        args={"var_ref": "orders", "var_expression": q})

def test_email(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "email_command", "email notebook to 'joe@example.com'",
        args={"email_object": "notebook", "recipients": "joe@example.com"})
    verify_parse(v, p, "email_command", "email chart 'bugs' to 'joe@example.com'",
        args={"email_object": "chart", "recipients": "joe@example.com"})

    verify_parse(v, p, "email_command", "email github.users to 'joe@example.com' subject 'All gh users'",
        args={"email_object": "github.users", "recipients": "joe@example.com", "subject": "All gh users"})

def test_run_at_command(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "run_notebook_command", "run at 12:00",
        args={"run_at_time": "12:00"})
    verify_parse(v, p, "run_notebook_command", "run 'do the calcs' at 12:00",
        args={"run_at_time": "12:00", "notebook_path": "do the calcs"})
    verify_parse(v, p, "run_notebook_command", "run at 2022-10-23 14:23",
        args={"run_at_time": "2022-10-23 14:23"})

def test_run_every_command(visitor, parser):
    v = visitor
    p = parser

    verify_parse(v, p, "run_notebook_command", "run every week starting at 12:00",
        args={"run_at_time": "12:00", "repeater": "week"})
    verify_parse(v, p, "run_notebook_command", "run every day starting at 2021-01-05 12:00",
        args={"run_at_time": "2021-01-05 12:00", "repeater": "day"})

    verify_parse(v, p, "run_schedule", "run schedule")

    verify_parse(v, p, "delete_schedule", "run delete 'a12838'",
        args={"schedule_id": "a12838"})

def test_peek_command(visitor, parser, gsheets_url):
    v = visitor
    p = parser

    verify_parse(v, p, "peek_table", "peek github.pulls",
        args={"peek_object": "github.pulls"})
    verify_parse(v, p, "peek_table", "peek at jira.issues",
        args={"peek_object": "jira.issues"})
    verify_parse(v, p, "peek_table", "peek pulls",
        args={"peek_object": "pulls"})

    verify_parse(v, p, "peek_table", "peek github.my_pulls 10",
        args={"peek_object": "github.my_pulls", "line_count":10})

    verify_parse(v, p, "peek_table", "peek file 'projects.csv'",
        args={"qualifier": "file", "peek_object": "'projects.csv'"})

    verify_parse(v, p, "peek_table", f"peek file {gsheets_url}",
        args={"qualifier": "file", "peek_object": gsheets_url})

def test_select_parsing(visitor, parser):
    # Mostly we just pass select queries through to the underlying database, but the RESTAdapter allows populating
    # API parameters calls by issuing queries against parent tables. We want to be strict on query parsing in this
    # case so we rely on the command parser to parse the query
    v = visitor
    p = parser

    col_list = "id, name, created_at"
    table_list = "table1, new_table2"
    where_clause = "where name <> 'scooter' and id < 5"
    order_clause = "order by id desc"
    limit_clause = "50"

    verify_parse(v, p, 
        "select_query", 
        f"select {col_list} from {table_list} {where_clause} {order_clause} limit {limit_clause}",
        args={
            "col_list": ["id","name","created_at"],
            "table_list": ["table1", "new_table2"],
            "where_clause": where_clause,
            "order_clause": order_clause,
            "limit_clause": limit_clause
        }
    )

    # verify_parse(v, p,
    #     "select_query",
    #     "select cast(now() as date)",
    #     args= {
    #         "col_list": ["cast(now() as date)"]
    #     }
    # )
