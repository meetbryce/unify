import inspect
import os
from pprint import pprint
import re
import functools
import time
import threading
import typing
from typing import Dict, Generator, Iterable
import webbrowser

import pandas as pd
from sqlalchemy.orm.session import Session
from sqlalchemy import select
from prompt_toolkit.completion import NestedCompleter, WordCompleter, Completer, CompleteEvent, Completion
from prompt_toolkit.document import Document

import lark
from lark.lark import Lark
from lark.visitors import Visitor
from lark.visitors import v_args

from http.server import HTTPServer
from ReadEm.serve import MDRequestHandler

from .adapters import Adapter, OutputLogger, Connection
from .loading import TableLoader, TableExporter, LoaderJob, add_logging_handler
from .db_wrapper import DBSignals, TableHandle, TableMissingException, dbmgr, SavedVar, RunSchedule, ColumnInfo
from .file_adapter import LocalFileAdapter
from .metabase_setup import MetabaseSetup

from .parsing_utils import (
    find_subtree, 
    find_node_return_child, 
    find_node_return_children,
    collect_child_strings,
    collect_child_text,
    collect_child_string_list
)

class ParserVisitor(Visitor):
    """ Utility class for visiting our parse tree and assembling the relevant parts
        for the parsed command. Also works on incomplete results so can be used
        for autocompletion in the Jupyter kernel.
    """
    MATPLOT_CHART_MAP: Dict[str, str] = {
        'bar_chart': 'bar',
        'pie_chart': 'pie',
        'line_chart': 'line',
        'area_chart': 'area',
        'hbar_chart': 'barh'
    }

    def perform_new_visit(self, parse_tree, full_code):
        self._the_command = None
        self._the_command_args = {}
        self._full_code = full_code
        try:
            self.visit(parse_tree)
        except Exception as e:
            print(parse_tree.pretty)
            raise
        return self._the_command

    def clear_table(self, tree):
        self._the_command = 'clear_table'
        self._the_command_args['table_schema_ref'] = find_node_return_children("table_schema_ref", tree)
        if self._the_command_args['table_schema_ref']:
            self._the_command_args['table_schema_ref'] = ".".join(self._the_command_args['table_schema_ref'])
        return tree

    def show_connections(self, tree):
        self._the_command = 'show_connections'
        return tree

    def connect_command(self, tree):
        self._the_command = 'connect_command'
        return tree

    def count_table(self, tree):
        self._the_command = "count_table"
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )
        return tree

    def create_chart(self, tree):
        self._the_command = 'create_chart'
        self._the_command_args['chart_name'] = find_node_return_child('chart_name', tree)
        self._the_command_args['chart_type'] = find_node_return_child('chart_type', tree)
        self._the_command_args['chart_source'] = collect_child_text(
            "chart_source", 
            tree, 
            full_code=self._full_code
        )

        where_clause = find_subtree('create_chart_where', tree)
        # collect chart params

        params = {}
        if where_clause:
            key = value = None
            for child in where_clause.children:
                key = key or find_node_return_child("chart_param", child)
                value = value or find_node_return_child("param_value", child)
                if value is not None:
                    value = value.strip("'")
                if key and value:
                    params[key] = value
                    key = value = None
        self._the_command_args['chart_params'] = params

    def create_statement(self, tree):
        self._the_command = 'create_statement'

    def create_view_statement(self, tree):
        self._the_command = 'create_view_statement'

    def delete_connection(self, tree):
        self._the_command = 'delete_connection'
        return tree

    def delete_schedule(self, tree):
        self._the_command = 'delete_schedule'
        self._the_command_args['schedule_id'] = find_node_return_child("schedule_ref", tree).strip("'")

    def delete_statement(self, tree):
        self._the_command = 'delete_statement'

    def describe(self, tree):
        self._the_command = 'describe'
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )
        return tree

    def drop_table(self, tree):
        self._the_command = "drop_table"
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )

    def drop_schema(self, tree):
        self._the_command = "drop_schema"
        self._the_command_args["schema_ref"] = find_node_return_child("schema_ref", tree)
        self._the_command_args["cascade"] = self._full_code.strip().lower().endswith("cascade")

    def email_command(self, tree):
        self._the_command = "email_command"
        self._the_command_args['email_object'] = collect_child_text("email_object", tree, self._full_code)
        self._the_command_args['recipients'] = find_node_return_child("recipients", tree).strip("'")
        subject = find_node_return_child("subject", tree)
        if subject:
            self._the_command_args['subject'] = subject.strip("'")

    def export_table(self, tree):
        self._the_command = "export_table"
        self._the_command_args['table_ref'] = collect_child_text("table_ref", tree, full_code=self._full_code)
        self._the_command_args['adapter_ref'] = find_node_return_child("adapter_ref", tree)
        fileref = find_node_return_child("file_ref", tree)
        if fileref:
            self._the_command_args['file_ref'] = fileref.strip("'")
        else:
            self._the_command_args['file_ref'] = self._the_command_args['adapter_ref'].strip("'")
            self._the_command_args['adapter_ref'] = None
        self._the_command_args['write_option'] = find_node_return_child("write_option", tree)

    def help(self, tree):
        self._the_command = 'help'
        self._the_command_args['help_choice'] = collect_child_strings('HELP_CHOICE', tree)

    def help_last(self, tree):
        self._the_command = "help_last"

    def import_command(self, tree):
        self._the_command = 'import_command'
        self._the_command_args['file_path'] = collect_child_text("file_path", tree, full_code=self._full_code).strip("'")
        opts = find_node_return_children("options", tree)
        if opts:
            opts = re.split(r"\s+", opts[0])
            self._the_command_args['options'] = [o for o in opts if o]

    def insert_statement(self, tree):
        self._the_command = 'insert_statement'

    def show_files(self, tree):
        self._the_command = 'show_files'
        self._the_command_args['schema_ref'] = find_node_return_child("schema_ref", tree)
        filter = collect_child_strings("match_expr", tree)
        if filter:
            self._the_command_args['match_expr'] = filter.strip()

    def open_command(self, tree):
        self._the_command = 'open_command'
        self._the_command_args['open_target'] = collect_child_text("open_target", tree, full_code=self._full_code).strip("'")

    def peek_table(self, tree):
        self._the_command = 'peek_table'
        self._the_command_args['qualifier'] = \
            collect_child_text("qualifier", tree, full_code=self._full_code)
        self._the_command_args['peek_object'] = \
            collect_child_text("peek_object", tree, full_code=self._full_code)
        count = find_node_return_child("line_count", tree)
        if count:
            self._the_command_args['line_count'] = int(count)

    def refresh_table(self, tree):
        self._the_command = 'refresh_table'
        self._the_command_args['table_ref'] = \
            collect_child_text("table_ref", tree, full_code=self._full_code)

    def alter_table(self, tree):
        self._the_command = 'alter_table'
        self._the_command_args['table_ref'] = \
            collect_child_text("table_ref", tree, full_code=self._full_code)
        self._the_command_args['new_table'] = \
            collect_child_text("new_table", tree, full_code=self._full_code)

    def reload_table(self, tree):
        self._the_command = 'reload_table'
        self._the_command_args['table_ref'] = \
            collect_child_text("table_ref", tree, full_code=self._full_code)

    def run_notebook_command(self, tree):
        self._the_command = 'run_notebook_command'
        nb = find_node_return_child("notebook_ref", tree)
        if nb:
            self._the_command_args["notebook_path"] = nb.strip("'")
        dt = collect_child_strings('datetime', tree)
        self._the_command_args["run_at_time"] = dt
        if find_subtree('run_every_command', tree):
            self._the_command_args["repeater"] = collect_child_strings("repeater", tree)

    def run_schedule(self, tree):
        self._the_command = 'run_schedule'

    def run_info(self, tree):
        self._the_command = 'run_info'
        self._the_command_args['schedule_id'] = find_node_return_child("schedule_ref", tree).strip("'")

    def search_command(self, tree):
        self._the_command = 'search_command'
        query = collect_child_strings('query', tree)
        if query:
            query = query = query.strip("'")
            self._the_command_args['query'] = query


    def select_query(self, tree):
        # lark: select_query: "select" WS col_list WS "from" WS table_list (WS where_clause)? (WS order_clause)? (WS limit_clause)?
        self._the_command = 'select_query'
        cols = collect_child_string_list("col_list", tree)
        cols = [c for c in cols if c.strip() != ""]
        self._the_command_args["col_list"] = cols
        tabs = collect_child_string_list("table_list", tree)
        tabs = [t for t in tabs if t.strip() != ""]
        self._the_command_args["table_list"] = tabs
        self._the_command_args["where_clause"] = collect_child_text("where_clause", tree, self._full_code)
        self._the_command_args["order_clause"] = collect_child_text("order_clause", tree, self._full_code)
        lim = collect_child_strings("limit_clause", tree)
        if lim:
            self._the_command_args["limit_clause"] = lim.strip()

    
    def select_for_writing(self, tree):
        self._the_command = "select_for_writing"
        self._the_command_args["adapter_ref"] = find_node_return_child("adapter_ref", tree)
        self._the_command_args["file_ref"] = find_node_return_child("file_ref", tree).strip("'")
        self._the_command_args["select_query"] = collect_child_text("select_query", tree, self._full_code)

    def set_variable(self, tree):
        self._the_command = "set_variable"
        self._the_command_args["var_ref"] = find_node_return_child("var_ref", tree)
        self._the_command_args["var_expression"] = collect_child_text("var_expression", tree, self._full_code).strip()

    def show_columns(self, tree):
        """" Always returns 'table_ref' either qualified or unqualified by the schema name."""
        self._the_command = 'show_columns'
        self._the_command_args['table_ref'] = collect_child_text(
            "table_ref", 
            tree, 
            full_code=self._full_code
        )
        filter = collect_child_strings("column_filter", tree)
        if filter:
            self._the_command_args['column_filter'] = filter.strip()
        return tree

    def show_schemas(self, tree):
        self._the_command = 'show_schemas'
        return tree

    def show_tables(self, tree):
        self._the_command = 'show_tables'
        self._the_command_args['schema_ref'] = find_node_return_child("schema_ref", tree)
        return tree

    def show_variable(self, tree):
        self._the_command = "show_variable"
        self._the_command_args["var_ref"] = find_node_return_child("var_ref", tree)        

    def show_variables(self, tree):
        self._the_command = "show_variables"

    def system_command(self, tree):
        self._the_command = "system_command"
        self._the_command_args["command"] = collect_child_text("sys_command", tree, self._full_code)        

class CommandContext:
    def __init__(self, command: str, input_func=None, get_notebook_func=None, interactive: bool=True):
        self.has_run = False
        self.command = command
        self.input_func = input_func
        self.get_notebook_func = get_notebook_func
        self.interactive = interactive
        self.logger: OutputLogger = OutputLogger()
        self.print_buffer = []
        self.parser_visitor = ParserVisitor()
        self.result: pd.DataFrame = None
        self.interp_command = None
        self.lark_parse_error = None
        self.print_wide = False
        self.recent_tables: list[TableHandle] = []

    @property
    def df(self) -> pd.DataFrame:
        return self.result

    @property 
    def lines(self) -> list[str]:
        return self.logger.get_output()

    def mark_ran(self):
        self.has_run = True

    def parse_command(self, parser):
        parse_tree = parser.parse(self.command)
        self.interp_command = self.parser_visitor.perform_new_visit(parse_tree, full_code=self.command)

    def starts_with_bang(self):
        return self.command.startswith("!")

    def add_notebook_path_to_args(self):
        nb_path = self.get_notebook_func() if self.get_notebook_func else None
        if 'notebook_path' not in self.parser_visitor._the_command_args:
            self.parser_visitor._the_command_args['notebook_path'] = nb_path

    def should_run_after_db_closed(self):
        return self.interp_command == 'email_command' and \
            self.parser_visitor._the_command_args.get('email_object') == 'notebook'

    def get_input(self, prompt: str):
        if self.input_func:
            return self.input_func(prompt)
        else:
            raise RuntimeError("Input requested by session is non-interactive")

    def set_recent_tables(self, tables: list[TableHandle]):
        self.recent_tables.extend(tables)


class TableListCompleter(Completer):
    def __init__(self, table_list) -> None:
        super().__init__()
        self._table_list = table_list

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        if document.text.endswith("from ") or document.text.endswith("from"):
            offset = document.text.endswith(" ") and "" or " "
            for table in self._table_list:
                yield Completion(offset + table, start_position=0)
        elif re.match(r".*from\s+\S+$", document.text):
            prefix = document.text.split(" ")[-1]
            for table in self._table_list:
                if table.startswith(prefix):
                    yield Completion(table[len(prefix):], start_position=0)

class DottedWordCompleter(NestedCompleter):
    # Unfortuante monkey patch so NestedCompleter configures the final WordCompleter
    # to allow dots in words instead of considering it a separator.
    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        # Split document.
        text = document.text_before_cursor.lstrip()
        stripped_len = len(document.text_before_cursor) - len(text)

        # If there is a space, check for the first term, and use a
        # subcompleter.
        if " " in text:
            first_term = text.split(" ")[0]
            completer = self.options.get(first_term)

            # If we have a sub completer, use this for the completions.
            if completer is not None:
                remaining_text = text[len(first_term) :].lstrip()
                move_cursor = len(text) - len(remaining_text) + stripped_len

                new_document = Document(
                    remaining_text,
                    cursor_position=document.cursor_position - move_cursor,
                )

                yield from completer.get_completions(new_document, complete_event)

        # No space in the input: behave exactly like `WordCompleter`.
        else:
            completer = WordCompleter(
                list(self.options.keys()), ignore_case=self.ignore_case, WORD=True
            )
            yield from completer.get_completions(document, complete_event)


class CommandInterpreter:
    """
        The interpreter for Unify. You call `run_command` with the code you want to execute
        and this class will parse it and execute the command, returning the result as a
        tuple of (output_lines[], result_object) where result_object is usually a DataFrame
        but could be a dict containing an image result instead.
    """
    _last_result: pd.DataFrame = None

    def __init__(self, silence_errors=False):
        self.parser = None # defer loading grammar until we need it
        self.loader: TableLoader = TableLoader(silence_errors, leader=True)
        self.adapters: dict[str, Adapter] = self.loader.adapters
        self.context: CommandContext = None
        self.session_vars: dict[str, object] = {}
        self._email_helper = None
        self.duck: DBManager = None
        self.last_chart = None
        # Some commands we only support in interactive sessions. In non-interactive cases (background execution)
        # these commands will be no-ops.
        self.commands_needing_interaction = [
            'run_notebook_command'
        ]
        self.recent_tables: list[TableHandle] = []

    def run_command(
        self, 
        cmd, 
        input_func=input, 
        get_notebook_func=None, 
        interactive: bool=True) -> CommandContext:
        """ Runs a command through the execution pipeline. Returns the CommandContext
            created to manage the pipeline.
        """

        context = CommandContext(cmd, input_func, get_notebook_func, interactive)
        self.context = context
        self._debug_command_pipeline = False

        pipeline = [
            self.check_debug_command,
            self.check_print_wide,
            self.run_command_direct_to_db,
            self.substitute_variables,
            self.replace_last_table_reference,
            self.run_adapter_commands,
            self.lark_parse_command,
            self.skip_non_interactive,
            self.run_interp_commands
        ]
        with dbmgr() as db:
            self._tenant_id = db.tenant_id
            db.register_for_signal(DBSignals.TABLE_DROP, self.on_table_drop)
            db.register_for_signal(DBSignals.TABLE_RENAME, self.on_table_rename)

            self.duck = db
            for stage in pipeline:
                if context.has_run:
                    break
                stage(context)
                if self._debug_command_pipeline:
                    print(f"After {stage.__name__}, ran: {context.has_run}, command is: {context.command}")
        if not context.has_run:
            self.run_commands_after_db_closed(context)
        self.clean_df_result(context)
        self.print_df_header(context)
        if isinstance(context.result, pd.DataFrame):
            self._last_result = context.result
        self.recent_tables = context.recent_tables
        return context

    def check_debug_command(self, context: CommandContext):
        if context.command.startswith("?") and not context.command.startswith("??"):
            self._debug_command_pipeline = True
            context.command = context.command[1:]

    def check_print_wide(self, context: CommandContext):
        if context.command.endswith("!"):
            context.print_wide = True
            context.command = context.command[0:-1]

    def lark_parse_command(self, context: CommandContext):
        try:
            context.parse_command(self._get_parser())
        except lark.exceptions.LarkError as e:
            if self._debug_command_pipeline:
                print("Lark parsing failed on command: ", e)
            # Let any parsing exceptions send the command down to the db
            context.lark_parse_error = e

    def skip_non_interactive(self, context: CommandContext):
        if not context.interactive and context.interp_command in self.commands_needing_interaction:
            self.print(f"Skipping command: {context.interp_command}")
            context.mark_ran()

    def rewrite_query_for_db(self, context: CommandContext):
        if context.sqlparse:
            new_query = self.duck.rewrite_query(context.sqlparse)
            if new_query:
                context.command = new_query

    def print_df_header(self, context: CommandContext):
        if isinstance(context.result, pd.DataFrame):
            # print the row count
            self.print("{} row{}".format(context.result.shape[0], "s" if context.result.shape[0] != 1 else ""))

    def run_command_direct_to_db(self, context: CommandContext):
        if context.starts_with_bang():
            context.result = self._execute_duck(context.command[1:])
            context.mark_ran()

    def run_adapter_commands(self, context: CommandContext):
        output: OutputLogger = self.pre_handle_command(context.command)
        if output is not None:
            context.result = output.get_df()
            context.logger = output
            context.mark_ran()

    def run_interp_commands(self, context: CommandContext):
        if context.interp_command:
            method = getattr(self, context.interp_command)
            if 'notebook_path' in inspect.signature(method).parameters:
                context.add_notebook_path_to_args()

            if not context.should_run_after_db_closed():
                context.result = getattr(self, context.interp_command)(
                    **context.parser_visitor._the_command_args
                )
        else:
            # Interp parser failed, but just fall back to the db
            context.result = self._execute_duck(context.command)
            context.mark_ran()

    def run_commands_after_db_closed(self, context: CommandContext):
        if context.should_run_after_db_closed():
            context.result = getattr(self, context.interp_command)(**context.parser_visitor._the_command_args)

    def clean_df_result(self, context):
        if isinstance(context.result, pd.DataFrame):
            if 'count_star()' in context.result.columns:
                context.result.rename(columns={'count_star()': 'count'}, inplace=True)

    def _get_parser(self):
        if self.parser is None:
            path = os.path.join(os.path.dirname(__file__), "grammar.lark")
            self.parser = Lark(open(path).read(), propagate_positions=True)
        return self.parser

    def _get_email_helper(self):
        from .email_helper import EmailHelper

        if self._email_helper is None:
            self._email_helper = EmailHelper()
        return self._email_helper

    def _list_schemas(self, match_prefix=None):
        with dbmgr() as duck:
            return duck.list_schemas()
        
    def _list_tables_filtered(self, schema, table=None):
        try:
            conn = self.loader.lookup_connection(schema)
            table = table or ''
            return sorted(list(t.name[len(table):] for t in conn.list_tables() if t.name.startswith(table)))
        except StopIteration:
            return []

    def _list_schedules(self):
        with dbmgr() as duck:
            with Session(bind=self.duck.engine) as session:
                return session.query(RunSchedule).all()
            
    def _truncate_schedules(self):
        with dbmgr() as duck:
            with Session(bind=self.duck.engine) as session:
                return session.query(RunSchedule).delete()

    def _analyze_columns(self, table_ref: str):
        self.loader.analyze_columns(table_ref)

    def pre_handle_command(self, code):
        m = re.match(r"\s*([\w_0-9]+)\s+(.*$)", code)
        if m:
            first_word = m.group(1)
            rest_of_command = m.group(2)
            if first_word in self.adapters:
                logger: OutputLogger = OutputLogger()
                handler: Adapter = self.adapters[first_word]
                return handler.run_command(rest_of_command, logger)

    def replace_last_table_reference(self, context: CommandContext):
        pos = context.command.find(" ??")
        if pos > 0:
            if len(self.recent_tables) > 0:
                context.command = context.command[0:pos] + " " + str(self.recent_tables[0]) + " " + context.command[pos + 3:]
                self.print(context.command)
                context.set_recent_tables(self.recent_tables)
            else:
                raise RuntimeError("Can't find previous table to replace '??'")

    def substitute_variables(self, context: CommandContext):
        if re.match(r"\s*\$[\w_0-9]+\s*$", context.command):
            return context.command # simple request to show the variable value

        def lookup_var(match):
            var_name = match.group(1)
            value = self._get_variable(var_name)
            if isinstance(value, pd.DataFrame):
                ref_var =f"{var_name}__actualized"
                self.duck.create_memory_table(ref_var, value)
                #self.duck.register(ref_var, value)
                return ref_var
            elif value is not None:
                # literal substitution
                if isinstance(value, str):
                    return f"'{value}'"
                else:
                    return str(self.session_vars[var_name])
            else:
                return "$" + var_name

        match = re.match(r"\s*(\$[\w_0-9]+)\s*=(.*)", context.command, re.DOTALL)
        if match:
            # var assignment, only interpolate the right hand side
            rhs = CommandContext(match.group(2))
            self.substitute_variables(rhs)
            context.command = match.group(1) + "=" + rhs.command
        else:
            # interpolate the whole command
            context.command = re.sub(r"\$([\w_0-9]+)", lookup_var, context.command)

    def _execute_duck(self, query: typing.Union[str, CommandContext]) -> pd.DataFrame:
        if isinstance(query, CommandContext):
            query = query.command
        return self.duck.execute(query, context=self.context)

    def get_prompt_completer(self):
        schema_lists = {}
        with dbmgr() as db:
            result = {}
            for m in self.get_help_messages():
                m = m.replace("[", "").replace("]", "")
                match = re.match(r"([\w<> ]+)\s*[\-\[']", m)
                if match:
                    command_words = match.group(1).split()
                    wc = len(command_words)
                    if wc == 1:
                        if command_words[0] not in result:
                            result[command_words[0]] = {}
                    elif wc > 1:
                        cur_dict = result
                        for idx, word in enumerate(command_words):
                            if word == "<schema>":
                                if word not in schema_lists:
                                    schema_lists[word] = db.list_schemas()['schema_name']
                                for schema in schema_lists[word]:
                                    cur_dict[schema]=None 
                            if word == "<table>":
                                if word not in schema_lists:
                                    schema_lists[word] = db.list_tables().apply(
                                        lambda x: f"{x.table_schema}.{x.table_name}", axis=1
                                    )
                                for table in schema_lists[word]:
                                    cur_dict[table]=None 
                            if idx < (wc - 1):
                                if word not in cur_dict:
                                    cur_dict[word] = {}
                                cur_dict = cur_dict[word]
                            else:
                                try:
                                    if word not in cur_dict:
                                        cur_dict[word] = {}
                                except:
                                    pass
        #pprint(result)
        result["select"] = TableListCompleter(schema_lists["<table>"])
        result["move"] = {"left.first": None, 
        "right": None, "up": None, "down": None}
        return DottedWordCompleter.from_nested_dict(result)


    def print(self, *args):
        self.context.logger.print(*args)

    ################
    ## Commands 
    #
    # All commands either "print" to the result buffer, or else they return
    # a DataFrame result (or both). It the the responsibilty of the host
    # program to render the result. Commands should call `context.get_input` to
    # retrieve input from the user interactively.
    ################
    def alter_table(self, table_ref, new_table):
        self.duck.rename_table(TableHandle(table_ref), new_table)

    def show_connections(self):
        """ show connections - list all connections. Use 'connect' to create a new connection. """
        for c in self.loader.connections:
            self.print(f"Adapter {c.adapter.name} ({c.adapter.base_api_url}) - schema: {c.schema_name}")

    def connect_command(self):
        """ connect - connect to a new system. """
        specs = sorted(Connection.ADAPTER_SPECS.keys())
        specs_print = "\n".join(f"{idx+1}: {name}" for idx, name in enumerate(specs))
        spec_number = self.context.get_input(specs_print + "\nPick an adapter: ")
        if spec_number == "":
            return
        adapter_tuple = Connection.ADAPTER_SPECS[specs[int(spec_number)-1]]
        adapter = adapter_tuple[0](adapter_tuple[1], None, "new_schema")
        
        print(f"Ok! Let's setup a new {adapter.name} connection.")
        while True:
            schema = self.context.get_input(f"Specify the schema name ({adapter.name}): ")
            if schema == "":
                schema = adapter.name

            if schema in self.adapters:
                print("Schema name must be unique")
            else:
                break

        while True:
            print("Please provide the configuration parameters:")
            config_opts = adapter.get_config_parameters()
            config_dict = {}
            for opt, desc in config_opts.items():
                config_dict[opt] = self.context.get_input(f"{opt} ({desc}): ")

            try:
                print("Testing connection...")
                self.loader.add_connection(adapter.name, schema, config_dict)
                self.adapters: dict[str, Adapter] = self.loader.adapters
                print(f"New {adapter.name} connection created in schema {schema}")
                print("The following tables are available, use peek or select to load data:")
                return self.show_tables(schema_ref=schema)
            except RuntimeError as e:
                print("Connection failed: " + str(e))


    def count_table(self, table_ref):
        """ count <table> - returns count of rows in a table """
        return self._execute_duck(f"select count(*) as count from {table_ref}")

    def get_help_messages(self) -> Generator[str, None, None]:
        for f in sorted(inspect.getmembers(self.__class__, inspect.isfunction)):
            if f[0] in ['help','__init__']:
                continue
            doc = inspect.getdoc(f[1])
            if doc:
                yield doc

    def help(self, help_choice):
        """ help - show this message 
        help schemas - overview of schemas
        help charts - help on generating charts
        help import - help on importing data
        help export - help on exporting data
        """
        if help_choice is None:
            for l in inspect.getdoc(self.help).splitlines():
                self.print(l)
            for m in self.get_help_messages():
                self.print(m)
            return
        helps = {
            "schemas": """Every connected system is represented in the Unify database as a schema.
            The resources for the system appear as tables within this schema. Initially all tables
            are empty, but they are imported on demand from the connected system whenver you access
            the table.

            Some systems support custom commands, which you can invoke by using the schema name
            as the command prefix. You can get help on the connected system and its commands by
            typing "help <schema>".
            """,
            "charts": """Help for charts"""
        }
        msg = helps[help_choice]
        for l in msg.splitlines():
            self.print(l.strip())

    def help_last(self):
        """ ?? - shows column info for tables referenced in the most recent query """
        for table in (self.recent_tables or []):
            self.print(f"Table {table}:")
            self.print(self.duck.list_columns(table).to_string(index=False))
            self.print("\n")

    def import_command(self, file_path: str, options: list=[]):
        """ import URL | file path - imports a file or spreadsheet as a new table """
        # See if any of our adapters want to import the indicated file
        # Use FileAdapter as last resort
        adapters = sorted(list(self.adapters.items()), key=lambda x: x[1].name == 'files' and 1 or 0)

        for schema, adapter in adapters:
            if adapter.can_import_file(file_path):
                adapter.logger = self.context.logger
                table_root = adapter.import_file(file_path, options=options) # might want to run this in the background
                table = schema + "." + table_root
                context = self.run_command(f"select * from {table} limit 10")
                self.print(f"Imported file to table: {table}")
                return context.result
        self.print("File not found")
             
    def drop_table(self, table_ref):
        """ drop table <table> - removes the table from the database """
        if self.context.interactive:
            val = self.context.get_input(f"Are you sure you want to drop the table '{table_ref}' (y/n)? ")
        else:
            val = 'y'
        if val == "y":
            return self.duck.drop_table(TableHandle(table_ref))
            # note that Adpater will be signaled in the on_table_drop callback

    def drop_schema(self, schema_ref, cascade: bool):
        """ drop schema <schema> [cascade] - removes the entire schema from the database """
        val = self.context.get_input(f"Are you sure you want to drop the schema '{schema_ref}' (y/n)? ")
        if val == "y":
            return self.duck.drop_schema(schema_ref, cascade)

    def email_command(self, email_object, recipients, subject=None, notebook_path: str=None):
        """ email [notebook|<table>|chart <chart>| to '<recipients>' [subject 'msg subject'] - email a chart or notebook to the recipients """

        recipients = re.split(r"\s*,\s*", recipients)
        if email_object == "notebook":
            # Need to render and email the current notebook
            if notebook_path:
                notebook = os.path.basename(notebook_path)
                self.print(f"Emailing {notebook} to {recipients}")
                if subject is None:
                    subject = f"{notebook} notebook - Unify"
                self._get_email_helper().send_notebook(notebook_path, recipients, subject)
            else:
                self.print("Error, could not determine notebook name")
        elif email_object == "chart":
            if self.last_chart:                
                self.print(f"Emailing last chart to {recipients}")
                self._get_email_helper().send_chart(self.last_chart, recipients, subject)
            else:
                self.print("Error no recent chart available")
        else:
            if subject is None:
                subject = f"{email_object} contents - Unify"
            # Email the contents of a table
            df: pd.DataFrame = self._execute_duck(f"select * from {email_object}")
            if subject is None:
                subject = f"Unify - {email_object}"
            self.print(f"Emailing {email_object} to {recipients}")
            fname = email_object.replace(".", "_") + ".csv"
            self._get_email_helper().send_table(df, fname, recipients, subject)

    def export_table(self, adapter_ref, table_ref, file_ref, write_option=None):
        """ export <table> <adapter> <file> [append|overwrite] - export a table to a file """
        if file_ref.startswith("(") and file_ref.endswith(")"):
            # Evaluate an expression for the file name
            result = self.duck.execute(f"select {file_ref}").fetchone()[0]
            file_ref = result

        if adapter_ref:
            if adapter_ref in self.adapters:
                adapter = self.adapters[adapter_ref]
            else:
                raise RuntimeError(f"Uknown adapter '{adapter_ref}'")
        else:
            # assume a file export
            adapter = self.adapters['files']

        exporter = TableExporter(
            table=table_ref, 
            adapter=adapter, 
            target_file=file_ref,
            allow_overwrite=(write_option == "overwrite"),
            allow_append=(write_option == "append")
        )
        exporter.run(self.context.logger)
        self.print(f"Exported query result to '{file_ref}'")

    def on_table_drop(self, dbmgr, table):
        # remove all column info
        with Session(bind=dbmgr.engine) as session:
            cols = session.query(ColumnInfo).filter(
                ColumnInfo.table_name == table.table_root(),
                ColumnInfo.table_schema == table.schema()
            ).delete()
        if table.schema() in self.adapters:
            self.adapters[table.schema()].drop_table(table.table_root())

    def on_table_rename(self, dbmgr, old_table, new_table):
        if old_table.schema() in self.adapters:
            self.adapters[old_table.schema()].rename_table(old_table.table_root(), new_table.table_root())

    def open_command(self, open_target: str):
        if open_target == "metabase":
            return self.open_metabase()
        elif open_target == "tutorial":
            docsdir = os.path.join(os.path.dirname(__file__), '../docs')
            Handler = functools.partial(MDRequestHandler, directory=docsdir)
            setattr(MDRequestHandler, 'log_message', lambda *args: None)
            http = HTTPServer(('127.0.0.1', 9010), Handler)
            self.print("Docs server listening on port 9010...")
            t = threading.Thread(target=http.serve_forever, daemon=True)
            t.start()
            webbrowser.open("http://localhost:9010/TUTORIAL.md")


    def open_metabase(self):
        """ open metabase - install and open Metabase to analyze your data """
        ms: MetabaseSetup = MetabaseSetup(os.path.expanduser("~/unify"), self.context.get_input)
        if ms.mb_is_running():
            return ms.open_metabase()
        if ms.mb_is_installed():
            ms.launch_metabase()
            return ms.open_metabase()
            
        choice = self.context.get_input("Do you want to install and run Metabase (y/n)? ")
        if choice != 'y':
            return
        ms.ensure_java_installed()
        ms.ensure_metabase_installed()
        print("Please enter info to create your local Metabase account:")
        email = self.context.get_input("Enter your email address: ")
        while True:
            password = self.context.get_input("Choose a Metabase login password: ")
            password2 = self.context.get_input("Confirm the password: ")
            if password == password2:
                break
        if ms.launch_metabase():
            ms.setup_metabase(
                ch_host="localhost", 
                ch_database=self.duck.tenant_db, 
                ch_user="default", 
                ch_password = None,
                mb_password=password,
                mb_email=email)
            ms.open_metabase()

    def peek_table(self, qualifier, peek_object, line_count=20, build_stats=True, debug=False):
        # Get column weights and widths.
        if qualifier == 'file':
            # Peek at file instead of table
            for schema, adapter in self.adapters.items():
                file_path = peek_object.strip("'")
                if adapter.can_import_file(file_path):
                    return adapter.peek_file(file_path, line_count, self.context.logger)
            return

        # Need to re-implement this without relying on our "pre-analyzed" column list which
        # wasn't worth the effort precalculate it.

    def refresh_table(self, table_ref):
        """ refresh table <table> - updates the rows in a table from the source adapter """
        self.loader.refresh_table(table_ref)

    def reload_table(self, table_ref):
        """ reload table <table> - reloads the entire table from the source adapter """
        self._execute_duck(f"drop table if exists {table_ref}")
        schema, table_root = table_ref.split(".")
        self.load_adapter_data(schema, table_root)

    def run_info(self, schedule_id):
        """ run info <notebook> - Shows details on the schedule for the indicated notebook """
        with Session(bind=self.duck.engine) as session:
            schedule = session.query(RunSchedule).filter(RunSchedule.id == schedule_id).first()
            if schedule:
                self.print("Schedule: ", schedule.run_at, " repeat: ", schedule.repeater)
                contents = schedule['contents']
                body = json.loads(contents)
                for cell in body['cells']:
                    if 'source' in cell:
                        if isinstance(cell['source'], list):
                            for line in cell['source']:
                                self.print("| ", line)
                        else:
                            self.print("| ", cell['source'])
            else:
                self.print("Schedule not found")

    def run_notebook_command(self, run_at_time: str, notebook_path: str, repeater: str=None):
        """ run [every day|week|month] at <date> <time> - Execute this notebook on a regular schedule """
        if notebook_path is None:
            self.print("Error, must supply a notebook name or full path")
            return
        contents = None
        if not os.path.exists(notebook_path):
            # Try to find the notebook in the Unify notebooks directory
            notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", notebook_path)

        if os.path.exists(notebook_path):
            # Jankily jam the whole notebook into the db so we can run it on the server
            contents = open(notebook_path, "r").read()
        else:
            raise RuntimeError(f"Cannot find notebook '{notebook_path}'")

        run_at_time = pd.to_datetime(run_at_time) # will assign the current date if no date
        run_id = os.path.basename(notebook_path)
        with Session(bind=self.duck.engine) as session:
            session.query(RunSchedule).filter(RunSchedule.id == run_id).delete()
            session.commit()
            session.add(RunSchedule(
                id = run_id,
                notebook_path = notebook_path,
                run_at = run_at_time,
                repeater = repeater,
                contents = contents
            ))
            session.commit()

        self.print(f"Scheduled to run notebook {notebook_path}")

    def run_schedule(self):
        """ run schedule - displays the current list of scheduled tasks """
        with Session(bind=self.duck.engine) as session:
            return pd.read_sql_query(
                sql = select(RunSchedule),
                con = self.duck.engine
            )

    def delete_connection(self):
        """ delete connection - delete an existing connection and all it's data """
        clist = []
        for idx, c in enumerate(self.loader.connections):
            clist.append(f"{idx+1}: Adapter {c.adapter.name} ({c.adapter.base_api_url}) - schema: {c.schema_name}")
        number = self.context.get_input("\n".join(clist) + "\nChoose which connection to delete: ")
        if number:
            conn = self.loader.connections[int(number)-1]
            print("!! WARNING: THIS WILL DELETE ALL TABLES SYNCHRONIZED FROM THIS CONNECTION !!")
            match = self.context.get_input(f"Type the name of the schema ({conn.schema_name}) to confirm: ")
            if match == conn.schema_name:
                print(f"Removing connection '{conn.schema_name}'")
                self.loader.delete_connection(conn)

    def delete_schedule(self, schedule_id):
        """ run delete <notebook> - Delete the schedule for the indicated notebook """
        with Session(bind=self.duck.engine) as session:
            sched = session.query(RunSchedule).filter(RunSchedule.id == schedule_id).first()
            if sched:
                self.print(f"Deleted schedule {schedule_id} for notebook: ", sched.notebook_path)

    def set_variable(self, var_ref: str, var_expression: str):
        """ $<var> = <expr> - sets a variable. Use all caps for var to set a global variable. """
        is_global = (var_ref.upper() == var_ref)
        if not var_expression.lower().startswith("select "):
            # Need to evaluate the scalar expression
            val = self.duck.execute("select " + var_expression).iloc[0][0]
            self._save_variable(var_ref, val, is_global)
            self.print(val)
        else:
            val = self.duck.execute(var_expression)
            if not val.empty and val.shape == (1, 1):
                val = val.iloc[0][0]
            self._save_variable(var_ref, val, is_global)
            return val

    def _get_variable(self, name: str):
        if name.upper() == name:
            with Session(bind=self.duck.engine) as session:
                savedvar = session.query(SavedVar).filter(SavedVar.name==name).first()
                if savedvar:
                    return savedvar.value
                else:
                    # Maybe it was stored as full table
                    table_name = "var_" + LocalFileAdapter.convert_string_to_table_name(name)
                    return self.duck.execute(f"select * from meta.{table_name}")
        else:
            return self.session_vars[name]

    def _save_variable(self, name: str, value, is_global: bool):
        if is_global:
            if isinstance(value, pd.DataFrame):
                table_name = "var_" + LocalFileAdapter.convert_string_to_table_name(name)
                self.duck.write_dataframe_as_table(value, TableHandle(table_name, "meta"))
            with Session(bind=self.duck.engine) as session:
                session.query(SavedVar).filter(SavedVar.name==name).delete()
                session.commit()
                session.add(SavedVar(name=name, value=value))
                session.commit()
        else:
            self.session_vars[name] = value

    def show_schemas(self):
        """ show schemas - list schemas in the datbase """
        return self.duck.list_schemas()

    def show_tables(self, schema_ref=None):
        """ show tables [from <schema> [like '<expr>']] - list all tables or those from a schema """
        if schema_ref:
            records = []
            if schema_ref in self.adapters:
                for tableDef in self.adapters[schema_ref].list_tables():
                    records.append({
                        "table_name": tableDef.name,
                        "table_schema": schema_ref,
                        "comment": tableDef.description
                    })

            df: pd.DataFrame = pd.DataFrame(records)
            actuals: pd.DataFrame = self.duck.list_tables(schema_ref)
            actual_names = []
            if not actuals.empty:
                df = pd.concat([df, actuals]).drop_duplicates('table_name',keep='last').reset_index(drop=True)
                actual_names = actuals['table_name'].tolist()
            if not df.empty:
                df.sort_values('table_name', ascending = True, inplace=True)
                df['materialized'] = ['✓' if t in actual_names else '☐' for t in df['table_name']]
                return df
            else:
                self.print("No tables")
                return None
        else:
            self.print("{:20s} {}".format("schema", "table"))
            self.print("{:20s} {}".format("---------", "----------"))
            return self.duck.list_tables(schema_ref)

    def show_columns(self, table_ref, column_filter=None):
        """ show columns [from <table> [like '<expr>']] - list all columns or those from a table """
        return self.duck.list_columns(TableHandle(table_ref), column_filter)

    def describe(self, table_ref):
        """ describe [<schema>|<table>] - list all tables, tables in a schema, or columns from a table """
        if table_ref is None:
            return self.show_schemas()
        elif table_ref is not None and "." in table_ref:
            return self.show_columns(table_ref)
        else:
            return self.show_tables(table_ref)

    def create_statement(self):
        """ create table 'table' ... - create a new table """
        return self._execute_duck(self.context.command)

    def create_view_statement(self):
        """ create view 'view' ... - create a new view """
        return self._execute_duck(self.context.command)

    def insert_statement(self):
        """ insert into <table> ... - insert records into a table """
        return self._execute_duck(self.context.command)

    def delete_statement(self):
        """ delete from <table> [where ...] """
        return self._execute_duck(self.context.command)

    def load_adapter_data(self, schema_name, table_name):
        loaded = self.loader.run_job(
            LoaderJob(
                self._tenant_id, 
                LoaderJob.ACTION_LOAD_TABLE, 
                TableHandle(table_name, schema_name)
            )
        )
        return loaded

    def select_query(self, fail_if_missing=False, **kwargs):
        """ select <columns> from <table> [where ...] [order by ...] [limit ...] [offset ...] """
        try:
            return self._execute_duck(self.context.command)

        except TableMissingException as e:
            if fail_if_missing:
                self.print(e)
                return
            schema, table_root = e.table.split(".")
            print("Loading table...")
            if self.load_adapter_data(schema, table_root):
                return self.select_query(fail_if_missing=True)
            else:
                for x in range(10):
                    try:
                        return self._execute_duck(self.context.command)
                    except TableMissingException:
                        time.sleep(1.0)

    def select_for_writing(self, select_query, adapter_ref, file_ref):
        if adapter_ref in self.adapters:
            adapter = self.adapter[adapter_ref]
            exporter = TableExporter(select_query, adapter, file_ref)
            exporter.run()
            self.print(f"Exported query result to '{file_ref}'")
        else:
            self.print(f"Error, uknown adapter '{adapter_ref}'")

    def show_variable(self, var_ref):
        value = self._get_variable(var_ref)
        if isinstance(value, pd.DataFrame):
            return value
        else:
            self.print(value)

    def show_variables(self):
        """ show variables - list all defined variables"""
        rows = [(k, "[query result]" if isinstance(v, pd.DataFrame) else v) for k, v in self.session_vars.items()]
        with Session(bind=self.duck.engine) as session:
            rows.extend([(k.name, k.value) for k in session.query(SavedVar)])
        return pd.DataFrame(rows, columns=["variable", "value"])

    def clear_table(self, table_schema_ref=None):
        """ clear <table> - removes all rows from a table """
        self.loader.truncate_table(table_schema_ref)
        self.print("Table cleared: ", table_schema_ref)

    def old_create_chart_with_matplot(
        self, 
        chart_name=None, 
        chart_type=None, 
        chart_source=None, 
        chart_where=None,
        chart_params={}):
        """ create chart from <$var or table> as <chart_type> where x = <col> and y = <col> [opts] - see 'help charts' for info """
        # FIXME: observe chart_source
        if "x" not in chart_params:
            raise RuntimeError("Missing 'x' column parameter for chart X axis")
        df = self._last_result
        if df is None:
            raise RuntimeError("No query result available")
        plt.rcParams["figure.figsize"]=10,8
        plt.rcParams['figure.dpi'] = 100 
        if chart_type == "pie_chart":
            df = df.set_index(chart_params["x"])
        kind = ParserVisitor.MATPLOT_CHART_MAP[chart_type]
        title = chart_params.get("title", "")

        fig, ax = plt.subplots()

        df.plot(x = chart_params["x"], y = chart_params.get("y"), kind=kind,
                title=title, stacked=chart_params.get("stacked", False))
        plt.tight_layout()
        
        imgdata = io.BytesIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        return {"mime_type": "image/png", "data": imgdata.getvalue()}

    def create_chart(
        self, 
        chart_name=None, 
        chart_type=None, 
        chart_source=None, 
        chart_params: dict={}):
        import altair  # defer this so we dont pay this cost at startup

        # Note of these renderers worked for both Jupyterlab and email
        # --> mimetype, notebook, html

        altair.renderers.enable('png')

        if len(chart_params.keys()) == 0:
            source = pd.DataFrame({"category": [1, 2, 3, 4, 5, 6], "value": [4, 6, 10, 3, 7, 8]})

            print(source)
            chart = altair.Chart(source).mark_arc().encode(
                theta="value", #altair.Theta(field="value", type="quantitative"),
                color="category" #altair.Color(field="category", type="nominal"),
            )
            return chart

            self._last_result = pd.DataFrame({
                'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
                'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
            })
            chart_type = "bar_chart"
            chart_params["x"] = 'a'
            chart_params["y"] = 'b'           

        if chart_source:
            if chart_source.startswith('$'):
                df = self._get_variable(chart_source[1:])
            df = self._execute_duck(f"select * from {chart_source}")
        else:
            df = self._last_result
        print(df)

        if df is None or df.shape[0] == 0:
            raise RuntimeError("No recent query, or query returned no rows")

        title = chart_params.pop('title', '')

        trendline = chart_params.pop('trendline', None)

        chart_methods = {
            'bar_chart': 'mark_bar',
            'pie_chart': 'mark_arc',
            'hbar_chart': 'mark_bar',
            'line_chart': 'mark_line',
            'area_chart': 'mark_area'
        }
        # To make a horizontal bar chart we have to annotate the X axis value as the
        # "quantitative" value like: x="total:Q" rather than the "ordinal" value as "date:O".
        if chart_type not in chart_methods:
            raise RuntimeError(f"Uknown chart type '{chart_type}'")

        #chart_params["tooltip"] = {"content":"data"}
        rolling = chart_params.pop('rolling', None)

        print(chart_params)
        chart = altair.Chart(df)
        chart = getattr(chart, chart_methods[chart_type])(tooltip=True). \
            encode(**chart_params). \
            properties(title=title, height=400, width=600)

        if trendline and 'y' in chart_params:
            if trendline == 'average':
                trendline="average({}):Q".format(chart_params['y'])
            elif trendline == 'mean':                
                trendline="mean({}):Q".format(chart_params['y'])
            else:
                val = float(trendline)
                df['_trend'] = val
                trendline = "_trend"
            trend = altair.Chart(df).mark_line(color='red').encode(x=chart_params['x'], y=trendline)
            chart = chart + trend

        if rolling:
            try:
                window = int(rolling)
            except ValueError:
                raise RuntimeError("rolling parameter must be the window size as an integer")
            
            rollavg = altair.Chart(df).mark_line(color="#FFAA00").transform_window(
                rolling_mean='mean(' + chart_params['y'] + ')',
                frame=[-window, window]
            ).encode(
                **(chart_params | {'y': 'rolling_mean:Q'})
            )
            chart = chart + rollavg

        self.last_chart = chart
        return chart

    def search_command(self, query):
        """ search <query> - search for schemas, tables, or columns matching your query """
        # TODO: implement
        pass

    def system_command(self, command: str):
        if command == "status":
            if self.loader.task_daemon_is_running():
                self.print("Task daemon is running")
            else:
                self.print("Task daemon is not running")
        elif command == "stop daemon":
            self.print("Stopping task daemon...")
            self.loader.stop_daemon();
        elif command == "restart daemon":
            self.print("Restarting task daemon...")
            self.loader.stop_daemon()
            time.sleep(1)
            self.loader.start_daemon()
        elif command == "cancel":
            self.print("Cancelling background jobs...")
            self.loader.cancel_all_jobs()


    #########
    ### FILE system commands
    #########
    def show_files(self, schema_ref=None, match_expr=None):
        """ 
            show files [from <connection>] [like '<pattern>'] - Lists files on the file system or from the indicated connection 
        """
        if schema_ref is None:
            schema_ref = 'files'

        if schema_ref not in self.adapters:
            raise RuntimeError(f"Uknown schema '{schema_ref}'")

        self.adapters[schema_ref].logger = self.context.logger
        for file in self.adapters[schema_ref].list_files(match_expr):
            self.print(file)

def setup_job_log_handler(handler):
    add_logging_handler(handler)