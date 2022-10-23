from distutils.cmd import Command
import io
import traceback
import pandas as pd
import urllib
import base64
import time
import logging

from ipykernel.kernelbase import Kernel
from unify import CommandInterpreter, CommandContext
from lark.visitors import Visitor
from unify.parsing_utils import find_node_return_children
import ipynbname

from unify.db_wrapper import TableMissingException, QuerySyntaxException

logger = logging.getLogger(__name__)

class AutocompleteParser(Visitor):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser

    def unify_parse_command(self, command):
        self.visited = []
        self.parts_found = {}
        parse_tree = self.parser.parse(command)
        self._remember_command = command
        self.visit(parse_tree)

    def show_tables(self, tree):
        self.visited.append('show_tables')
        schema_ref = find_node_return_children('schema_ref', tree)
        if schema_ref:
            self.parts_found['schema_ref'] = schema_ref[0] 

    def show_schemas(self, tree):
        self.visited.append('show_schemas')

    def show_columns(self, tree):
        self.visited.append('show_columns')
        table_ref = find_node_return_children('table_ref', tree)
        if table_ref:
            self.parts_found['schema_ref'] = table_ref[0] 
        else:
            tschema_ref = find_node_return_children('table_schema_ref', tree)
            if tschema_ref:
                self.parts_found['table_schema_ref'] = tschema_ref

    def describe(self, tree):
        self.visited.append('describe')

    def select_query(self, tree):
        self.visited.append('select_query')

class UnifyKernel(Kernel):
    implementation = 'Unify'
    implementation_version = '1.0'
    language = 'SQL'
    language_version = '0.1'
    language_info = {
        'name': 'text',
        'mimetype': 'text/plain',
        'file_extension': '.txt',
    }
    banner = "Unify kernel - universal cloud data access"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unify_runner = CommandInterpreter()
        self.autocomplete_parser = AutocompleteParser(self.unify_runner._get_parser())

    def _send_string(self, msg):
        stream_content = {'name': 'stdout', 'text': msg}
        self.send_response(self.iopub_socket, 'stream', stream_content)

    def _find_notebook(self):
        try:
            return str(ipynbname.path())
        except Exception as e:
            logger.error("Error finding notebook name: %s", e)
            return None

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):
        self._allow_stdin = allow_stdin
        try:
            context: CommandContext = self.unify_runner.run_command(
                code, 
                input_func=self.raw_input, 
                get_notebook_func=self._find_notebook,
                interactive=(not silent)
            )
            lines = context.lines
            object = context.result
            if not silent:
                if lines:
                    self.send_response(self.iopub_socket, 'stream', {'name': 'stdout', 'text': "\n".join(lines)})
                if isinstance(object, pd.DataFrame):
                    with pd.option_context('display.float_format', '{:0.2f}'.format):
                        content = {
                            'source': 'kernel',
                            'data': { 
                                'text/html': object.to_html(index=False, render_links=True, escape=False),
                                'text/plain': object.to_string(index=False)
                            },
                            'metadata' : {}
                        }
                        self.send_response(self.iopub_socket, 'display_data', content)
                elif isinstance(object, dict) and 'mime_type' in object:
                    print("Got a dict with mime_type: ", object['mime_type'], " returning image")
                    enc_data = urllib.parse.quote(base64.b64encode(object['data']))
                    content = {
                        'source': 'kernel',
                        'data': { object['mime_type']:  enc_data},
                        'metadata' : {}
                    }
                    print(content)
                    self.send_response(self.iopub_socket, 'display_data', content)
                elif object is not None:
                    # iPython has some crazy rules to use _repr_XX methods to render the object. Implement some of
                    # those here
                    if hasattr(object, '_repr_mimebundle_'):
                        data = object._repr_mimebundle_()
                        if isinstance(data, (tuple,list)):
                            data = data[0]
                        logger.critical("Chart data keys are: {}".format(str(data.keys())))
                        content = {
                            'source': 'kernel',
                            'data': data,
                            'metadata' : {}
                        }
                        self.send_response(self.iopub_socket, 'display_data', content)
                    else:
                        self._send_string("Unrecognized internal result object: " + str(object))

            return {'status': 'ok',
                    # The base class increments the execution count
                    'execution_count': self.execution_count,
                    'payload': [],
                    'user_expressions': {},
                }
        except Exception as e:
            if isinstance(e, (TableMissingException, QuerySyntaxException)):
                self._send_string(str(e))
                return {'status': 'ok',
                        # The base class increments the execution count
                        'execution_count': self.execution_count,
                        'payload': [],
                        'user_expressions': {},
                    }
            else:
                self.log.exception("Error runnning kernel command")
                stream_content = {'name': 'stderr', 'text': (str(e) + "\n" + traceback.format_exc())}
                self.send_response(self.iopub_socket, 'stream', stream_content)
                return {'status': 'error',
                        'execution_count': self.execution_count,
                        'ename': type(e).__name__,
                        'evalue': str(e),
                        'traceback': traceback.format_exc().split("\n")
                        }

    def do_complete(self, code, cursor_pos):
        matches = []
        if code.strip() == 'show':
            matches = ["tables", "schemas", "columns from "]
        else:
            self.autocomplete_parser.unify_parse_command(code)
            visit = self.autocomplete_parser.visited
            if visit == ['show_tables']:
                # Suggest a schema to show from
                schema_ref = self.autocomplete_parser.parts_found.get('schema_ref')
                matches = self.unify_runner._list_schemas(schema_ref)
            elif visit == ['show_columns']:
                # "show columns from <schema>.<table>" - complete schema name or table name
                schema_ref = self.autocomplete_parser.parts_found.get('schema_ref')
                table_schema_ref = self.autocomplete_parser.parts_found.get('table_schema_ref')
                if schema_ref:
                    matches = self.unify_runner._list_schemas(schema_ref)
                elif table_schema_ref:
                    if len(table_schema_ref) > 1:
                        schema, table = table_schema_ref
                    else:
                        schema, table = (table_schema_ref[0], None)
                    matches = self.unify_runner._list_tables_filtered(schema, table)
                else:
                    matches = self.unify_runner._list_schemas()

        return {
            'status': 'ok',
            'matches': matches,
            'cursor_start': cursor_pos,
            'cursor_end': cursor_pos + 5,
            'metadata': {}
        }