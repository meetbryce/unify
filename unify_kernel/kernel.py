import io
import traceback

from ipykernel.kernelbase import Kernel
from unify import RunCommand
from lark.visitors import Visitor
from parsing_utils import find_node_return_children


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
        self.unify_runner = RunCommand(wide_display=True, read_only=True)
        self.autocomplete_parser = AutocompleteParser(self.unify_runner.parser)

    def _send_string(self, msg):
        stream_content = {'name': 'stdout', 'text': msg}
        self.send_response(self.iopub_socket, 'stream', stream_content)

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):

        buffer = io.StringIO()
        try:
            response = self.unify_runner._run_command(code, output_buffer=buffer, use_pager=False)
            if not silent:
                if response["response_type"] == "stream":
                    buffer.seek(0)
                    stream_content = {'name': 'stdout', 'text': buffer.read()}
                    self.send_response(self.iopub_socket, 'stream', stream_content)
                elif response["response_type"] == "display_data":
                    content = {
                        'source': 'kernel',
                        'data': { 'image/png': buffer.getvalue() },
                        # We can specify the image size
                        # in the metadata field.
                        'metadata' : {}
                    }
                    # We send the display_data message with
                    # the contents.
                    self.send_response(self.iopub_socket, 'display_data', content)

            return {'status': 'ok',
                    # The base class increments the execution count
                    'execution_count': self.execution_count,
                    'payload': [],
                    'user_expressions': {},
                }
        except Exception as e:
            stream_content = {'name': 'stderr', 'text': str(e)}
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