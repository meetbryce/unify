import io
import sys
import traceback

from ipykernel.kernelbase import Kernel
from unify import RunCommand
from lark.visitors import Visitor

class AutocompleteParser(Visitor):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser

    def parse_command(self, command):
        self.visited = []
        parse_tree = self.parser.parse(command)
        self.visit(parse_tree)

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
        self.unify_runner = RunCommand(wide_display=True)
        self.autocomplete_parser = AutocompleteParser(self.unify_runner.parser)

    def _send_string(self, msg):
        stream_content = {'name': 'stdout', 'text': msg}
        self.send_response(self.iopub_socket, 'stream', stream_content)

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):

        buffer = io.StringIO()
        try:
            self.unify_runner._run_command(code, output_buffer=buffer, use_pager=False)
            if not silent:
                buffer.seek(0)
                stream_content = {'name': 'stdout', 'text': buffer.read()}
                self.send_response(self.iopub_socket, 'stream', stream_content)

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
            self.autocomplete_parser.parse_command(code)
            if self.autocomplete_parser.visited == ['show_tables']:
                # Suggest a schema to show from
                matches = self.unify_runner._list_schemas()

        return {
            'status': 'ok',
            'matches': matches,
            'cursor_start': cursor_pos,
            'cursor_end': cursor_pos + 5,
            'metadata': {}
        }