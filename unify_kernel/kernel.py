import io

from ipykernel.kernelbase import Kernel
from unify import RunCommand

class UnifyKernel(Kernel):
    implementation = 'Unify'
    implementation_version = '1.0'
    language = 'SQL'
    language_version = '0.1'
    language_info = {
        'name': 'Any text',
        'mimetype': 'text/plain',
        'file_extension': '.txt',
    }
    banner = "Unify kernel - universal cloud data access"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unify_runner = RunCommand(wide_display=True)

    def do_execute(self, code, silent, store_history=True, user_expressions=None,
                   allow_stdin=False):

        buffer = io.StringIO()
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
