import logging
import pydoc
import os
import sys
import pandas as pd
import traceback
import time
import threading

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.patch_stdout import patch_stdout

from .interpreter import CommandInterpreter, CommandContext, setup_job_log_handler

logger: logging.Logger = logging.getLogger('unify')
logger.setLevel(logging.DEBUG)
global job_record
job_record = None

class LoaderJobHandler(logging.Handler):  
    def __init__(self):
        super().__init__()

    def emit(self, record):
        global job_record
        #if '_job_id' in record.__dict__ and '_table_root' in record.__dict__:
        job_record = record
        #print("!!JOB LOG:", record)
        print("", end="", flush=True)

    @staticmethod
    def toolbar():
        global job_record
        if job_record:
            if hasattr(job_record, '_table_root'):
                return f"[Loading {job_record._schema}:{job_record._table_root}] {job_record.msg} "
            else:
                return job_record.msg % job_record.args
        else:
            return "Not loading"

setup_job_log_handler(LoaderJobHandler())

completer = NestedCompleter.from_nested_dict({
    'show': {
        'schemas': None,
        'connections': None,
        'tables': {
            'from': None
        }
    },
    'exit': None,
})

class UnifyRepl:
    def __init__(self, interpreter: CommandInterpreter, wide_display=False):
        self.interpreter = interpreter
        if wide_display:
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)

    def loop(self):
        session = PromptSession(history=FileHistory(os.path.expanduser("~/.pphistory")))
        suggester = AutoSuggestFromHistory()
        try:
            with patch_stdout():
                while True:
                    try:
                        cmd = session.prompt("> ", 
                            auto_suggest=suggester, 
                            completer=completer, 
                            bottom_toolbar=LoaderJobHandler.toolbar,
                            refresh_interval=1.0,
                        )
                        if cmd.strip() == "":
                            continue
                        context: CommandContext = self.interpreter.run_command(cmd)
                        outputs, df = [context.logger.get_output(), context.result]
                        print("\n".join(outputs))
                        if isinstance(df, pd.DataFrame):
                            with pd.option_context('display.max_rows', None):
                                if df.shape[0] == 0:
                                    continue
                                fmt_opts = {
                                    "index": False,
                                    "max_rows" : None,
                                    "min_rows" : 10,
                                    "max_colwidth": 50 if not context.print_wide else None,
                                    "header": True,
                                    "float_format": '{:0.2f}'.format
                                }
                                if df.shape[0] > 40:
                                    pydoc.pager(df.to_string(**fmt_opts))
                                else:
                                    print(df.to_string(**fmt_opts))
                        elif 'altair' in str(type(df)): # instead of isinstance(altair.Chart) we can avoid loading altari at the start
                            df.show()
                            sys.exit(0)
                        elif df is not None:
                            print(df)
                    except RuntimeError as e:
                        print(e)
                    except KeyboardInterrupt:
                        print("Operation interrupted")
                    except Exception as e:
                        if isinstance(e, EOFError):
                            raise
                        traceback.print_exc()
        except EOFError:
            sys.exit(0)
