import logging
import pydoc
import os
import sys
import pandas as pd
import threading
import traceback

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.patch_stdout import patch_stdout
from pygments.lexers.sql import SqlLexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.filters import has_focus
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.key_binding import KeyBindings

from .interpreter import CommandInterpreter, CommandContext, setup_job_log_handler

logger: logging.Logger = logging.getLogger('unify')
logger.setLevel(logging.INFO)
global job_record
job_record = None

class LoaderJobHandler(logging.Handler):  
    def __init__(self):
        super().__init__()

    def emit(self, record):
        global job_record
        job_record = record
        if record.levelno == logging.CRITICAL:
            print(record.msg)

    @staticmethod
    def toolbar():
        global job_record
        if job_record:
            if hasattr(job_record, '_rows_loaded'):
                row_count = f"- {job_record._rows_loaded} rows"
            else:
                row_count = ""
            if hasattr(job_record, '_table_root'):
                return f"[Loading {job_record._schema}:{job_record._table_root} {row_count}] {job_record.msg} "
            else:
                try:
                    return (job_record.msg + row_count) % job_record.args
                except:
                    return job_record.msg
        else:
            return "No log record"

setup_job_log_handler(LoaderJobHandler())
last_content: str = ""

class UnifyRepl:
    def __init__(self, interpreter: CommandInterpreter, wide_display=False):
        self.interpreter = interpreter
        if wide_display:
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)

    def loop(self):
        kb = KeyBindings()
        @kb.add('enter', filter=has_focus(DEFAULT_BUFFER))
        def handle_enter(event):
            global last_content
            # Your stuff.
            content = session.default_buffer.text.strip()
            if len(content) > len(last_content) and content.startswith("select") and not content.endswith(";"):
                # continue the prompt for select statements
                session.default_buffer.insert_text("\n> ")
                last_content = content
                return
            else:
                # finsh the prompt
                last_content = ""
                session.default_buffer.validate_and_handle()

        session = PromptSession(history=FileHistory(os.path.expanduser("~/.pphistory")), key_bindings=kb)
        suggester = AutoSuggestFromHistory()
        try:
            with patch_stdout():
                while True:
                    try:
                        cmd = session.prompt("> ", 
                            auto_suggest=suggester, 
                            completer=self.interpreter.get_prompt_completer(), 
                            bottom_toolbar=LoaderJobHandler.toolbar,
                            refresh_interval=1.0,
                            lexer=PygmentsLexer(SqlLexer)
                        )
                        cmd = cmd.replace("\n> ", " ")
                        cmd = cmd.strip()
                        if cmd == "":
                            continue

                        context: CommandContext = self.interpreter.run_command(cmd, input_func=session.prompt)
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
                            t = threading.Thread(target=df.show)
                            t.start()
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
