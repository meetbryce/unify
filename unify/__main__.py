import pydoc
import os
import sys
import pandas as pd
import traceback

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from .interpreter import CommandInterpreter, CommandContext
from .db_wrapper import (
    ClickhouseWrapper, 
    DBManager, 
    DuckDBWrapper
)

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
            while True:
                try:
                    cmd = session.prompt("> ", auto_suggest=suggester)
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


if __name__ == '__main__':
    if '-silent' in sys.argv:
        silent = True
    else:
        silent = False

    interpreter = CommandInterpreter(silence_errors=silent)

    for i in range(len(sys.argv)):
        if sys.argv[i] == '-e':
            command = sys.argv[i+1]
            with pd.option_context('display.max_rows', None):
                lines, df = interpreter.run_command(command)
                print("\n".join(lines))
                if df is not None:
                    print(df)
            sys.exit(0)

    UnifyRepl(interpreter).loop()
