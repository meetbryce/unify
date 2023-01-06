import sys
import pandas as pd

from .interpreter import CommandInterpreter
from .command_repl import UnifyRepl

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
