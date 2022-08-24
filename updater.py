# Simple script to keep Unify tables up to date
import subprocess
import sys
from datetime import datetime

from unify import CommandInterpreter

if __name__ == '__main__':
    interpreter = CommandInterpreter(debug=True, silence_errors=True)

    with open("/tmp/unify-updater.log", "a") as log_file:
        def log(msg, *args):
            print(msg, *args)
            print(str(datetime.now()), ": ", msg, *args, file=log_file)

        for schema in interpreter._list_schemas():
            tables = interpreter._list_tables_filtered(schema)
            for table_root in tables:
                log("Reloading ", schema, ".", table_root)
                table = schema + "." + table_root
                for sub_cmd in [
                    f"python unify.py -e 'count {table}'",
                    f"python unify.py -e 'refresh table {table}'"
                ]:
                    log(sub_cmd)
                    output = subprocess.run(sub_cmd, shell=True, check=False, capture_output=True)
                    log(output.stdout)

