# Module and class layout
#
# __main__ - contains the repl main loop and creates the db connection
#   interpreter.py - Defines the CommandInterpreter which implements all Unify commands
#   loading.py - Defines TableLoader which knows how to load local db tables from Adapters (via Connections)
#   adapters.py - Defines Connection and all Adapter base classes 
#   db_wrapper.py - Defines all local db managers
#
#     rest_adapter.py - Defines the core REST API adapter
#     grammar.lark - Contains the Lark grammar for the interpreter
#  
import os
import os.path
from dotenv import load_dotenv

if 'UNIFY_HOME' not in os.environ:
    os.environ['UNIFY_HOME'] = os.path.expanduser("~/unify")
conf_path = os.path.join(os.environ['UNIFY_HOME'], 'unify_config')
if os.path.exists(conf_path):
    load_dotenv(conf_path)

if 'DATABASE_BACKEND' not in os.environ:
    print("Welcome to Unify. Do you want to use DuckDB or Clickhouse as your local database?")
    choice = input("1) DuckDB, 2) Clickhouse: ")
    if choice == '2':
        os.environ['DATABASE_BACKEND'] = 'clickhouse'
    else:
        os.environ['DATABASE_BACKEND'] = 'duckdb'
if 'DATABASE_USER' not in os.environ:
    os.environ['DATABASE_USER'] = os.getenv('USER', 'default')

if not os.path.exists(conf_path):
    with open(conf_path, 'w') as f:
        for var in [key for key in os.environ.keys() if key.startswith("DATABASE")]:
            f.write(f"{var}={os.environ[var]}\n")
    
from .interpreter import CommandInterpreter, CommandContext
from .db_wrapper import dbmgr

__version__ = "0.2.0"


