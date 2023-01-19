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
import webbrowser
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
        answer = input("Do you need to install Clickhouse? (y/n): ")
        if answer == 'y':
            webbrowser.open("https://clickhouse.com/docs/en/install/#self-managed-install")
        print("Please enter your Clickhouse credentials:")
        host = input("Host name [localhost]: ")
        username = input("Username [default]: ")
        password = input("Password: ")
        if not host:
            host = "localhost"
        if not username:
            username = "default"
        os.environ['DATABASE_HOST'] = host
        os.environ['DATABASE_USER'] = username
        os.environ['DATABASE_PASSWORD'] = password
        os.environ['DATABASE_BACKEND'] = 'clickhouse'
    else:
        os.environ['DATABASE_BACKEND'] = 'duckdb'
if 'DATABASE_USER' not in os.environ:
    os.environ['DATABASE_USER'] = os.getenv('USER', 'default')

if not os.path.exists(conf_path):
    os.makedirs(os.environ['UNIFY_HOME'], exist_ok=True)
    with open(conf_path, 'w') as f:
        for var in [key for key in os.environ.keys() if key.startswith("DATABASE")]:
            f.write(f"{var}={os.environ[var]}\n")
    
from .interpreter import CommandInterpreter, CommandContext
from .db_wrapper import dbmgr

__version__ = "0.2.3"


