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

if 'UNIFY_HOME' not in os.environ:
    os.environ['UNIFY_HOME'] = os.path.expanduser("~/unify")
    
from .interpreter import CommandInterpreter, CommandContext
from .db_wrapper import dbmgr

__version__ = "0.2.0"


