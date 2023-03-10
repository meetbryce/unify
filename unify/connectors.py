import importlib.resources
import glob
import os
import re
from pprint import pprint
import logging
import shutil
import sys
from tempfile import NamedTemporaryFile
import yaml
from typing import List, AnyStr, Dict, Union, Iterable, Generator
import typing
from datetime import datetime
from collections import namedtuple

from .storage_manager import StorageManager
from .data_utils import interp_dollar_values

logger = logging.getLogger(__name__)

Connector = typing.NewType("Connector", None)
TableUpdater = typing.NewType("TableUpdater", None)


class Connection:
    CONNECTOR_SPECS: dict = {}
    SPEC_PACKAGE = "unify.rest_specs"

    def __init__(self, connector, schema_name, opts):
        self.schema_name: str = schema_name
        self.connector: Connector = connector
        self.connector.resolve_auth(self.schema_name, opts['options'])
        self.is_valid = self.connector.validate()

    @staticmethod
    def connections_config_path():
        return os.path.join(os.environ['UNIFY_HOME'], "unify_connections.yaml")

    @staticmethod
    def connections_config_exists():
        return os.path.exists(Connection.connections_config_path())
        
    @staticmethod
    def load_connections_config():
        return yaml.safe_load(open(Connection.connections_config_path()))

    @staticmethod
    def update_connections_config(schema_name, connector_name, options):
        path = Connection.connections_config_path()
        try:
            conf = yaml.safe_load(open(path))
            orig_lines = open(path).readlines()
        except FileNotFoundError:
            conf = []
            orig_lines = []

        conf.append({schema_name: {"connector": connector_name, "options": options}})
        
        # write out the new config, but preserve comments from the old file
        with open(path, "w") as f:
            for line in orig_lines:
                if line.startswith("#"):
                    f.write(line)
                elif conf is not None:
                    yaml.dump(conf, f, default_flow_style=False)
                    conf = None
            if conf is not None:
                yaml.dump(conf, f, default_flow_style=False)

    @staticmethod
    def delete_connection_config(schema_name):
        path = Connection.connections_config_path()
        conf: list = yaml.safe_load(open(path))

        conf = [conn for conn in conf if list(conn.keys())[0] != schema_name]

        # write out the new config, but preserve comments from the old file
        with open(path, "w") as f:
            yaml.dump(conf, f, default_flow_style=False)

    @classmethod
    def setup_connections(cls, conn_list=None, connections_path=None, storage_mgr_maker=None):
        from .rest_connector import RESTConnector

        connector_table = {}

        Connection.externalize_spec_file("spec_doc.yaml")

        def find_connector_specs() -> Generator[str, None, None]:
            for f in glob.glob(os.path.join(os.environ['UNIFY_HOME'], "connectors", "*spec.yaml")):
                yield f
            for f in importlib.resources.contents(Connection.SPEC_PACKAGE):
                if f.endswith("spec.yml")  or f.endswith("spec.yaml"):
                    with importlib.resources.path(Connection.SPEC_PACKAGE, f) as path:
                        yield path

        for f in find_connector_specs():
            base_name = os.path.basename(f)
            try:
                spec = yaml.load(open(f), Loader=yaml.FullLoader)
            except Exception as e:
                print(f"Error loading connector spec {f}: {e}")
                continue
            if spec.get('enabled') == False:
                continue
            if 'name' not in spec:
                print(f"Skipping connector spec {f}: no name specified")
                continue
            name = spec['name']
            if name in connector_table:
                continue
            klass = RESTConnector
            if 'class' in spec:
                if spec['class'].lower() == 'gsheetsadapter':
                    from gsheets_unify_adapter.gsheets_adapter import GSheetsAdapter
                    klass = GSheetsAdapter
                elif spec['class'].lower() == 'postgresconnector':
                    from .postgres_connector import PostgresConnector
                    klass = PostgresConnector           
            connector_table[name] = (klass, spec, f)

        if conn_list:
            connections = conn_list
        elif connections_path is not None:
            connections = yaml.safe_load(open(connections_path))
        else:
            try:
                connections = Connection.load_connections_config()
            except FileNotFoundError:
                #print("Warning, no connections config file found. Use 'connect' command to create one.")
                connections = []
        result = []
        # Instantiate each connector, resolve auth vars, and validate the connection
        for opts in connections:
            schema_name = next(iter(opts))
            opts = opts[schema_name]
            if opts['connector'] not in connector_table:
                print(f"Error: cannot find connector {opts['connector']} for connection {schema_name}")
                print("Available connectors: ", connector_table.keys())
                continue
            connector_klass, spec, _ = connector_table[opts['connector']]
            connector = connector_klass(spec, storage_mgr_maker(schema_name), schema_name)
            c = Connection(connector, schema_name, opts)
            if c.is_valid:
                result.append(c)
            else:
                print(f"Failed to load connection '{schema_name}' as connector is invalid", file=sys.stderr)

        Connection.CONNECTOR_SPECS.update(connector_table)
        return result

    @classmethod
    def create_connection(cls, connector_name: str, schema_name: str, opts: dict, storage_mgr_maker=None):
        connector_klass, spec, spec_file = cls.CONNECTOR_SPECS[connector_name]
        if not Connection.externalize_spec_file(spec_file):
            print(f"Using user spec {spec_file}")
        connector = connector_klass(spec, storage_mgr_maker(schema_name), schema_name)
        c = Connection(connector, schema_name, {"options": opts})
        c.test_connection()
        Connection.update_connections_config(schema_name, connector_name, opts)
        return c

    @classmethod
    def externalize_spec_file(cls, spec_file: str) -> bool:
        """ Returns True if the provided spec is bult-in and we externalized it, otherwise returns False. """
        user_specs = os.path.join(os.environ['UNIFY_HOME'], 'connectors')
        if os.path.dirname(spec_file) == user_specs and os.path.exists(spec_file):
            return False # already externalized
        dest_path = os.path.join(user_specs, spec_file)
        os.makedirs(user_specs, exist_ok=True)

        if not os.path.exists(dest_path):
            with importlib.resources.path(Connection.SPEC_PACKAGE, spec_file) as path:
                shutil.copy(path, user_specs)
        return True


    def list_tables(self):
        return self.connector.list_tables()

    def test_connection(self):
        # Used to verify valid auth for a new connection
        self.connector.test_connection(logger=logger)

class OutputLogger:
    def __init__(self) -> None:
        self.buffer = []
        self.df = None

    def print(self, *args) -> None:
        self.buffer.append("".join([str(s) for s in args]))

    def print_block(self, msg):
        """ Print a (potentially) multi-line message """
        self.buffer.extend(msg.split("\n"))

    def print_df(self, df):
        self.df = df

    def get_output(self):
        return self.buffer

    def get_df(self):
        return self.df

    def clear(self):
        self.buffer = []
        self.df = None
        

ConnectorQueryResult = namedtuple(
    'ConnectorQueryResult', 
    ['json','size_return','merge_cols','rows_written','message'],
    defaults=(None, None, None)
)

class TableDef:
    def __init__(self, name, description=None):
        self._name = name
        self._select_list = []
        self._result_body_path = None
        self._result_object_path = None
        self.result_meta_paths = None
        self._key = None
        self._queryDateFormat = None #Use some ISO default
        self._params: dict = {}
        self.description = description
        self._strip_prefixes: Union[str,list] = []

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, val):
        self._key = val

    @property
    def params(self) -> dict:
        return self._params

    @params.setter
    def params(self, val: dict) -> None:
        self._params = val

    @property
    def query_date_format(self) -> str:
        return self._queryDateFormat

    @query_date_format.setter
    def query_date_format(self, val):
        self._queryDateFormat = val

    @property
    def select_list(self) -> list:
        return self._select_list

    @select_list.setter
    def select_list(self, selects: list):
        self._select_list = selects

    @property
    def strip_prefixes(self):
        return self._strip_prefixes

    @strip_prefixes.setter
    def strip_prefixes(self, value):
        self._strip_prefixes = value

    def query_resource(self, tableLoader, logger: logging.Logger) -> Generator[ConnectorQueryResult, None, None]:
        """ Yields ConnectorQueryResults for each page of an API endpoint """
        return None

    @property
    def result_body_path(self):
        return self._result_body_path

    @result_body_path.setter
    def result_body_path(self, val):
        self._result_body_path = val

    @property
    def result_object_path(self):
        return self._result_object_path

    @result_object_path.setter
    def result_object_path(self, val):
        self._result_object_path = val

    def get_table_updater(self, updates_since: datetime) -> TableUpdater:
        # default is just to reload
        return ReloadStrategy(self)

    def get_table_source(self):
        # Returns a description of where we are loading data from for the table
        return None

class TableUpdater:
    def __init__(self, table_def: TableDef, updates_since: datetime=None) -> None:
        self.table_def: TableDef = table_def
        self.updates_timestamp = updates_since

    """
        Defines the contract for classes which update existing tables. We support 
        multiple update strategies (the "refresh strategy") so we have a subclass
        for each strategy.
    """
    def should_replace(self) -> bool:
        """ Returns True if data should be loaded into a temp table and that table used
            to replace the existing table. Otherwise data is appended in place. 
            Generally this is only used by the "reload" refresh strategy.
        """
        return False

    def query_resource(self, tableLoader, logger: logging.Logger) -> Generator[ConnectorQueryResult, None, None]:
        """ Generator which yields ConnectorQueryResults for all records updated
            since the `updates_since` timestamp.
        """
        return None

class ReloadStrategy(TableUpdater):
    def __init__(self, table_def: TableDef) -> None:
        super().__init__(table_def)

    def should_replace(self) -> bool:
        return True

    def query_resource(self, tableLoader, logger: logging.Logger):
        """ Just delegate to the TableDef like a first load. """
        for query_result in self.table_def.query_resource(tableLoader, logger):
            yield query_result

class UpdatesStrategy(TableUpdater):
    def __init__(self, table_def: TableDef, updates_since: datetime) -> None:
        super().__init__(table_def, updates_since)
        # The refresh strategy config spec should indicate the query parameter
        # expression to use for filtering for updated records. The source table
        # must also define a key column
        if self.table_def.key is None:
            raise RuntimeError(
                f"Table '{self.table_def.name}' needs to define a key to use 'updates' refresh strategy")
        self.refresh = self.table_def.refresh
        self.params = self.refresh.get('params')
        for k, v in self.params.items():
            if not isinstance(k, str):
                raise RuntimeError(
                    f"Invalid refresh strategy parameter '{k}' type {type(k)} for table '{self.table_def.name}'")
            elif not isinstance(v, str):
                raise RuntimeError(
                    f"Invalid refresh strategy parameter '{v}' type {type(v)} table '{self.table_def.name}'")

        if not self.params:
            raise RuntimeError(
                f"Table '{self.table_def.name}' missing 'params' for 'updates' refresh strategy")


    def query_resource(self, tableLoader, logger: logging.Logger):
        # Need interpolate the checkpoint time into the GET request
        # parameters, using the right format for the source system

        timestamp = self.updates_timestamp.strftime(self.table_def.query_date_format)

        args = interp_dollar_values(self.params, {
            "timestamp": timestamp
        })

        # Overwride the static params set in the Table spec
        try:
            save_params = self.table_def.params
            self.table_def.params = dict(args)

            """ Just delegate to the TableDef like a first load. """
            for query_result in self.table_def.query_resource(tableLoader, logger):
                yield query_result
        finally:
            self.table_def.params = save_params


RESTView = namedtuple(
    'RESTView', 
    ['name','from_list','query', 'help'], 
    defaults={'help':None}
)


class Connector:
    def __init__(self, name, storage: StorageManager):
        self.name = name
        self.help = ""
        self.auth: dict = {}
        self.storage: StorageManager = storage

    def get_config_parameters(self):
        """ Returns a dict mapping config parameter names to descriptions. """       
        return {}

    def test_connection(self, logger: logging.Logger):
        pass

    @staticmethod
    def convert_string_to_table_name(title: str):
        title = title.lower()
        title = re.sub(r"\s+", "_", title)
        if re.match(r"^\d+", title):
            # Can't start with numbers
            title = "tab" + title
        title = re.sub(r"[^\w]+", "", title) # remove special characters
        if len(title) < 7:
            title = "table_" + title
        return title

    def validate(self) -> bool:
        return True

    def list_tables(self) -> List[TableDef]:
        pass

    def lookupTable(self, tableName: str) -> TableDef:
        return next(t for t in self.list_tables() if t.name == tableName)

    def resolve_auth(self, connection_name: AnyStr, connection_opts: Dict):
        # The connector spec has an auth clause (self.auth) that can refer to "Connection options". 
        # The Connection options can refer to environment variables or hold direct values.
        # We need to:
        # 1. Resolve the env var references in the Connection options
        # 2. Copy the connection options into the REST API spec's auth clause
        if not isinstance(connection_opts, dict):
            raise RuntimeError(
                f"Bad connection options, expected a dict got type {type(connection_opts)}: ", 
                connection_opts
            )
        for k, value in connection_opts.items():
            if isinstance(value,str) and value.startswith("$"):
                try:
                    value = os.environ[value[1:]]
                    connection_opts[k] = value
                except KeyError:
                    print(f"Authentication for {connection_name} failed, missing env var '{value[1:]}'")
                    sys.exit(1)
        def resolve_auth_values(auth_tree, conn_opts):
            for key, value in auth_tree.items():
                if key == 'type':
                    continue
                elif isinstance(value, dict):
                    resolve_auth_values(value, conn_opts)
                else:
                    if key in conn_opts:
                        auth_tree[key] = conn_opts[key]
                    # elif "{" in  value:
                    #     # Resolve a string with connection opt references
                    #     auth_tree[key] = value.format(**conn_opts)
                    # elif not re.match(r"[A-Z_]+", value):
                    #     # allow static values that are not like XXX_XXX
                    #     pass
                    # else:
                    #     print(f"Error: auth key {key} missing value in connection options")
        resolve_auth_values(self.auth, connection_opts)

    def __repr__(self) -> AnyStr:
        return f"{self.__class__.__name__}({self.name}) ->\n" + \
            ", ".join(map(lambda t: str(t), self.tables or [])) + "\n"

    def __str__(self) -> str:
        return self.name

    @property
    def base_api_url(self) -> str:
        return getattr(self, 'base_url', '')

    def supports_commands(self) -> bool:
        return self.help is not None

    def run_command(self, code: str, output_logger: OutputLogger) -> OutputLogger:
        if code.strip() == 'help':
            if self.help:
                output_logger.print_block(self.help)
            else:
                output_logger.print(f"No help available for connector {self.name}")
            return output_logger
        else:
            return None

    # Importing data
    def can_import_file(self, file_uri: str):
        """ Return true if this connector nows how to import the indicated file_uri """
        return False

    def import_file(self, file_uri: str, options: dict={}):
        """" Import data from the indicated file. Returns the name of the table created from the import
             or raises an error if there was one. """
        pass

    def peek_file(self, file_uri: str, line_count: int, logger: OutputLogger):
        logger.print("Subclass must implement peek_file command")

    # Exporting data
    def create_output_table(self, file_name, output_logger:OutputLogger, overwrite=False, opts={}):
        raise RuntimeError(f"Connector {self.name} does not support writing")

    def write_page(self, output_handle, page, output_logger:OutputLogger, append=False):
        raise RuntimeError(f"Connector {self.name} does not support writing")

    def close_output_table(self, output_handle):
        raise RuntimeError(f"Connector {self.name} does not support writing")

    def list_views(self) -> List[RESTView]:
        return []

    def drop_table(self, table_root: str):
        pass

    def rename_table(self, table_root: str, new_name: str):
        pass
