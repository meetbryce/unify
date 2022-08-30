import glob
import os
import re
import string
import sys
import yaml
from typing import List, AnyStr, Dict, Union
import typing
from datetime import datetime

import requests
import pandas as pd
from jsonpath_ng import parse

from .storage_manager import StorageManager

Adapter = typing.NewType("Adapter", None)
TableUpdater = typing.NewType("TableUpdater", None)

class Connection:
    def __init__(self, adapter, schema_name, opts):
        self.schema_name: str = schema_name
        self.adapter: Adapter = adapter
        self.adapter.resolve_auth(self.schema_name, opts['options'])
        self.is_valid = self.adapter.validate()

    @classmethod
    def setup_connections(cls, path=None, conn_list=None, storage_mgr_maker=None):
        adapter_table = {}
        for f in glob.glob(os.path.join(os.path.dirname(__file__), "../rest_specs/*spec.yaml")):
            spec = yaml.load(open(f), Loader=yaml.FullLoader)
            if spec.get('enabled') == False:
                continue
            klass = RESTAdapter
            if 'class' in spec and spec['class'].lower() == 'gsheetsadapter':
                from .gsheets.gsheets_adapter import GSheetsAdapter
                klass = GSheetsAdapter
            adapter_table[spec['name']] = (klass, spec)
        
        if conn_list:
            connections = conn_list
        else:
            connections = yaml.load(open(path), Loader=yaml.FullLoader)
        result = []
        # Instantiate each adapter, resolve auth vars, and validate the connection
        for opts in connections:
            schema_name = next(iter(opts))
            opts = opts[schema_name]
            adapter_klass, spec = adapter_table[opts['adapter']]
            adapter = adapter_klass(spec, storage_mgr_maker(schema_name))
            c = Connection(adapter, schema_name, opts)
            if c.is_valid:
                result.append(c)
            else:
                print("Failed to load connection {schema_name} as adapter is invalid", file=sys.stderr)
        return result

    def list_tables(self):
        return self.adapter.list_tables()

class RESTCol:
    def __init__(self, dictvals):
        self.name = dictvals['name'].lower()
        self.comment = dictvals.get('comment')
        self.type = dictvals.get('type', 'VARCHAR')
        # Valid types are: VARCHAR, INT, BOOLEAN, JSON
        self.source = dictvals.get('source') or self.name
        self.source_fk = dictvals.get('source_fk')
        self.is_key = dictvals.get('key') == True
        self.is_timestamp = dictvals.get('timestamp') == True
        self.hidden = dictvals.get('hidden') == True

    @staticmethod
    def build(name=None, type='VARCHAR', source=None):
        return RESTCol({"name": name, "type": type, "source": source})

    def __repr__(self):
        return f"RESTCol({self.name}<-{self.source},{self.type})"


########## REST Paging helpers
class PagingHelper:
    def __init__(self, options):
        page_size = (options or {}).get('page_size')
        if page_size:
            self.page_size = int(page_size)
        else:
            self.page_size = 1

    @staticmethod
    def get_pager(options):
        if not options:
            return NullPager(options)

        if options['strategy'] == 'pageAndCount':
            return PageAndCountPager(options)
        elif options['strategy'] == 'offsetAndCount':
            return OffsetAndCountPager(options)
        elif options['strategy'] == 'pagerToken':
            return PagerTokenPager(options)
        else:
            raise RuntimeError("Unknown paging strategy: ", options)

    def next_page(self, last_page_size: int, json_result: Union[dict, list]) -> bool:
        return False

class NullPager(PagingHelper):
    def __init__(self, options):
        super().__init__(options)

    def get_request_params(self):
        return {}

    def next_page(self, last_page_size: int, json_result: Union[dict, list]) -> bool:
        return False


class PageAndCountPager(PagingHelper):
    def __init__(self, options):
        super().__init__(options)
        if 'page_param' not in options:
            raise RuntimeError("page_param not specified in options")
        if 'count_param' not in options:
            raise RuntimeError("count_param not specified in options")
        self.index_param = options['page_param']
        self.count_param = options['count_param']
        self.current_page = 1

    def get_request_params(self):
        return {self.index_param: self.current_page, self.count_param: self.page_size}

    def next_page(self, last_page_size: int, json_result: Union[dict, list]) -> bool:
        self.current_page += 1
        return last_page_size >= self.page_size

class OffsetAndCountPager(PagingHelper):
    def __init__(self, options):
        super().__init__(options)
        if 'offset_param' not in options:
            raise RuntimeError("offset_param not specified in options")
        if 'count_param' not in options:
            raise RuntimeError("count_param not specified in options")
        self.offset_param = options['offset_param']
        self.count_param = options['count_param']
        self.current_offset = 0

    def get_request_params(self):
        return {self.offset_param: self.current_offset, self.count_param: self.page_size}

    def next_page(self, last_page_size: int, json_result: Union[dict, list]) -> bool:
        self.current_offset += last_page_size
        return last_page_size >= self.page_size

class PagerTokenPager(PagingHelper):
    def __init__(self, options):
        super().__init__(options)
        if 'token_param' not in options:
            raise RuntimeError("pager_token_param not specified in options")
        if 'count_param' not in options:
            raise RuntimeError("count_param not specified in options")
        if 'pager_token_path' not in options:
            raise RuntimeError("pager_token_path not specified in options")
        self.token_param = options['token_param']
        self.count_param = options['count_param']
        self.token_expr = parse(options['pager_token_path'])
        self.current_token = None

    def get_request_params(self):
        if self.current_token:
            return {self.token_param: self.current_token, self.count_param: self.page_size}
        else:
            return {self.count_param: self.page_size}

    def next_page(self, last_page_size: int, json_result: Union[dict, list]) -> bool:
        for match in self.token_expr.find(json_result):
            self.current_token = match.value
            return last_page_size >= self.page_size
        return False

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
        
class UnifyLogger:
    INFO = 1
    WARNING = 2
    ERROR = 3

    def log_table(table: str, level: int, *args) -> None:
        pass


class TableDef:
    def __init__(self, name):
        self._name = name
        self._select_list = []
        self._result_body_path = None
        self._key = None
        self._queryDateFormat = None #Use some ISO default
        self._params: dict = {}

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

    def query_resource(self, tableLoader, logger: UnifyLogger):
        pass

    @property
    def result_body_path(self):
        return self._result_body_path

    @result_body_path.setter
    def result_body_path(self, val):
        self._result_body_path = val

    def get_table_updater(self, updates_since: datetime) -> TableUpdater:
        # default is just to reload
        return ReloadStrategy(self)

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

    def query_resource(self, tableLoader, logger: UnifyLogger):
        """ Generator which yields (page, size_return) tuples for all records updated
            since the `updates_since` timestamp.
        """
        pass


class RESTTable(TableDef):
    def __init__(self, spec, dictvals):
        super().__init__(dictvals['name'])
        fmt = string.Formatter()
        self.max_pages = 50000

        self._supports_paging = dictvals.get('supports_paging', False)
        if self._supports_paging:
            self.paging_options = spec.paging_options
        else:
            self.paging_options = None

        self.spec = spec
        self.key = dictvals.get('key_column')
        self.name = dictvals['name']
        self.query_path = dictvals.get('resource_path', '')
        self.query_method = dictvals.get('method', 'GET')
        self.query_date_format = spec.query_date_format

        if dictvals.get('query_resource'):
            self.query_method, self.query_path = self.parse_resource(dictvals.get('query_resource'))
        self.query_args = [t[1] for t in fmt.parse(self.query_path) if t[1] is not None]        

        self.select_list = dictvals.get('select')
        if self.select_list:
            self.select_list = re.split(r",\s*", self.select_list)

        if 'refresh' in dictvals:
            self.refresh = dictvals['refresh']
            self.refresh_strategy = self.refresh['strategy']
        else:
            self.refresh_strategy = 'reload'

        # REMOVE OLD REFRESH CODE
        self.refresh_params = dictvals.get('refresh_params', {})
        if '_ts_format' in self.refresh_params:
            self.refresh_ts_format = self.refresh_params.pop('_ts_format')
        else:
            self.refresh_ts_format = '%Y-%m-%dT%H:%M:%S'

        self.create_method, self.create_path = self.parse_resource(dictvals.get('create_resource'))
        self.create_args = [t[1] for t in fmt.parse(self.create_path or '') if t[1] is not None]
        self.update_method, self.update_path = self.parse_resource(dictvals.get('update_resource'))
        self.update_args = [t[1] for t in fmt.parse(self.update_path or '') if t[1] is not None]
        self.delete_method, self.delete_path = self.parse_resource(dictvals.get('delete_resource'))
        self.delete_args = [t[1] for t in fmt.parse(self.delete_path or '') if t[1] is not None]

        if (self.query_path == '<inline>'):
            self.static_values = dictvals.get('values')
        self.result_body_path = dictvals.get('result_body_path')
        self.result_type = dictvals.get('result_type') or 'list'

        # parse columns
        self.columns = []
        # add <args> columns from the resource path
        for idx, qarg in enumerate(self.query_args or []):
            if qarg.startswith('*'):
                self.columns.append(RESTCol.build(name=qarg[1:], source="<args>"))
                self.query_args[idx] = qarg[1:]
        # strip column annotations
        self.query_path = re.sub(r'{\*(\w+)\}', '{\\1}', self.query_path)        

        self.colmap = {col.name: col for col in self.columns}
        self.params = dictvals.get('params', {})
        self.request_data = dictvals.get('body')
        if dictvals.get('args_query'):
            self.args_query_table = dictvals['args_query']['table']
            self.args_query_mappings = dictvals['args_query']['mapping']
            self.args_query_exclusions = (dictvals['args_query'].get('exclude') or "").split(",")
        else:
            self.args_query_table = None
        self.keyColumnName = self.keyColumnType = None
        for c in self.columns:
            if c.is_key:
                self.keyColumnName = c.name
                self.keyColumnType = c.type
        if self.keyColumnName is None:
            #print(f"!!WARNING: table {spec.name}.{self.name} has no key column defined")
            self.is_cacheable = False
        else:
            self.is_cacheable = True

        # TODO: What about multiple keys?

    def parent_table(self):
        if self.args_query_table:
            return self.spec.lookupTable(self.args_query_table)
        else:
            return None

    def supports_paging(self):
        return self._supports_paging

    def supports_updates(self):
        return self.update_path is not None

    def parse_resource(self, resource_spec):
        if resource_spec is None:
            return None, None
        parts = resource_spec.split(" ")
        if len(parts) == 2:
            method = parts[0].upper()
            path = parts[1]
            if method not in ['GET','POST','PATCH','PUT','DELETE']:
                raise Exception(f"Invalid resource spec method '{method}' for resource: '{resource_spec}'")
            return method, path
        else:
            raise Exception(f"Invalid resource '{resource_spec}', must be '<http method> <path>'")

    def __repr__(self):
        cols = sorted({c.name for c in self.columns})
        cols = ", ".join(cols)
        return f"RESTTable({self.name})[{cols}] ->"

    def qualified_name(self):
        return self.spec.name + "." + self.name

    def get_table_updater(self, updates_since: datetime) -> TableUpdater:
        if self.refresh_strategy == 'reload':
            return ReloadStrategy(self)
        elif self.refresh_strategy == 'updates':
            return UpdatesStrategy(self, updates_since=updates_since)
        else:
            raise RuntimeError(
                f"Invalid refresh strategy '{self.refresh_strategy}' for table {self.name}"
            )

    def query_resource(self, tableLoader, logger: UnifyLogger):
        """ Generator which yields (page, size_return) tuples for all rows from
            an API endpoint. Each page should be a list of dicts representing
            each row of results.

            Because a resource may reference a parent query to provide API query
            args, the 'tableLoader' argument is provided for querying the rows
            of a parent table.
        """
        if self.parent_table():
            # Run our parent query and then our query with parameters from each parent record
            print(">>Running parent query: ", self.parent_table())
            for df in tableLoader.read_table_rows(self.parent_table().qualified_name()):
                for record in df.to_dict('records'):
                    do_row = True
                    for key in self.args_query_mappings:
                        parent_col = self.args_query_mappings[key]
                        if parent_col in record:
                            record[key] = record[parent_col]
                            if record[key] in self.args_query_exclusions:
                                logger.log_table(
                                    self.name, 
                                    UnifyLogger.DEBUG,
                                    "Excluding parent row with key: ", record[key]
                                )
                                do_row = False
                        else:
                            logger.log_table(
                                self.name,
                                UnifyLogger.WARNING,
                                f"Error: parent record missing col {parent_col}: ", record
                            )
                    if do_row:
                        for page, size_return in self._query_resource(record, logger):
                            yield (page, size_return)
        else:
            # Simple query
            for page, size_return in self._query_resource(logger=logger):
                yield (page, size_return)

    def _query_resource(
        self, 
        query_params={}, 
        logger: UnifyLogger = None
        ):
        session = requests.Session()
        self.spec._setup_request_auth(session)

        pager = PagingHelper.get_pager(self.paging_options)
        params = self.params.copy()
        page = 1
        safety_max_pages = 200000 # prevent infinite loop in case "pages finished logic" fails

        while page < safety_max_pages:
            url = (self.spec.base_url + self.query_path).format(**query_params)
            params.update(pager.get_request_params())
            
            print(url, params)
            r = session.get(url, params=params)

            if r.status_code >= 400:
                print(r.text)
            if r.status_code == 404:
                logger.log_table(self.name, UnifyLogger.ERROR, f"404 returned from {url}")
                return
            if r.status_code == 401:
                logger.log_table(self.name, UnifyLogger.ERROR, f"401 returned from {url}")
                raise Exception("API call unauthorized: " + r.text)
            if r.status_code >= 400:
                logger.log_table(
                    self.name, 
                    UnifyLogger.ERROR, 
                    f"HTTP error {r.status_code} returned from {url}"
                )
                return

            size_return = []

            json_result = r.json()
            yield (json_result, size_return)

            if not pager.next_page(size_return[0], json_result):
                break

            page += 1
            if page > self.max_pages:
                print("Aborting table scan after {} pages", page-1)
                break

class ReloadStrategy(TableUpdater):
    def __init__(self, table_def: TableDef) -> None:
        super().__init__(table_def)

    def should_replace(self) -> bool:
        return True

    def query_resource(self, tableLoader, logger: UnifyLogger):
        """ Just delegate to the TableDef like a first load. """
        for page, size_return in self.table_def.query_resource(tableLoader, logger):
            yield (page, size_return)

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


    def query_resource(self, tableLoader, logger: UnifyLogger):
        # Need interpolate the checkpoint time into the GET request
        # parameters, using the right format for the source system

        timestamp = self.updates_timestamp.strftime(self.table_def.query_date_format)
        args = {k:v.replace("{timestamp}",timestamp) for (k,v) in self.params.items()}

        # Overwride the static params set in the Table spec
        try:
            save_params = self.table_def.params
            self.table_def.params = dict(args)

            """ Just delegate to the TableDef like a first load. """
            for page, size_return in self.table_def.query_resource(tableLoader, logger):
                yield (page, size_return)
        finally:
            self.table_def.params = save_params


class Adapter:
    def __init__(self, name, storage: StorageManager):
        self.name = name
        self.help = ""
        self.auth: dict = {}
        self.storage: StorageManager = storage

    def validate(self) -> bool:
        return True

    def list_tables(self) -> List[TableDef]:
        pass

    def lookupTable(self, tableName: str) -> TableDef:
        return next(t for t in self.tables if t.name == tableName)

    def resolve_auth(self, connection_name: AnyStr, connection_opts: Dict):
        # The adapter spec has an auth clause (self.auth) that can refer to "Connection options". 
        # The Connection options can refer to environment variables or hold direct values.
        # We need to:
        # 1. Resolve the env var references in the Connection options
        # 2. Copy the connection options into the REST API spec's auth clause
        for k, value in connection_opts.items():
            if value.startswith("$"):
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
                    if value in conn_opts:
                        auth_tree[key] = conn_opts[value]
                    elif "{" in  value:
                        # Resolve a string with connection opt references
                        auth_tree[key] = value.format(**conn_opts)
                    else:
                        print(f"Error: auth key {key} missing value in connection options")
        resolve_auth_values(self.auth, connection_opts)

    def __repr__(self) -> AnyStr:
        return f"{self.__class__.__name__}({self.name}) ->\n" + \
            ", ".join(map(lambda t: str(t), self.tables)) + "\n"

    def __str__(self) -> str:
        return self.name

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

    # Exporting data
    def create_output_table(self, file_name, output_logger:OutputLogger, overwrite=False, opts={}):
        raise RuntimeError(f"Adapter {self.name} does not support writing")

    def write_page(self, output_handle, page: pd.DataFrame, output_logger:OutputLogger, append=False):
        raise RuntimeError(f"Adapter {self.name} does not support writing")

    def close_output_table(self, output_handle):
        raise RuntimeError(f"Adapter {self.name} does not support writing")
        

class RESTAdapter(Adapter):
    def __init__(self, spec, storage: StorageManager=None):
        super().__init__(spec['name'], storage)
        self.base_url = spec['base_url']
        self.paging_options = spec.get('paging')
        self.auth = spec.get('auth', {}).copy()
        self.help = spec.get('help', None)

        self.dateParserFormat = spec.get('dateParser')
        self.query_date_format = spec.get('queryDateFormat')
        self.cacheTTLSecs = spec.get('cacheTTLSecs')
        self.polling_interval_secs = spec.get('polling_interval_secs', 60*60*4)
        self.next_page_token = spec.get('next_page_token')

        if "tables" in spec:
            self.tables = [RESTTable(self, d) for d in spec['tables']]
        else:
            print("Warning: spec '{}' has no tables defined".format(self.name))


    def list_tables(self) -> List[TableDef]:
        return self.tables

    def _setup_request_auth(self, session):
        user = ''
        token = ''
        session.headers.update({"Content-Type" : "application/json"})
        authType = self.auth['type']

        if authType in ['HEADERS', 'PARAMS']:
            dynValues = self.auth[authType.lower()]

        if authType == 'BASIC':
            user = self.auth['uservar']
            token = self.auth['tokenvar']
            session.auth = (user, token)
        elif authType == 'BEARER':
            token = os.environ.get(self.auth['tokenvar'], self.auth['tokenvar'])
            headers = {"Authorization": f"Bearer {token}"}
            session.headers.update(headers)
        elif authType == 'HEADERS':
            session.headers.update(dynValues)
        elif authType == 'PARAMS':
            session.params.update(dynValues)
        elif authType == 'NONE':
            pass
        else:
            raise Exception("Error unknown auth type")
