import glob
import io
import os
import re
import string
import sys
import yaml
from typing import List, AnyStr, Dict
import typing

import requests
from storage_manager import StorageManager
import pandas as pd

Adapter = typing.NewType("Adapter", None)

class Connection:
    def __init__(self, adapter, schema_name, opts):
        self.schema_name: str = schema_name
        self.adapter: Adapter = adapter
        self.adapter.resolve_auth(self.schema_name, opts['options'])
        self.adapter.validate()

    @classmethod
    def setup_connections(cls, path=None, conn_list=None, storage_mgr_maker=None):
        adapter_table = {}
        for f in glob.glob(f"./rest_specs/*spec.yaml"):
            spec = yaml.load(open(f), Loader=yaml.FullLoader)
            if spec.get('enabled') == False:
                continue
            klass = RESTAdapter
            if 'class' in spec and spec['class'].lower() == 'gsheetsadapter':
                from gsheets.gsheets_adapter import GSheetsAdapter
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
            result.append(c)
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
        else:
            raise RuntimeError("Unknown paging strategy: ", options)

class NullPager(PagingHelper):
    def __init__(self, options):
        super().__init__(options)

    def get_request_params(self):
        return {}

    def next_page(self, page_count):
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

    def next_page(self, page_count):
        self.current_page += 1
        return page_count >= self.page_size

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

    def next_page(self, page_count):
        self.current_offset += page_count
        return page_count >= self.page_size

class TableDef:
    def __init__(self, name):
        self._name = name
        self._select_list = []
        self._result_body_path = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def select_list(self) -> list:
        return []

    @select_list.setter
    def select_list(self, selects: list):
        self._select_list = selects

    def query_resource(self, tableLoader):
        pass

    @property
    def result_body_path(self):
        return self._result_body_path

    @result_body_path.setter
    def result_body_path(self, val):
        self._result_body_path = val


class RESTTable(TableDef):
    def __init__(self, spec, dictvals):
        fmt = string.Formatter()
        self.max_pages = 50000

        self._supports_paging = dictvals.get('supports_paging', False)
        if self._supports_paging:
            self.paging_options = spec.paging_options
        else:
            self.paging_options = None

        self.spec = spec
        self.name = dictvals['name']
        self.query_path = dictvals.get('resource_path', '')
        self.query_method = dictvals.get('method', 'GET')
        if dictvals.get('query_resource'):
            self.query_method, self.query_path = self.parse_resource(dictvals.get('query_resource'))
        self.query_args = [t[1] for t in fmt.parse(self.query_path) if t[1] is not None]        

        self.select_list = dictvals.get('select')
        if self.select_list:
            self.select_list = re.split(r",\s*", self.select_list)

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

        # if dictvals.get('json_response'):
        #     self.columns.extend(self.parse_json_for_columns(json.loads(dictvals['json_response'])))
        #     if self.result_body_path is None:
        #         self.result_type = 'object'
        # else:
        #     self.columns.extend([RESTCol(d) for d in dictvals.get('columns', [])])

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

    def query_resource(self, tableLoader):
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
                                print("Excluding parent row with key: ", record[key])
                                do_row = False
                        else:
                            print(f"Error: parent record missing col {parent_col}: ", record)
                    if do_row:
                        for page, size_return in self._query_resource(record):
                            yield (page, size_return)
        else:
            # Simple query
            for page, size_return in self._query_resource():
                yield (page, size_return)

    def _query_resource(self, query_params={}):
        session = requests.Session()
        self.spec._setup_request_auth(session)

        pager = PagingHelper.get_pager(self.paging_options)
        params = self.params.copy()
        page = 1

        while True:
            url = (self.spec.base_url + self.query_path).format(**query_params)
            params.update(pager.get_request_params())
            print(url, params)
            r = session.get(url, params=params)

            if r.status_code >= 400:
                print(r.text)
            if r.status_code == 404:
                raise Exception("Query returned no results")
            if r.status_code == 401:
                print("Auth failed. Spec auth is: ", self.spec.auth)
                raise Exception("API call unauthorized: " + r.text)
            if r.status_code >= 400:
                raise Exception("API call failed: " + r.text)

            size_return = []

            yield (r.json(), size_return)

            if not pager.next_page(size_return[0]):
                break

            page += 1
            if page > self.max_pages:
                print("Aborting table scan after {} pages", page-1)
                break


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
                    else:
                        print(f"Error: auth key {key} missing value in connection options")
        resolve_auth_values(self.auth, connection_opts)

    def __repr__(self) -> AnyStr:
        return f"{self.__class__.__name__}({self.name}) ->\n" + \
            ", ".join(map(lambda t: str(t), self.tables)) + "\n"

    def supports_commands(self) -> bool:
        return self.help is not None

    def run_command(self, code: str, output_buffer: io.TextIOBase) -> bool:
        if code.strip() == 'help':
            if self.help:
                print(self.help, file=output_buffer)
            else:
                print(f"No help available for connector {self.name}")
            return True
        else:
            return False

    # Exporting data
    def create_output_table(self, file_name, opts={}):
        raise RuntimeError(f"Adapter {self.name} does not support writing")

    def write_page(self, output_handle, page: pd.DataFrame):
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

        # Pages are indicated by absolute row offset
        self.pageStartArg = spec.get('pageStartArg')
        # Pages are indicated by "page number"
        self.pageIndexArg = spec.get('pageIndexArg')
        # Pages are indicated by providing the "next cursor" from the previous call
        self.page_cursor_arg = spec.get('page_cursor_arg')
        self.pageMaxArg = spec.get('pageMaxArg')
        self.dateParserFormat = spec.get('dateParser')
        self.queryDateFormat = spec.get('queryDateFormat')
        self.maxPerPage  = spec.get('maxPerPage', 100)
        self.cacheTTLSecs = spec.get('cacheTTLSecs')
        self.polling_interval_secs = spec.get('polling_interval_secs', 60*60*4)
        self.next_page_token = spec.get('next_page_token')

        if "tables" in spec:
            self.tables = [RESTTable(self, d) for d in spec['tables']]
        else:
            print("Warning: spec '{}' has no tables defined".format(self.name))


    def list_tables(self) -> List[TableDef]:
        return self.tables

    def supports_paging(self):
        return (self.pageIndexArg or self.pageStartArg or self.page_cursor_arg)

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

