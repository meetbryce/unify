import glob
import json
import os
import re
import string
import sys
import yaml

import requests

Columns = list[str]

class Connector:
    def __init__(self, spec_table, opts):
        self.schema_name = next(iter(opts))
        opts = opts[self.schema_name]
        spec_name = opts['spec']
        self.spec = spec_table[spec_name] 
        self.spec.resolve_auth(self.schema_name, opts['options'])

    @classmethod
    def setup_connections(cls, path=None, conn_list=None):
        spec_table = {}
        for f in glob.glob(f"./rest_specs/*spec.yaml"):
            spec = yaml.load(open(f), Loader=yaml.FullLoader)
            klass = RESTAPISpec
            if 'class' in spec and spec['class'].lower() == 'gsheetsconnector':
                from gsheets.gsheets_connector import GSheetsConnector
                klass = GSheetsConnector
            spec_inst = klass(spec)
            spec_table[spec_inst.name] = spec_inst
        
        if conn_list:
            connections = conn_list
        else:
            connections = yaml.load(open(path), Loader=yaml.FullLoader)
        return [Connector(spec_table, opts) for opts in connections]


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


class RESTTable:
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
                            breakpoint()
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
            #with open(f"page-{page}.json", "w") as f:
            #    f.write(r.text)

            yield (r.json(), size_return)

            if not pager.next_page(size_return[0]):
                break

            page += 1
            if page > self.max_pages:
                print("Aborting table scan after {} pages", page-1)
                break

class RESTAPISpec(Connector):
    def __init__(self, spec):
        self.name = spec['name']
        self.base_url = spec['base_url']
        self.paging_options = spec.get('paging')
        self.auth = spec.get('auth', {})

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


    def list_tables(self):
        for t in self.tables:
            yield t

    def lookupTable(self, tableName):
        return next(t for t in self.tables if t.name == tableName)

    def supports_paging(self):
        return (self.pageIndexArg or self.pageStartArg or self.page_cursor_arg)

    def __repr__(self):
        return f"RESTAPISpect({self.name}) ->\n" + \
            ", ".join(map(lambda t: str(t), self.tables)) + "\n"

    def resolve_auth(self, connection_name, opts):
        # opts will be a dictionary of either env var references starting with '$'
        # or else actual values
        for key, var_name in self.auth.items():
            if key == 'type':
                continue
            if var_name in opts:
                value = opts[var_name]
                if value.startswith("$"):
                    try:
                        value = os.environ[value[1:]]
                    except KeyError:
                        print(f"Authentication for {connection_name} failed, missing env var '{value[1:]}'")
                        sys.exit(1)
                self.auth[key] = value 

    def _setup_request_auth(self, session):
        user = ''
        token = ''
        session.headers.update({"Content-Type" : "application/json"})
        authType = self.auth['type']

        if authType in ['HEADERS', 'PARAMS']:
            entries = self.auth[authType.lower()]
            dynValues = {k: os.environ.get(v, v) for k, v in entries.items()}

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

