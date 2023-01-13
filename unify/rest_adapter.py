from cgi import parse_multipart
import copy
import json
import logging
import logging.handlers
import os
import re
import time
from pprint import pprint
import string
from tempfile import NamedTemporaryFile
from typing import List, AnyStr, Dict, Union, Iterable, Generator
import typing
from datetime import datetime
from collections.abc import Iterable
from collections import namedtuple

import requests
from jsonpath_ng import parse

from google.oauth2.credentials import Credentials
import google.auth.transport.requests

from .storage_manager import StorageManager
from .data_utils import interp_dollar_values, flatten_dict
from .adapters import (
    Adapter, 
    AdapterQueryResult,
    TableDef, 
    TableUpdater, 
    ReloadStrategy, 
    RESTView,
    UpdatesStrategy, 
)

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
            # FIXME: We should return True as long as a next token came back. This allows
            # us to get short pages without stopping.
            return self.current_token is not None
        return False


class RESTTable(TableDef):
    VALID_KEYS = {
        'name': str,
        'description': str,
        'resource_path': str,
        'result_body_path' : (str, list),
        'result_object_path': (str, list),
        'result_meta_paths': list,
        'supports_paging': bool,
        'paging': dict,
        'headers': dict,
        'params': dict,
        'post': dict,
        'select': str,
        'copy_params_to_output': list,
        'key_columns': (list, str),
        'refresh': dict,
        'strip_prefixes': (list, str),
        # deprecated
        'query_resource': str,
        'columns': list,
        'key_column': str
    }

    def __init__(self, spec, dictvals):
        super().__init__(dictvals['name'], dictvals.get('description'))
        fmt = string.Formatter()
        self.max_pages = 50000 # TODO: Alow adapter to override this

        self.validate_dict(
            "Table definition", 
            spec.name + "." + self.name, 
            dictvals, 
            valid_keys=RESTTable.VALID_KEYS
        )

        self._supports_paging = dictvals.get('supports_paging', False)
        if self._supports_paging:
            if 'paging' in dictvals:
                # Allow per-table paging options
                self.paging_options = dictvals['paging']
            else:
                self.paging_options = spec.paging_options
        else:
            self.paging_options = None

        self.spec = spec
        self.key = dictvals.get('key_column')
        self.name = dictvals['name']
        self.query_path = dictvals.get('resource_path', '')
        self.query_method = dictvals.get('method', 'GET')
        self.query_date_format = spec.query_date_format
        self.strip_prefixes = dictvals.get('strip_prefixes')
        if isinstance(self.strip_prefixes, str):
            self.strip_prefixes = [self.strip_prefixes]

        self.headers = dictvals.get('headers', {})
        # List of parameters that should be merged into the output table
        self.copy_params_to_output = dictvals.get('copy_params_to_output')
        if self.copy_params_to_output and not isinstance(self.copy_params_to_output, list):
            self.copy_params_to_output = [self.copy_params_to_output]

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

        self.create_method, self.create_path = self.parse_resource(dictvals.get('create_resource'))
        self.create_args = [t[1] for t in fmt.parse(self.create_path or '') if t[1] is not None]
        self.update_method, self.update_path = self.parse_resource(dictvals.get('update_resource'))
        self.update_args = [t[1] for t in fmt.parse(self.update_path or '') if t[1] is not None]
        self.delete_method, self.delete_path = self.parse_resource(dictvals.get('delete_resource'))
        self.delete_args = [t[1] for t in fmt.parse(self.delete_path or '') if t[1] is not None]

        if (self.query_path == '<inline>'):
            self.static_values = dictvals.get('values')
        self.result_body_path = dictvals.get('result_body_path')
        self.result_object_path = dictvals.get('result_object_path')
        self.result_meta_paths = dictvals.get('result_meta_paths')

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
        self.post = dictvals.get('post')
        self.keyColumnName = self.keyColumnType = None       

    def validate_dict(self, object_type: str, object_name: str, opts: dict, valid_keys: dict):
        for key in opts.keys():
            if key not in valid_keys:
                raise RuntimeError(f"Invalid key '{key}' for {object_type} - {object_name}")
            else:
                val_type = valid_keys[key]
                if not isinstance(opts[key], val_type):
                    raise RuntimeError(f"Invalid type for key '{key}' for {object_type} - {object_name}, expected: {val_type}")

    def get_sql_query_param(self, key, value):
        if not isinstance(value, str):
            return None
        m = re.match(r"sql@\((.*)\)", value)
        if m:
            return m.group(1)
        else:
            return None

    def get_table_source(self):
        # Returns a description of where we are loading data from for the table
        return {"api": (self.query_method + " " + self.query_path), "adapter": self.spec.name}

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

    def has_multivalue_parameters(self):
        """ returns true if this table has multi-valued parameters (which means that this API resource requires
            multiple API calls to satisfy. 
        """
        for pname, value in self.params.items():
            sql_query = self.get_sql_query_param(pname, value)
            if sql_query:
                return True
            if not isinstance(value, str) and isinstance(value, Iterable):
                return True
        return False

    def query_resource(self, tableLoader, logger: logging.Logger) -> Generator[AdapterQueryResult, None, None]:
        for params_record in self.generate_param_values(tableLoader):
            if self.copy_params_to_output:
                merge_cols = {k: params_record.get(k) for k in self.copy_params_to_output}
            else:
                merge_cols = None
            for page, size_return in self._query_resource(tableLoader, params_record.copy(), logger):
                yield AdapterQueryResult(json=page, size_return=size_return, merge_cols=merge_cols)
                if self.spec.throttle:
                    time.sleep(self.spec.throttle['sleep'])

    def generate_param_values(self, tableLoader):
        """ 
            Generates multiple param values for the API call. If all parameters are scalar values
            then this function just yields a single parameters dict. But if we find parameters that 
            generate multiple values, either via an embedded SQL query, or via a literal list of values,
            then we generate a parameter dict for each value.

            It is possible to have multiple "multi-value" parameters. In this case they will be expanded
            hierarchically via their order of declaration in the spec:
            
            params:
              arg1: [1, 2, 3]
              arg2: ["house", "car", "elevator"]

            This will generate API calls as a union of both lists:

              GET url?arg1=1&arg2=house
              GET url?arg1=1&arg2=car
              GET url?arg1=1&arg2=elevator
              GET url?arg1=2&arg2=house
              GET url?arg1=2&arg2=car
              GET url?arg1=2&arg2=elevator
              GET url?arg1=3&arg2=house
              GET url?arg1=3&arg2=car
              GET url?arg1=3&arg2=elevator
        """
        # Setup parameters. Copy the literal ones.
        params = self.params.copy()

        # Now expand any SQL query references. A query could return multiple results or a
        # single one.

        for pname, value in self.params.items():
            sql_query = self.get_sql_query_param(pname, value)
            if sql_query:
                for df in tableLoader.query_table(self.spec.name, sql_query):
                    if df.shape[0] == 0:
                        # no values, so remove the parameter
                        del params[pname]
                    elif df.shape[0] == 1:
                        params[pname] = df.to_numpy()[0].astype('str')[0]
                    else:
                        params[pname] = df.to_numpy().astype('str')

        # Now run through parameters and if we find one with multiple values we generate
        # param dictionaries with each value


        # Recursively iterate through each parameter in the list. If we encounter
        # a multi-value then we generate a row for each value. Otherwisewe just generate
        # for the single scalar vlaue. We continue until we our out of params.
        # This works even for multiple multi-value parameters.

        def emit_params(accum: dict, params: list, pos:int = 0):
            if pos >= len(params):
                yield accum
            else:
                pname, value = params[pos]
                if not isinstance(value, str) and isinstance(value, Iterable):
                    param_cols = re.split(r"\s*,\s*", pname)
                    for row in value:
                        if len(param_cols) == 1:
                            accum[pname] = (row[0] if hasattr(row, 'size') else row)
                        else:
                            # Unpack multiple param values
                            accum.update(dict(zip(param_cols, row)))
                        yield from emit_params(accum.copy(), params, pos+1)
                else:
                    accum[pname] = value
                    yield from emit_params(accum, params, pos+1)

        yield from emit_params({}, list(params.items()))

                    
    def _query_resource(
        self, 
        tableLoader,
        orig_api_params={},
        logger: logging.Logger= None):
        """
            Call a REST API given the provide api_params. Also handles pagination of the API.
        """
        session = requests.Session()
        self.spec._setup_request_auth(session)
        session.headers.update(self.headers)

        pager = PagingHelper.get_pager(self.paging_options)
        
        page = 1
        safety_max_pages = 200000 # prevent infinite loop in case "pages finished logic" fails

        while page < safety_max_pages:
            api_params = orig_api_params.copy() # Fresh copy each page

            try:
                url = (self.spec.base_url + self.query_path).format(**api_params)
            except KeyError as e:
                url = self.spec.base_url + self.query_path
                raise RuntimeError(f"Cannot query API resource '{url}' because missing API param from: {api_params}. {e}")

            api_params.update(pager.get_request_params())
            
            if self.post:
                remove_keys = []
                post = self.interpolate_post_values(self.post, api_params, remove_keys, logger)
                for k in remove_keys:
                    api_params.pop(k, None)
                logger.debug("POST %s %s", url, api_params, post)
                r = session.post(url, params=api_params, json=post)
            else:
                if len(api_params.keys()) > 4:
                    show_params = {k:api_params[k]for k in list(api_params.keys())[0:5]}
                else:
                    show_params = api_params
                logger.debug(url + " " + str(api_params))
                r = session.get(url, params=api_params)

            if r.status_code >= 400:
                print(r.text)
            if r.status_code == 404:
                logger.error(f"%s 404 returned from {url}", self.name)
                return
            if r.status_code == 401:
                logger.error(f"%s 401 returned from {url}", self.name)
                raise RuntimeError("API call unauthorized: " + r.text)
            if r.status_code >= 400:
                logger.error(f"%s HTTP error {r.status_code} returned from {url}")
                return

            size_return = []

            json_result = r.json()
            #pprint(json_result)

            yield (json_result, size_return)

            if len(size_return) == 0 or not pager.next_page(size_return[0], json_result):
                break

            page += 1
            if page > self.max_pages:
                logger.warning("Aborting table scan after %d pages", page-1)
                break

    def interpolate_post_values(self, node, params: dict, remove_keys: list, logger: logging.Logger):
        """ Substitutes parameters into a request POST body by finding
            ${var} references. Returns a copy of the POST body with the substituted
            values.
        """
        if isinstance(node, str):
            for k, v in params.items():
                pattern = "${" + k + "}"
                if node == pattern:
                    node = v # allow to substitute the whole value
                    remove_keys.append(k)
                elif pattern in node:
                    node = node.replace(pattern, str(v))
                    remove_keys.append(k)
            if isinstance(node, str) and re.match(r"\$\{[a-zA-Z0-9_-]+\}", node):
                # remove key references not supplied in params
                logger.debug("Removing undefined reference: %s", node)
                node = None
            return node
        elif isinstance(node, list):
            return [self.interpolate_post_values(v, params, remove_keys, logger) for v in node]
        elif isinstance(node, dict):
            dup = node.copy()
            for key, value in node.items():
                dup[key] = self.interpolate_post_values(value, params, remove_keys, logger)
            return dup
        else:
            # Non-strings just return as is
            return node  

    
class RESTAdapter(Adapter):
    def __init__(self, spec, storage: StorageManager, schema_name: str):
        super().__init__(spec['name'], storage)
        self.base_url = spec['base_url']
        self.paging_options = spec.get('paging')
        self.auth = copy.deepcopy(spec.get('auth', {}))
        self.auth_spec = spec.get('auth', {})
        self.help = spec.get('help', None)

        self.dateParserFormat = spec.get('dateParser')
        self.query_date_format = spec.get('queryDateFormat')
        self.cacheTTLSecs = spec.get('cacheTTLSecs')
        self.polling_interval_secs = spec.get('polling_interval_secs', 60*60*4)
        self.next_page_token = spec.get('next_page_token')
        self.throttle = spec.get('throttle')

        if "tables" in spec:
            self.tables = [RESTTable(self, d) for d in spec['tables']]
        else:
            print("Warning: spec '{}' has no tables defined".format(self.name))

        # Extract auth parameter descriptions
        self.auth_help = self.auth.pop("help", {})

        self.views = []
        if "views" in spec:
            for d in spec['views']:
                if 'name' in d and 'from' in d and 'query' in d:
                    self.views.append(
                        RESTView(name=d['name'], 
                            from_list=d['from'], 
                            query=d['query'],
                            help=d.get('help')
                        )
                    )
                else:
                    raise RuntimeError(f"Missing one of name, from or query from view: {d}")                   


    def get_config_parameters(self) -> dict:
        """ Returns a dict mapping config parameter names to descriptions. """
        return self.auth_spec.get('params', {})

    def list_tables(self) -> List[TableDef]:
        return self.tables

    def list_views(self) -> List[RESTView]:
        return self.views

    def _setup_request_auth(self, session: requests.Session):
        user = ''
        token = ''
        params = self.auth.get('params', {})
        session.headers.update({"Content-Type" : "application/json"})
        authType = self.auth['type']

        if authType in ['HEADERS', 'PARAMS']:
            dynValues = self.auth[authType.lower()]

        if authType == 'BASIC':
            user = params.get('username')
            token = params.get('password')
            session.auth = (user, token)
        elif authType == 'BEARER':
            token = os.environ.get(params['bearer_token'], params['bearer_token'])
            headers = {"Authorization": f"Bearer {token}"}
            session.headers.update(headers)
        elif authType == 'HEADERS':
            session.headers.update(dynValues)
        elif authType == 'PARAMS':
            session.params.update(dynValues)
        elif authType == 'AWS4Auth':
            from requests_aws4auth import AWS4Auth
            session.auth = AWS4Auth(
                params.get('access_key_id'),
                params.get('secret_key'),
                params.get('region'),
                self.auth['service']
            )
        elif authType == 'GOOGLE_AUTH':
            creds_path = os.path.expanduser(self.auth['client_creds_path'])
            if os.path.exists(creds_path):
                cred_data = json.loads(open(creds_path).read())
                self.creds = Credentials(**cred_data)
                req: google.auth.transport.requests.Request = google.auth.transport.requests.Request()
                try:
                    self.creds.refresh(req)
                    self.creds.apply(session.headers)
                except google.auth.exceptions.RefreshError:
                    raise
        elif authType == 'NONE':
            pass
        else:
            raise Exception(f"Error unknown auth type: {authType}")

    def validate(self) -> bool:
        # Make sure all auth parameters have been supplied
        auth_params = self.auth.get('params')
        if auth_params is None:
            return True
        for k, v in auth_params.items():
            if self.auth_spec['params'][k] == v:
                raise RuntimeError(f"Cannot validate auth for {self.name} adapter, missing required auth parameter: {k}")
        return True

    def test_connection(self, logger: logging.Logger):
        # Used to verify valid auth for a new connection
        class DetectErrorHandler(logging.NullHandler):
            def __init__(self):
                super().__init__()
                self.saw_error = False
                self.saw_msg = None

            def handle(self, record):
                if record.levelno >= logging.ERROR:
                    self.saw_error = True
                    self.saw_msg = record.msg
                super().handle(record)

        for table_spec in self.tables:
            if not table_spec.has_multivalue_parameters():
                detector = DetectErrorHandler()
                logger.addHandler(detector)
                for page in table_spec.query_resource(tableLoader=None, logger=logger):
                    logger.removeHandler(detector)
                    if detector.saw_error:
                        raise RuntimeError(detector.saw_msg)
                    return True #only need to test one request

        # TODO: fallback to use a multi-paramed request
        return True
    