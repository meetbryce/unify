# python
import io
import os
import json
import logging
import re
import sys
import traceback
import urllib

# vendor
from lark import Lark
from lark.visitors import Visitor
from lark.tree import Tree

from google.oauth2.credentials import Credentials
from googleapiclient import discovery
import google.auth.transport.requests
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd

# project
from unify.adapters import Adapter, AdapterQueryResult, OutputLogger, StorageManager, TableDef
from unify.parsing_utils import collect_child_strings, find_node_return_child
from unify.schemata import LoadTableRequest

logger = logging.getLogger(__file__)

class GSheetsClient:
    DEFAULT_SCHEMA = "default"
    ALL_SPREADSHEETS_TABLE = "all_spreadsheets"
    MAPPED_SHEETS_TABLE = "mapped_sheets"

    SCOPES = [
        "https://www.googleapis.com/auth/drive", 
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/gmail.readonly"
    ]

    def __init__(self, spec):
        self.creds: Credentials = None

    def validate(self, adapter, silent=False):
        # FIXME: Implement a metadata store for user creds rathen than the
        # filesystem. Store creds per Connection so we can have multiple connections.
        needs_auth = True
        params = adapter.auth['params']
        creds_path = os.path.expanduser(params['client_creds_path'])
        logger.info(f"Loading google creds from path: {creds_path}")
        if os.path.exists(creds_path):
            cred_data = json.loads(open(creds_path).read())
            self.creds = Credentials(**cred_data)
            req = google.auth.transport.requests.Request()
            try:
                self.creds.refresh(req)
                needs_auth = False
            except google.auth.exceptions.RefreshError:
                pass #fall through to re-auth

        if needs_auth:
            if not self.stdin_available() or silent:
                raise RuntimeError("Warning, silent mode but Gsheets auth has lapsed")
            print("Press <enter> to re-authorize the GSheets connection")
            input()
            flow = InstalledAppFlow.from_client_secrets_file(
                os.path.expanduser(params['client_json_path']),
                self.SCOPES
            )
            flow.run_local_server()
            session = flow.authorized_session()
            with(open(creds_path, "w")) as f:
                f.write(session.credentials.to_json())
            self.creds = session.credentials

        self.DRIVE = discovery.build('drive', 'v3', credentials=self.creds)
        self.SHEETS = discovery.build('sheets', 'v4', credentials=self.creds)
        return True

    def stdin_available(self):
        try:
            if not os.isatty(sys.stdin.fileno()):
                return False
        except:
            # Pytest overrides stdin and throws exception on 'fileno'
            return False
        return True


    def get_all_sheets_files(self, name_match=None):
        page_token = None
        query = "mimeType='application/vnd.google-apps.spreadsheet'"
        if name_match:
            query += f" and name contains '{name_match}'"
        while True:
            args = {"q" :query,
                "fields": "nextPageToken,files(id,name,createdTime,modifiedTime,size,webViewLink)",
                "pageSize": 100,
                "orderBy" :"name"}
            if page_token:
                args.update({"pageToken": page_token})
            response = self.DRIVE.files().list(**args).execute()
            page_token = response.get('nextPageToken', None)
            for f in response['files']:
                yield(f)
            if not page_token:
                break


    def writeRows(self, table, records):
        print("***RECORDS: ", records)
        if table == GSheetsClient.ALL_SPREADSHEETS_TABLE:
            raise RuntimeError("cannot write to table")
        elif table == GSheetsClient.MAPPED_SHEETS_TABLE:
            # request to map a new spreadsheet
            for sheetinfo in records:
                if sheetinfo.get('spreadsheet_id'):
                    sheetId = sheetinfo.get('spreadsheet_id')
                    request = self.SHEETS.spreadsheets().get(
                        spreadsheetId=sheetId, 
                        ranges=[], 
                        includeGridData=False)
                    response = request.execute()
                    self.storeSheetInfo(response)
        else: #insert attempt into a table mapped to a spreadsheet
            sheetInfo = self.lookupTableInfo(table)
            columns = self.TABLE_STORE[f"{table}<schema>"]
            sheetsId = sheetInfo.get('spreadsheet_id')
            # retrieve the sheet's grid info to get the current data range so we can append after
            request = self.SHEETS.spreadsheets().get(
                spreadsheetId=sheetsId, 
                ranges=[], 
                includeGridData=False)
            response = request.execute()
            sheet = next(asheet for asheet in response['sheets'] \
                if asheet['properties']['title'] == sheetInfo['sheet_name'])

            range = f"{sheetInfo['sheet_name']}!1:1"
            # the only key is that we need to order input data in column order
            values = [] # array of arrays of row inputs to the spreadsheet
            for row in records:
                valrow = []
                for nameType in columns:
                    valrow.append(row.get(nameType[0]))
                values.append(valrow)
            value_range_body = {"range": range, "values": values}
            request = self.SHEETS.spreadsheets().values().append(
                spreadsheetId=sheetsId, 
                range=range,
                valueInputOption='RAW', 
                body=value_range_body)
            response = request.execute()
            print("Response: ", response)

        return len(records)

    def peek_sheet(self, sheetId, tab = None):
        range = "A1:Z10"
        if tab:
            range = tab + "!" + range
        response = self.SHEETS.spreadsheets().values().get(
            spreadsheetId=sheetId, 
            range=range, 
            #valueRenderOption="UNFORMATTED_VALUE"
        ).execute()
        return response.get('values', [])

    def create_new_sheet(self, newSheetName: str, overwrite: bool=False):
        for info in self.get_all_sheets_files(newSheetName):
            if info['name'] == newSheetName:
                if overwrite:
                    return info['id']
                else:
                    raise RuntimeError(f"Sheet {newSheetName} already exists")
        # No match so go ahead and create a new sheet
        spreadsheet_body = {"properties": {"title":newSheetName}}
        response = self.SHEETS.spreadsheets().create(body=spreadsheet_body).execute()
        return response['spreadsheetId']

    def write_data_to_sheet(self, sheetId=None, page: pd.DataFrame = None, append=False):
        # TODO: allow this to be called with multiple pages of data
        # Q: What if you want to write to a new tab of an existing sheet?

        # FIXME: handle different tabs
        range = f"1:1"
        # convert DataFrame to sheets data format
        value_range_body = {"range": range, "values": [list(page.columns)] + page.astype(str).values.tolist()}
        request = self.SHEETS.spreadsheets().values().append(
            spreadsheetId=sheetId,
            range=range,
            valueInputOption='RAW', 
            body=value_range_body)
        response = request.execute()
        if 'updates' in response and 'updatedRows' in response['updates']:
            return response['updates']['updatedRows']
        else:
            return None

    def storeSheetInfo(self, response, useTitle=None):
        info = {'title': response['properties']['title'],
                'url': response['spreadsheetUrl'],
                'spreadsheet_id': response['spreadsheetId'],
                'sheet_name': response['sheets'][0]['properties']['title']}
        info['table_name'] = useTitle or makeTableSafeName(info['title'])
        self.MAPPED.append(info)
        self.TABLE_STORE[response['spreadsheetId']] = info
        return info

    def getSheetInfo(self, sheetsId):
        info = self.SHEETS.spreadsheets().get(
            spreadsheetId=sheetsId
        ).execute()
        # Need the Drive API to get modified times
        meta = self.DRIVE.files().get(
            fileId=sheetsId,
            fields="id, name, mimeType, webContentLink, webViewLink, createdTime, modifiedTime"
        ).execute()
        info.update(meta)
        return info


    def storeColumnTuples(self, tableName, columnTuples):
        self.SCHEMA_CACHE[tableName] = columnTuples
        self.TABLE_STORE[f"{tableName}<schema>"] = columnTuples

    def lookupTableInfo(self, tableName):
        if tableName in [GSheetsClient.MAPPED_SHEETS_TABLE, GSheetsClient.ALL_SPREADSHEETS_TABLE]:
            return {"title": tableName}
        return next( table for table in self.MAPPED if table['table_name'] == tableName)

    def createTable(self, tableName, columnNames, columnTypes):
        # TODO: store column info in the sheet metadata
        spreadsheet_body = {"properties": {"title":f"Presto: {tableName}"}}
        response = self.SHEETS.spreadsheets().create(body=spreadsheet_body).execute()
        sheet_info = self.storeSheetInfo(response, tableName)

        columns = list(zip(columnNames, columnTypes))
        self.storeColumnTuples(tableName, columns)
        # Write header row in the sheet
        self.writeRows(tableName, [dict(zip(columnNames, columnNames))])

        return sheet_info['spreadsheet_id']


class GSheetsTableSpec(TableDef):
    # Represents a Google Sheet as a queryable Table spec to Unify.

    def __init__(self, sheetsClient: GSheetsClient, table: str, opts: dict):
        super().__init__(table)
        self.opts = opts # 'sheetId' and 'tab_name'
        self.sheetsClient: GSheetsClient = sheetsClient
    
    def query_resource(self, tableLoader, logger: logging.Logger):
        """ Generator which yields (page, size_return) tuples for all rows from
            a Google Sheet. Each row is returned as a dict mapping the column
            names to value.
        """
        spreadsheetId = self.opts['sheetId']
        sheet_name = self.opts['tab_name']
        pageSize = 1000
        lastRow = 0
        range = f"A{lastRow+1}:Z{lastRow+pageSize}"
        if sheet_name:
            range = sheet_name + "!" + range
        print("Downloading sheet range: ", range)
        response = self.sheetsClient.spreadsheets().values().get(
            spreadsheetId=spreadsheetId, 
            range=range, 
            valueRenderOption="UNFORMATTED_VALUE",
            dateTimeRenderOption="FORMATTED_STRING"
        ).execute()
        rows = response.get('values', [])
        print('{0} rows retrieved.'.format(len(rows)))
        results = []
        if len(rows) > 0:
            cols = rows[0]
            for row in rows[1:]:
                rowdict = dict(zip(cols, row))
                results.append(rowdict)
        size_return = []
        yield AdapterQueryResult(json=results, size_return=size_return)

class HideOurInstanceVars:
    pass

class GsheetCommandParser(Visitor):
    def __init__(self):
        self._safe = HideOurInstanceVars()
        self._safe.parser = Lark(open(
            os.path.join(os.path.dirname(__file__), "gsheets_grammar.lark")).read())

    def parse_and_run(self, code: str, output_logger: OutputLogger) -> str:
        self._safe.command = None
        self._safe.args = {}

        parse_tree = self._safe.parser.parse(code)
        self.visit(parse_tree)
        return self._safe.command

    def list_files(self, tree: Tree) -> Tree:
        self._safe.command = "list_files"
        return tree

    def search(self, tree: Tree) -> Tree:
        self._safe.command = "search"
        self._safe.args['search_query'] = collect_child_strings("search_query", tree).strip("'")
        return tree

    def info(self, tree: Tree) -> Tree:
        self._safe.command = "info"
        self._safe.args['file_or_gsheet_id'] = collect_child_strings("file_or_gsheet_id", tree).strip("'")
        return tree

    def peek(self, tree: Tree) -> Tree:
        self._safe.command = "peek"
        self._safe.args['file_or_gsheet_id'] = collect_child_strings("file_or_gsheet_id", tree).strip("'")
        tab = collect_child_strings("tab_name", tree)
        if tab:
            self._safe.args['tab_name'] = tab.strip("'")
        else:
            self._safe.args['tab_name'] = None
        return tree

    def create_table(self, tree: Tree) -> Tree:
        self._safe.command = "create_table"
        self._safe.args['table_name'] = find_node_return_child("table_name", tree)
        self._safe.args['file_or_gsheet_id'] = collect_child_strings("file_or_gsheet_id", tree).strip("'")
        tab = collect_child_strings("tab_name", tree)
        if tab:
            self._safe.args['tab_name'] = tab.strip("'")
        else:
            self._safe.args['tab_name'] = None
        return tree

    def create_view(self, tree: Tree) -> Tree:
        res = self.create_table(tree)
        self._safe.command = "create_view"
        return res
      

class GSheetsAdapter(Adapter):
    def __init__(self, spec, storage: StorageManager, schema_name: str):
        super().__init__(spec['name'], storage)
        self.auth = spec['auth'].copy()
        self.auth_spec = spec['auth']
        self.logger: OutputLogger = None

        self.parser: GsheetCommandParser = GsheetCommandParser()
        self.help = """
The GSheets connectors support reading and writing data from Google Sheets.
Try these commands:
  gsheets list files
  gsheets search '<file name>'
  gsheets info <file name> - list sheet names from a sheet  
  gsheets peek <sheets file> [<tab name>]  # peak at the first few rows in a sheet
  gsheets create table <name> from '<sheet title or id>' [tab '<tab name>']
  gsheets create view <name> from '<sheet title or id>' [tab '<tab name>']

'create table' will copy all data from the sheet into a local table. This is a good
option if the sheet is large and data is not changing frequently. By default the
data from the first tab will be imported unless a different tab name is specified.

Alternatively, 'create view' will create a view backed by the Google sheet, so that
queries will pull from the sheet dynamically each time. This incurs more latency but
is a good option if the sheet data is changing frequenly. 
        """
        self.client: GSheetsClient = GSheetsClient(spec)
        self.tables = []

    def validate(self, silent=False):
        return self.client.validate(self, silent)

    def get_config_parameters(self) -> dict:
        """ Returns a dict mapping config parameter names to descriptions. """
        return self.auth_spec.get('params', {})

    def list_tables(self):
        if not self.tables:
            self.tables = [
                GSheetsTableSpec(self.client.SHEETS, tup[0], tup[1]) \
                    for tup in self.storage.list_objects('tables')
            ]
        return self.tables

    def list_files(self, match_expr=None):
        return pd.DataFrame(
            [{"name": sheet['name'], "link": sheet['webViewLink']} \
                for sheet in self.client.get_all_sheets_files(match_expr)])

    def get_matching_sheets(self, search_query):
        return self.client.get_all_sheets_files(search_query)

    def search(self, search_query=None):
        for sheet in self.get_matching_sheets(search_query):
            self.logger.print(sheet['name'], "\t", sheet['webViewLink'])

    def print_sheet_header(self, sheetInfo, msg):
            t = sheetInfo['properties']['title']
            m = sheetInfo['modifiedTime']
            self.logger.print("Sheet '{}', last modified {}, {}: ".format(t, m, msg))

    def resolve_sheet_id(self, file_or_gsheet_id):
        if len(file_or_gsheet_id) > 15 and file_or_gsheet_id.find(" ") == -1:
            info = self.client.getSheetInfo(file_or_gsheet_id)
            return info['id']
        else:
            # Find the sheet by name but it has to be an exact match
            matches = list(self.get_matching_sheets(file_or_gsheet_id))
            if len(matches) == 1:
                return matches[0]['id']
            else:
                raise RuntimeError(f"Multiple sheets match name '{file_or_gsheet_id}'")

    def info(self, file_or_gsheet_id=None):
        if len(file_or_gsheet_id) > 15 and file_or_gsheet_id.find(" ") == -1:
            # assume a ghseet id
            infos = [self.client.getSheetInfo(file_or_gsheet_id)]
        else:
            infos = (self.client.getSheetInfo(sh['id']) \
                for sh in self.client.get_all_sheets_files(file_or_gsheet_id))
        for sheet in infos:
            self.print_sheet_header(sheet, "has tabs")
            for idx, tab in enumerate(sheet['sheets']):
                self.logger.print((idx+1), " - ", tab['properties']['title'])
            self.logger.print("\n")

    def peek(self, file_or_gsheet_id=None, tab_name=None):
        if len(file_or_gsheet_id) > 15 and file_or_gsheet_id.find(" ") == -1:
            sheetId = file_or_gsheet_id
        else:
            matches = list(self.client.get_all_sheets_files(file_or_gsheet_id))
            if len(matches) > 0:
                sheetId = matches[0]['id']
            else:
                raise RuntimeError("No matching spreadsheets found")
        info = self.client.getSheetInfo(sheetId)
        rows = self.client.peek_sheet(sheetId, tab_name)
        self.print_sheet_header(info, "peek")
        return pd.DataFrame(rows)

    def create_table(self, table_name, file_or_gsheet_id, tab_name):
        sheetId = self.resolve_sheet_id(file_or_gsheet_id)
        self.storage.put_object(
            'tables', 
            table_name,
            {'sheetId': sheetId, 'tab_name': tab_name}
        )
        self.tables = None
        # Todo: trigger a query which will load the table
        self.logger.print("Table created")

    def drop_table(self, table_root: str):
        self.storage.delete_object('tables', table_root)
        self.tables = None

    def rename_table(self, table_root: str, new_name: str):
        values = self.storage.get_object('tables', table_root)
        if values:
            self.storage.delete_object('tables', table_root)
            self.storage.put_object('tables', new_name, values)
        self.tables = None

    def supports_commands(self) -> bool:
        return True

    def run_command(self, code: str, output_logger: OutputLogger) -> None:
        # see if base wants to run it
        self.logger = output_logger
        res = super().run_command(code, output_logger)
        if res:
            return res
        else:
            self.last_code = code
            command = self.parser.parse_and_run(code, output_logger)
            if command:
                result = getattr(self, command)(**self.parser._safe.args)
                if isinstance(result, pd.DataFrame):
                    output_logger.print_df(result)
                return output_logger
            return None

    # Importing data
    def can_import_file(self, file_uri: str):
        return file_uri.lower().startswith("https://docs.google.com/spreadsheets")

    def import_file(self, file_uri: str, options: dict={}):
        # attempts to import a Google sheet into a new table.

        m = re.search(r"\/d\/([^\/]+)", file_uri)
        if m:
            sheetId = m.group(1)
            parts = urllib.parse.urlparse(file_uri)
            tab_id = None
            if parts.fragment and re.search(r"gid=\d+", parts.fragment):
                tab_id = int(re.search(r"gid=(\d+)", parts.fragment).group(1))

            # Will fail if we don't have access to this sheet
            info = self.client.getSheetInfo(sheetId)
            title = info['properties']['title']
            tab_name = None
            for tab in info['sheets']:
                if tab_id is None or tab['properties']['sheetId'] == tab_id:
                    tab_name = tab['properties']['title']
                    break
            if tab_id is not None and tab_id > 0:
                title += "_" + tab_name # qualify table name for tabs other than the first
            table_name = self.convert_string_to_table_name(title)
            self.create_table(table_name, sheetId, tab_name)
            return table_name

    # Exporting data
    def create_output_table(self, file_name, output_logger: OutputLogger, overwrite=False, opts={}):
        # FIXME: handle tabs
        return self.client.create_new_sheet(file_name, overwrite=overwrite) # returns the sheetId

    def write_page(self, output_handle, page: pd.DataFrame, output_logger: OutputLogger, append=False, page_num=1):
        sheetId = output_handle
        updated_rows = self.client.write_data_to_sheet(sheetId=sheetId, page=page, append=append)
        if updated_rows:
            output_logger.print(f"Wrote {updated_rows} rows to sheet")

    def close_output_table(self, output_handle):
        # Sheets REST API is stateless
        pass

