# python
import io
import os
import json
import sys
import traceback

# vendor
from lark import Lark
from lark.visitors import Visitor
from lark.tree import Tree

from google.oauth2.credentials import Credentials
from googleapiclient import discovery
import google.auth.transport.requests
from google_auth_oauthlib.flow import InstalledAppFlow

# project
from rest_schema import Adapter, StorageManager, TableDef
from parsing_utils import collect_child_strings, find_node_return_child
from schemata import LoadTableRequest

class GSheetsClient:
    DEFAULT_SCHEMA = "default"
    ALL_SPREADSHEETS_TABLE = "all_spreadsheets"
    MAPPED_SHEETS_TABLE = "mapped_sheets"

    SCOPES = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]

    def __init__(self, spec):
        self.creds: Credentials = None

    def validate(self, adapter):
        # FIXME: Implement a metadata store for user creds rathen than the
        # filesystem. Store creds per Connection so we can have multiple connections.
        needs_auth = True
        creds_path = os.path.join(os.path.dirname(__file__), 'user_creds.json')
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
            if not self.stdin_available():
                print("Warning, no TTY to request gsheets validation")
                return False
            print("Press <enter> to re-authorize the GSheets connection")
            input()
            flow = InstalledAppFlow.from_client_secrets_file(
                adapter.auth['client_json_path'],
                self.SCOPES
            )
            flow.run_local_server()
            session = flow.authorized_session()
            with(open(creds_path, "w")) as f:
                f.write(session.credentials.to_json())
            self.creds = session.credentials

        self.DRIVE = discovery.build('drive', 'v3', credentials=self.creds)
        self.SHEETS = discovery.build('sheets', 'v4', credentials=self.creds)
        print(self.SHEETS.spreadsheets().values().get(spreadsheetId='16YgB5XykiMBMXQfHk8lI9hYilJ2ctn6madVAJoKt12Q',range="A1:A2").execute())
        return True

    def stdin_available(self):
        try:
            if not os.isatty(sys.stdin.fileno()):
                return False
        except:
            # Pytest overrides stdin and throws exception on 'fileno'
            return False
        return True

    def _loadMappedTables(self):
        self.MAPPED.clear()
        self.SCHEMA_CACHE.clear()
        for key in self.TABLE_STORE.keys():
            if key.endswith("<schema>"):
                self.SCHEMA_CACHE[key[0:-8]] = self.TABLE_STORE[key]
            else:
                self.MAPPED.append(self.TABLE_STORE[key])

    def listTables(self):
        tables = [row['table_name'] for row in self.MAPPED]
        tables.extend(
            [GSheetsClient.ALL_SPREADSHEETS_TABLE, GSheetsClient.MAPPED_SHEETS_TABLE]
        )
        return tables

    def columnList(self, table):
        if table == GSheetsClient.ALL_SPREADSHEETS_TABLE:
            return [("title", "VARCHAR"), ("spreadsheet_id", "VARCHAR"), ("url", "VARCHAR")]
        elif table == GSheetsClient.MAPPED_SHEETS_TABLE:
            return [("title", "VARCHAR"), ("table_name", "VARCHAR"), 
                    ("spreadsheet_id", "VARCHAR"), ("sheet_name", "VARCHAR"),
                    ("url", "VARCHAR")]
        else:
            if table in self.SCHEMA_CACHE:
                return self.SCHEMA_CACHE[table]

            try:
                match = next( t for t in self.MAPPED if t['table_name'] == table)
                columns = self._downloadSchema(match['spreadsheet_id'], match['sheet_name'])

                self.storeColumnTuples(table, columns)
                return columns

            except StopIteration:
                raise RuntimeError(f"Unknown table '{table}'")
            except:
                traceback.print_exc()
                raise RuntimeError(f"Error downloading schema {table}")


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

    def getTableRows(self, table, lastRow):
        if table == GSheetsClient.ALL_SPREADSHEETS_TABLE:
            files = self.DRIVE.files().list(
                q="mimeType='application/vnd.google-apps.spreadsheet'",
                fields="files(id, name, mimeType, webContentLink, webViewLink)",
                orderBy="name"
            ).execute().get('files', [])

            col_map = {'title': 'name', 'spreadsheet_id': 'id', 'url': 'webViewLink'}
            return files, col_map, None
        elif table == GSheetsClient.MAPPED_SHEETS_TABLE:
            return self.MAPPED, None, None
        else:
            rows, lastRow = self._downloadSheetData(table, lastRow)
            return rows, None, lastRow


    def _downloadSheetData(self, tableName, lastRow = 0):
        print(self.SCHEMA_CACHE)
        colPairs = self.SCHEMA_CACHE[tableName]
        spec = self.lookupTableInfo(tableName)
        spreadsheetId = spec['spreadsheet_id']
        sheet_name = spec['sheet_name']
        pageSize = 1000
        if lastRow is None:
            lastRow = 0
        range = f"{sheet_name}!A{lastRow+1}:Z{lastRow+pageSize}"
        print("Downloading sheet range: ", range)
        response = self.SHEETS.spreadsheets().values().get(
            spreadsheetId=spreadsheetId, range=range).execute()
        rows = response.get('values', [])
        print('{0} rows retrieved.'.format(len(rows)))
        results = []
        if len(rows) > 0:
            for row in rows[1:]:
                rowdict = {}
                for idx, c in enumerate(colPairs):
                    try:
                        rowdict[c[0]] = row[idx]
                    except IndexError:
                        rowdict[c[0]] = ''
                results.append(rowdict)
        if len(rows) >= pageSize:
            lastRow = lastRow + len(results)
            print("Returning nextRow ", lastRow)
        else:
            lastRow = None
        return results, lastRow

    def _downloadSchema(self, spreadsheetId, sheet_name):
        response = self.SHEETS.spreadsheets().values().get(
            spreadsheetId=spreadsheetId, range=f"{sheet_name}!A1:Z2").execute()
        rows = response.get('values', [])
        print('{0} rows retrieved.'.format(len(rows)))
        columns = []
        if len(rows) > 0:
            for cell in rows[0]:
                columns.append((makeTableSafeName(cell), 'VARCHAR'))
        return columns

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

    def getTypeMap(self, tableName):
        # Returns a dict mapping column names to types
        if tableName in [GSheetsClient.ALL_SPREADSHEETS_TABLE, GSheetsClient.MAPPED_SHEETS_TABLE]:
            return {}
        elif tableName in self.SCHEMA_CACHE:
            return dict(self.SCHEMA_CACHE[tableName])
        else:
            raise RuntimeError(f"No schema for {tableName}")

    def removeTableRecord(self, tableName):
        if tableName in [GSheetsClient.ALL_SPREADSHEETS_TABLE, GSheetsClient.MAPPED_SHEETS_TABLE]:
            raise RuntimeError("Cannot drop system tables")
        else:
            for i in range(len(self.MAPPED)):
                if self.MAPPED[i]['table_name'] == tableName:
                    del self.MAPPED[i]
                    try:
                        del self.SCHEMA_CACHE[tableName]
                    except KeyError:
                        pass
                    try:
                        del self.TABLE_STORE[f"{tableName}<schema>"]
                    except KeyError:
                        pass
                    return

class GSheetsTableSpec(TableDef):
    def __init__(self, sheetsClient, table: str, opts: dict):
        super().__init__(table)
        self.opts = opts # 'sheetId' and 'tab_name'
        self.sheetsClient = sheetsClient
    
    def query_resource(self, tableLoader):
        """ Generator which yields (page, size_return) tuples for all rows from
            an API endpoint. Each page should be a list of dicts representing
            each row of results.
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
            spreadsheetId=spreadsheetId, range=range, valueRenderOption="UNFORMATTED_VALUE").execute()
        rows = response.get('values', [])
        print('{0} rows retrieved.'.format(len(rows)))
        results = []
        if len(rows) > 0:
            cols = rows[0]
            for row in rows[1:]:
                rowdict = dict(zip(cols, row))
                results.append(rowdict)
        size_return = []
        yield (results, size_return)


class GSheetsAdapter(Adapter):
    def __init__(self, spec, storage: StorageManager):
        super().__init__(spec['name'], storage)
        self.auth = spec['auth'].copy()
        self.output = sys.stdout

        self.parser: GsheetCommandParser = GsheetCommandParser()
        self.help = """
The GSheets connectors support reading and writing data from Google Sheets.
Try these commands:
  gsheets list files
  gsheets search '<file name>'
  gsheets info <file name> - list sheet names from a sheet  
  gsheets create table <name> from file '<sheet title or id>' [tab '<tab name>']
  gsheets create view <name> from file '<sheet title or id>' [tab '<tab name>']

'create table' will copy all data from the sheet into a local table. This is a good
option if the sheet is large and data is not changing frequently. By default the
data from the first tab will be imported unless a different tab name is specified.

Alternatively, 'create view' will create a view backed by the Google sheet, so that
queries will pull from sheet dynamically each time. This incurs more latency but
is a good option if the sheet data is changing frequenly. 
        """
        self.client: GSheetsClient = GSheetsClient(spec)
        self.tables = None

    def validate(self):
        return self.client.validate(self)

    def list_tables(self):
        if not self.tables:
            actuals = super().list_tables()
            # FIXME: Have parent method return actual tables, and merge that list
            # with "declared" ones
            self.tables = [
                GSheetsTableSpec(self.client.SHEETS, tup[0], tup[1]) \
                    for tup in self.storage.list_objects('tables')
            ]
        return self.tables

    def list_files(self):
        for sheet in self.client.get_all_sheets_files():
            print(sheet['name'], "\t", sheet['webViewLink'], file=self.output)

    def search(self, search_query=None):
        for sheet in self.client.get_all_sheets_files(search_query):
            print(sheet['name'], "\t", sheet['webViewLink'], file=self.output)

    def info(self, file_or_gsheet_id=None):
        if len(file_or_gsheet_id) > 15 and file_or_gsheet_id.find(" ") == -1:
            # assume a ghseet id
            infos = [self.client.getSheetInfo(file_or_gsheet_id)]
        else:
            infos = (self.client.getSheetInfo(sh['id']) \
                for sh in self.client.get_all_sheets_files(file_or_gsheet_id))
        for sheet in infos:
            t = sheet['properties']['title']
            m = sheet['modifiedTime']
            print("Sheet '{}', last modified {}, has tabs: ".format(t, m), file=self.output)
            for idx, tab in enumerate(sheet['sheets']):
                print((idx+1), " - ", tab['properties']['title'], file=self.output)
            print("\n", file=self.output)

    def create_table(self, table_name, file_or_gsheet_id, tab_name):
        sheetId = file_or_gsheet_id               
        self.storage.put_object(
            'tables', 
            table_name,
            {'sheetId': sheetId, 'tab_name': tab_name}
        )
        # Todo: trigger a query which will load the table

    def supports_commands(self) -> bool:
        return True

    def run_command(self, code: str, output_buffer: io.TextIOBase) -> None:
        # see if base wants to run it
        if output_buffer:
            self.output = output_buffer
        if not super().run_command(code, output_buffer):
            self.last_code = code
            command = self.parser.parse_and_run(code, output_buffer)
            if command:
                result = getattr(self, command)(**self.parser._safe.args)
                if result:
                    return result
            return True
        return True

class HideOurInstanceVars:
    pass

class GsheetCommandParser(Visitor):
    def __init__(self):
        self._safe = HideOurInstanceVars()
        self._safe.parser = Lark(open(
            os.path.join(os.path.dirname(__file__), "gsheets_grammar.lark")).read())

    def parse_and_run(self, code: str, output_buffer: io.TextIOBase) -> str:
        self._safe.command = None
        self._safe.args = {}
        self._safe.output_buffer = output_buffer

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

