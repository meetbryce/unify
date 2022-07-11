# python
import io
import os
import os
import re
import shelve
import marshal
import traceback

# vendor
from googleapiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools
from lark import Lark
from lark.visitors import Visitor
from lark.tree import Tree

# project
from rest_schema import APIConnector
from parsing_utils import collect_child_strings

class GSheetsClient:
    DEFAULT_SCHEMA = "default"
    ALL_SPREADSHEETS_TABLE = "all_spreadsheets"
    MAPPED_SHEETS_TABLE = "mapped_sheets"

    SCOPES = "https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/spreadsheets"

    def __init__(self):
        store = file.Storage(os.path.expanduser('~/src/github/storage.json'))
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets(
                os.path.expanduser('~/src/github/client_id.json'), 
                GSheetsClient.SCOPES)
            # FIXME: Pass a flags=opt value where opt.auth_host_port=<port> to override 8080
            # Otherwise the callback breaks cause Presto web is already listening on 8080
            creds = tools.run_flow(flow, store)
        self.DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http('.cache')))
        self.SHEETS = discovery.build('sheets', 'v4', http=creds.authorize(Http('.cache')))
        self.TABLE_STORE = shelve.open('mapped_sheets.shelve')
        self.MAPPED = []
        self.SCHEMA_CACHE = {}
        self._loadMappedTables()

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

class GSheetsConnector(APIConnector):
    def __init__(self, spec):
        self.name = spec['name']
        self.parser: GsheetCommandParser = GsheetCommandParser()
        self.help = """
The GSheets connectors support reading and writing data from Google Sheets.
Try these commands:
  gsheets list files
  gsheets search <file name>
  gsheets info <file name> - list sheet names from a sheet  
        """
        self.client = GSheetsClient()

    def resolve_auth(self, connection_name, opts):
        pass

    def list_tables(self):
        for i in []:
            yield(i)

    def supports_commands(self) -> bool:
        return True

    def run_command(self, code: str, output_buffer: io.TextIOBase) -> None:
        # see if base wants to run it
        if not super().run_command(code, output_buffer):
            self.parser.parse_and_run(code, output_buffer)
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
        print("sheet1\nsheet2\nsheet3", file=self._safe.output_buffer)
        return tree

    def search(self, tree: Tree) -> Tree:
        self._safe.command = "search"
        self._safe.args['search_query'] = collect_child_strings("search_query", tree)
        return tree

    def info(self, tree: Tree) -> Tree:
        self._safe.command = "info"
        self._safe.args['file_or_gsheet_id'] = collect_child_strings("file_or_gsheet_id", tree)
        return tree


