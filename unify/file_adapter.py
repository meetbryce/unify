from logging import root
import inspect
import os
from pathlib import Path
import subprocess
import mimetypes

import pandas as pd

from .adapters import Adapter, AdapterQueryResult, OutputLogger, UnifyLogger, StorageManager, TableDef

class LocalFileTableSpec(TableDef):
    # Represents a Google Sheet as a queryable Table spec to Unify.

    def __init__(self, table: str, opts: dict):
        super().__init__(table)
        self.file_uri = opts['file_uri']
        self.reader_name = opts['reader_name']
        self.options = opts.get('options', [])
    
    def get_table_source(self):
        return self.file_uri

    def query_resource(self, tableLoader, logger: UnifyLogger):
        path = Path(self.file_uri)

        size_return = []
        kwargs = {}

        for idx, k in enumerate(self.options):
            if k == 'skip':
                count = int(self.options[idx+1])
                kwargs["skiprows"] = count
            elif k == 'header':
                count = int(self.options[idx+1])
                kwargs["header"] = count

        method = getattr(pd, self.reader_name)
        if 'chunksize' in inspect.signature(method).parameters:
            kwargs["chunksize"] = 5000
            with method(path, **kwargs) as reader:
                for chunk in reader:
                    chunk.dropna(axis='rows', how='all', inplace=True)
                    chunk.dropna(axis='columns', how='all', inplace=True)
                    yield AdapterQueryResult(json=chunk, size_return=size_return)
        else:
            df = method(path, **kwargs)
            yield AdapterQueryResult(json=df, size_return=size_return)


class LocalFileAdapter(Adapter):
    def __init__(self, spec, root_path: str, storage: StorageManager, schema_name: str):
        super().__init__(spec['name'], storage)
        # FIXME: Use user-specific path when we have one
        self.root_path = Path(root_path)
        self.logger: OutputLogger = None
        self.tables = None

    def list_tables(self):
        if not self.tables:
            self.tables = [
                LocalFileTableSpec(tup[0], tup[1]) \
                    for tup in self.storage.list_objects('tables')
            ]
        return self.tables

    def drop_table(self, table_root: str):
        self.storage.delete_object('tables', table_root)
        self.tables = None

    def rename_table(self, table_root: str, new_name: str):
        values = self.storage.get_object('tables', table_root)
        if values:
            self.storage.delete_object('tables', table_root)
            self.storage.put_object('tables', new_name, values)
        self.tables = None

    def list_files(self, match: str) -> list[str]:
        if match is None:
            match = "*"
        else:
            match = match.replace('%', '*')
        return [f.name for f in self.root_path.glob(match)]

    def _resolve_path(self, path: str):
        userp = Path(path)
        if path.startswith("/"):
            return userp
        else:
            return self.root_path.joinpath(path)

    def can_import_file(self, path):
        # Path will be whatever the user entered. Either it will be a path relative
        # to the system root, or it could be absolute in which case it needs to
        # start with our same root
        userp = self._resolve_path(path)
        return userp.is_relative_to(self.root_path) and userp.exists()

    def peek_file(self, file_uri: str, line_count: int, logger: OutputLogger):
        file_path = self._resolve_path(file_uri)
        reader = self.determine_pandas_reader(file_path)

        if reader == pd.read_csv:
            with open(self._resolve_path(file_uri)) as f:
                for x in range(line_count):
                    line: str = f.readline()
                    if not line:
                        break
                    logger.print(str(x+1) + " " + line.strip())
            return None
        elif reader == pd.read_excel:
            df = pd.read_excel(file_path)
            df = df.head(n=line_count)
            df.insert(0, 'row', df.index + 1)
            return df

    def import_file(self, file_uri: str, options: list=[]):
        file_path = self._resolve_path(file_uri)
        if not file_path.exists():
            raise RuntimeError(f"Cannot find file '{file_uri}'")

        reader = self.determine_pandas_reader(file_path)
        if reader is None:
            raise RuntimeError(f"Cannot determine type of contents for '{file_uri}'")

        # Now create an empty table record which we will fill via the table scan later
        table_name = LocalFileAdapter.convert_string_to_table_name(file_path.stem)
        self.storage.put_object(
            'tables', 
            table_name,
            {'file_uri': file_path.as_uri(), 'reader_name': reader.__name__, 'options': options}
        )
        self.tables = None # force re-calc
        return table_name

    def determine_pandas_reader(self, file_uri: Path):
        res = mimetypes.guess_type(file_uri.as_posix())
        mime = None
        if res[0] is None:
            # Fall back to using 'file' system command
            res = subprocess.check_output(["file", "-b", file_uri])
            mime = res.decode("utf8").strip().lower()
        else:
            mime = res[0]
        if 'csv' in mime:
            return pd.read_csv
        elif 'spreadsheetml' in mime or 'excel' in mime:
            return pd.read_excel
        elif 'xml' in mime:
            return pd.read_xml
        elif 'parquet' in mime:
            return pd.read_parquet
        else:
            return None

    # Exporting data
    def create_output_table(self, file_name, output_logger: OutputLogger, overwrite=False, opts={}):
        return self.root_path.joinpath(file_name)

    def write_page(self, output_handle, page: pd.DataFrame, output_logger: OutputLogger, append=False, page_num=1):
        path = output_handle
        res = mimetypes.guess_type(path)
        if res[0] is not None and 'spreadsheetml' in res[0]:
            page.to_excel(path, index=False)
        elif str(path).lower().endswith(".parquet"):
            page.to_parquet(path, index=False)
        else:
            # csv chosen or fall back to csv
            page.to_csv(output_handle, header=(page_num==1), mode='a',index=False)

    def close_output_table(self, output_handle):
        # Sheets REST API is stateless
        pass
