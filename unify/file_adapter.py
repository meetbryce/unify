from logging import root
import os
from pathlib import Path
import subprocess
import mimetypes

import pandas as pd

from unify.rest_schema import Adapter, AdapterQueryResult, OutputLogger, UnifyLogger, StorageManager, TableDef

class LocalFileTableSpec(TableDef):
    # Represents a Google Sheet as a queryable Table spec to Unify.

    def __init__(self, table: str, opts: dict):
        super().__init__(table)
        self.file_uri = opts['file_uri']
        self.reader_name = opts['reader_name']
    
    def query_resource(self, tableLoader, logger: UnifyLogger):
        path = Path(self.file_uri)
        with getattr(pd, self.reader_name)(path, chunksize=5000) as reader:
            for chunk in reader:
                chunk.dropna(axis='rows', how='all', inplace=True)
                chunk.dropna(axis='columns', how='all', inplace=True)
                size_return = []
                yield AdapterQueryResult(json=chunk, size_return=size_return)


class LocalFileAdapter(Adapter):
    def __init__(self, spec, root_path: str, storage: StorageManager):
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

    def list_files(self, path: str) -> list[str]:
        if path is None:
            path = ""
        p = Path(os.path.join(self.root_path, path))
        return [f.name for f in p.glob("*")]

    def can_import_file(self, path):
        # Path will be whatever the user entered. Either it will be a path relative
        # to the system root, or it could be absolute in which case it needs to
        # start with our same root

        userp = Path(path)
        if path.startswith("/"):
            return userp.is_relative_to(self.root_path) and userp.exists()
        else:
            return self.root_path.joinpath(path).exists()

    def import_file(self, file_uri: str, options: dict={}):
        print("importing the file")

        if not file_uri.startswith("/"):
            file_uri = str(self.root_path.joinpath(file_uri))

        reader = self.determine_pandas_reader(file_uri)
        if reader is None:
            raise RuntimeError(f"Cannot determine type of contents for '{file_uri}'")

        # Now create an empty table record which we will fill via the table scan later
        table_name = self.convert_string_to_table_name(Path(file_uri).stem)
        self.storage.put_object(
            'tables', 
            table_name,
            {'file_uri': file_uri, 'reader_name': reader.__name__}
        )
        self.tables = None # force re-calc
        return table_name

    def determine_pandas_reader(self, file_uri: str):
        res = mimetypes.guess_type(file_uri)
        mime = None
        if res[0] is None:
            # Fall back to using 'file' system command
            res = subprocess.check_output(["file", "-b", "--mime-type", file_uri])
            mime = res.decode("utf8").strip()
        else:
            mime = res[0]
        if mime == 'text/csv':
            return pd.read_csv
        elif 'spreadsheetml' in mime:
            return pd.read_excel
        elif mime.endswith('xml'):
            return pd.read_xml
        elif 'parquet' in mime:
            return pd.read_parquet
        else:
            return None

    # Exporting data
    def create_output_table(self, file_name, output_logger: OutputLogger, overwrite=False, opts={}):
        return self.root_path.joinpath(file_name)

    def write_page(self, output_handle, page: pd.DataFrame, output_logger: OutputLogger, append=False, page_num=1):
        page.to_csv(output_handle, header=(page_num==1), mode='a',index=False)

    def close_output_table(self, output_handle):
        # Sheets REST API is stateless
        pass
