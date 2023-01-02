# DEPRECATED
# Utilities for full text search of metadata
import glob
import os
import typing

# from whoosh.index import create_in, open_dir
# from whoosh.fields import TEXT, Schema, KEYWORD, ID
# from whoosh.writing import IndexWriter
# from whoosh.qparser import QueryParser, FuzzyTermPlugin
# from whoosh.reading import ReaderClosed

class Searcher:
    """
        Uses the Whoosh library to implement full text search on database metadata
    """
    def __init__(self):
        self.ix_path = os.environ.get("UNIFY_SEARCH_DIR", "/tmp/unify_search_index")
        os.makedirs(self.ix_path, exist_ok=True)

        self.schema = Schema(
            type = KEYWORD(stored=True), # The type, like: schema/table/view/column
            key = ID(unique=True),
            name = KEYWORD(stored=True), # The object name
            parent = KEYWORD(stored=True), # The name of the parent (containing schema, or table)
            description = TEXT        # The object description
        )

        if len(glob.glob(os.path.join(self.ix_path, "*"))) == 0:
            self.index = create_in(self.ix_path, schema=self.schema)
        else:
            self.index = open_dir(self.ix_path)
        self.writer: IndexWriter = None

    def open_index(self):
        self.writer = self.index.writer()

    def clear_index(self):
        self.index.close()
        self.index = create_in(self.ix_path, schema=self.schema)

    def index_object(self, type: str, name: str, parent: str=None, description: str=None):
        self.writer.add_document(
            type=type,
            key=type + "." + (parent or '') + "." + name, 
            name=name,
            parent=parent,
            description=description
        )

    def delete_object(self, type: str, name: str, parent: str):
        key = type + "." + (parent or '') + "." + name
        self.writer.delete_by_term("key", key)

    def delete_child_objects(self, type: str, parent: str):
        q = parser = QueryParser("name", self.schema).parse(f"parent:{parent} type:{type}")
        self.writer.delete_by_query(q)

    def close_index(self):
        self.writer.commit()
        self.writer = None

    def search(self, query_str: str, type=None, deep=False):
        with self.index.searcher() as searcher:
            parser = QueryParser("name", self.schema)
            parser.add_plugin(FuzzyTermPlugin())
            query = parser.parse(query_str)
            res = searcher.search(query)
            try:
                return [hit.fields() for hit in res]
            except ReaderClosed:
                return []
