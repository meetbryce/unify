import time
from sqlalchemy.orm.session import Session

from .storage_manager import StorageManager
from .db_wrapper import DBManager, AdapterMetadata

class UnifyDBStorageManager(StorageManager):
    """
        Stores adapter metadata in DuckDB. Creates a "meta" schema, and creates
        tables named <adapter schema>_<collection name>.
    """
    def __init__(self, adapter_schema: str, duck):
        self.adapter_schema = adapter_schema
        self.duck : DBManager = duck

    def get_local_db(self):
        return self.duck
        
    def put_object(self, collection: str, id: str, values: dict) -> None:
        from unify import dbmgr
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                # Good to remember that Clickhouse won't enforce unique keys!
                session.query(AdapterMetadata).filter(
                    AdapterMetadata.id == id,
                    AdapterMetadata.collection==(self.adapter_schema + "." + collection)
                ).delete()
                session.commit()
                session.add(AdapterMetadata(
                    id=id, 
                    collection=self.adapter_schema + "." + collection,
                    values = values
                ))
                session.commit()

    def get_object(self, collection: str, id: str) -> dict:
        from unify import dbmgr
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                rec = session.query(AdapterMetadata).filter(
                    AdapterMetadata.id == id,
                    AdapterMetadata.collection==(self.adapter_schema + "." + collection)
                ).first()
                if rec:
                    return rec.values

    def delete_object(self, collection: str, record_id: str) -> bool:
        from unify import dbmgr
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                session.query(AdapterMetadata).filter(
                    AdapterMetadata.id == record_id,
                    AdapterMetadata.collection==(self.adapter_schema + "." + collection)
                ).execution_options(
                    settings={'mutations_sync': 1}
                ).delete()
                session.commit()
                time.sleep(0.1)

    def list_objects(self, collection: str) -> list[tuple]:
        from unify import dbmgr
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                return [
                    (row.id, row.values)for row in \
                        session.query(AdapterMetadata).filter(
                            AdapterMetadata.collection==(self.adapter_schema + "." + collection)
                        )
                ]

