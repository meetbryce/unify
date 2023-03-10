import time
from sqlalchemy.orm.session import Session

from .storage_manager import StorageManager
from .db_wrapper import DBManager, ConnectorMetadata, dbmgr

class UnifyDBStorageManager(StorageManager):
    """
        Stores connector metadata in DuckDB. Creates a "meta" schema, and creates
        tables named <connector schema>_<collection name>.
    """
    def __init__(self, connector_schema: str, duck):
        self.connector_schema = connector_schema
        self.duck : DBManager = duck

    def get_local_db(self):
        return self.duck
        
    def put_object(self, collection: str, id: str, values: dict) -> None:
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                # Good to remember that Clickhouse won't enforce unique keys!
                session.query(ConnectorMetadata).filter(
                    ConnectorMetadata.id == id,
                    ConnectorMetadata.collection==(self.connector_schema + "." + collection)
                ).delete()
                session.commit()
                session.add(ConnectorMetadata(
                    id=id, 
                    collection=self.connector_schema + "." + collection,
                    values = values
                ))
                session.commit()

    def get_object(self, collection: str, id: str) -> dict:
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                rec = session.query(ConnectorMetadata).filter(
                    ConnectorMetadata.id == id,
                    ConnectorMetadata.collection==(self.connector_schema + "." + collection)
                ).first()
                if rec:
                    return rec.values

    def delete_object(self, collection: str, record_id: str) -> bool:
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                session.query(ConnectorMetadata).filter(
                    ConnectorMetadata.id == record_id,
                    ConnectorMetadata.collection==(self.connector_schema + "." + collection)
                ).execution_options(
                    settings={'mutations_sync': 1}
                ).delete()
                session.commit()
                time.sleep(0.1)

    def list_objects(self, collection: str) -> list[tuple]:
        with dbmgr() as duck:
            with Session(bind=duck.engine) as session:
                return [
                    (row.id, row.values)for row in \
                        session.query(ConnectorMetadata).filter(
                            ConnectorMetadata.collection==(self.connector_schema + "." + collection)
                        )
                ]

