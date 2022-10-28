class StorageManager:
    """
        Passed to adapters to allow them to store additional metadata for their operation.
        Exposes a simple object storage interface:

        mgr.put_object(collection: str, id: str, values: dict)
        mgr.get_object(collection: str, id: str) -> values: dict
        mgr.list_objects(collection: str) -> results: list<dict>
    """
    def put_object(self, collection: str, id: str, values: dict) -> None:
        pass

    def get_object(self, collection: str, id: str) -> dict:
        return {}

    def delete_object(self, collection: str, id: str) -> bool:
        return False

    def list_objects(self, collection: str) -> list[dict]:
        return []

class MemoryStorageManager(StorageManager):
    def __init__(self):
        self.collections = {}

    def get_col(self, name) -> dict:
        if name not in self.collections:
            self.collections[name] = {}
        return self.collections[name]

    def put_object(self, collection: str, id: str, values: dict) -> None:
        self.get_col(collection)[id] = values

    def get_object(self, collection: str, id: str) -> dict:
        return self.get_col(collection)[id]

    def delete_object(self, collection: str, id: str) -> bool:
        col = self.get_col(collection)
        if id in col:
            del col[id]
            return True
        else:
            return False
            
    def list_objects(self, collection: str) -> list[tuple]:
        return [(key, val) for key, val in self.get_col(collection).items()]

