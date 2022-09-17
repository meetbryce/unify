from logging import root
import os
import glob

class FileSystem:
    """
        The FileSystem implements a virtual file system made available to the Unify environment.
        It mounts one or more FileAdapters which provide an import and export file interface,
        plus the ability to list files.
    """
    def __init__(self, root_path: str) -> None:
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        self.adapter_classes = {
            "local": "LocalFileAdapter"
        }
        self.mounts = {}
        self.root_path = root_path

    def add_adapter(self, config: dict):
        cname = self.adapter_classes[config['adapter']]
        self.root_paths[config['mount']] = globals()[cname](config, config['mount'])

    def list_mounts(self):
        return self.mounts.keys()

    def list_files(self, path):
        path = path.split("/")
        adapter = self.mounts[path[0]]
        return self.adapter.list_files(path[1:]) + self.list_mounts()

    def get_system_local_path(self, vpath: str):
        return os.path.join(self.root_path, vpath)

class FileAdapter:
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path

    def list_files(self, path: list) -> list[str]:
        return glob.glob(os.path.join(self.root_path, "*"))

        
