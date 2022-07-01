from rest_schema import Connector

class GSheetsConnector(Connector):
    def __init__(self, spec):
        self.name = spec['name']

    def resolve_auth(self, connection_name, opts):
        pass

    def list_tables(self):
        for i in []:
            yield(i)