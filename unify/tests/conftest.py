import os

os.environ['UNIFY_HOME'] = os.path.dirname(__file__)
os.environ['UNIFY_CONNECTIONS'] = os.path.join(os.path.dirname(__file__), 'connections.yaml')
