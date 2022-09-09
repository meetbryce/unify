import os

os.environ['UNIFY_CONNECTIONS'] = os.path.join(os.path.dirname(__file__), 'connections.yaml')
# hack for performance boost
os.environ['UNIFY_SKIP_COLUMN_INTEL'] = 'true'