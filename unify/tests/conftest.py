import os

os.environ['UNIFY_HOME'] = os.path.dirname(__file__)
os.environ['UNIFY_CONNECTIONS'] = os.path.join(os.path.dirname(__file__), 'connections.yaml')
os.environ['UNIFY_SEARCH_DIR'] = '/tmp/unify_test_searchidx'
# hack for performance boost
os.environ['UNIFY_SKIP_COLUMN_INTEL'] = 'true'
