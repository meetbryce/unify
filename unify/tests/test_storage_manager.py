import os
import pytest
import time

from unify.sqla_storage_manager import UnifyDBStorageManager
from unify.db_wrapper import dbmgr
from unify.db_wrapper import ClickhouseWrapper, DuckDBWrapper

@pytest.fixture
def duck():
    with dbmgr() as duck:
        yield duck

@pytest.fixture
def store(duck):
    yield UnifyDBStorageManager("github", duck)

def test_storage_manager(store):   
    d1 = {"joe":"machine", "nancy":"pelosi"}
    d2 = {"mitch":"mcconnel", "kevin":"ryan"}

    store.put_object("col1", "key1", d1)
    assert store.get_object("col1", "key1") == d1
    assert store.get_object("col1", "key1") != d2

    store.put_object("col1", "key2", d2)
    assert store.get_object("col1", "key2") == d2

    store.put_object("col2", "key1", d1)

    res = list(store.list_objects("col1"))
    print(res)
    assert sorted(res) == sorted([("key1", d1), ("key2", d2)])

    assert list(store.list_objects("col2")) == [("key1", d1)]

    store.delete_object("col1", "key1")
    assert store.get_object("col1", "key1") is None
    assert list(store.list_objects("col1")) == [("key2", d2)]

    # Ensure stores for different adapters don't clash
    store2 = UnifyDBStorageManager("jira", store.duck)
    store.put_object("col1", "key1", d1)
    store2.put_object("col1", "key1", d2)

    assert store.get_object("col1", "key1") == d1
    assert store2.get_object("col1", "key1") == d2
    
    


