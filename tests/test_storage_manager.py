import pytest

from unify import DuckdbStorageManager, DuckContext

@pytest.fixture
def duck():
    with DuckContext() as duck:
        yield duck

@pytest.fixture
def store(duck):
    yield DuckdbStorageManager("github", duck)

def test_storage_manager(store):   
    d1 = {"foo":"bar", "key2":"val2"}
    d2 = {"cat":"dog", "name":"house"}

    store.put_object("col1", "key1", d1)
    assert store.get_object("col1", "key1") == d1
    assert store.get_object("col1", "key1") != d2

    store.put_object("col1", "key2", d2)
    assert store.get_object("col1", "key2") == d2

    store.put_object("col2", "key1", d1)

    assert list(store.list_objects("col1")) == [("key1", d1), ("key2", d2)]
    assert list(store.list_objects("col2")) == [("key1", d1)]

    store.delete_object("col1", "key1")
    assert store.get_object("col1", "key1") is None
    assert list(store.list_objects("col1")) == [("key2", d2)]

    # Ensure stores for different adapters don't clash
    store2 = DuckdbStorageManager("jira", store.duck)
    store.put_object("col1", "key1", d1)
    store2.put_object("col1", "key1", d2)

    assert store.get_object("col1", "key1") == d1
    assert store2.get_object("col1", "key1") == d2
    
    


