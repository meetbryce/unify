from unify import BasicStorageManager

def test_storage_manager():
    store = BasicStorageManager()

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

    assert store.delete_object("col1", "key1") == True
    assert list(store.list_objects("col1")) == [("key2", d2)]
    assert store.delete_object("col1", "key1") == False


