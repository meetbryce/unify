import pytest
import os
import time
import json
import uuid
import pandas as pd
import logging

from unify import dbmgr
from unify.db_wrapper import DBManager, TableHandle, DBSignals

logging.basicConfig(level=logging.DEBUG)
#logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

@pytest.fixture
def db():
    with dbmgr() as _db:
        yield _db

@pytest.mark.skip(reason="")
def test_schemas(db: DBManager):
    db.create_schema("myscheme1")
    scs = db.list_schemas()   
    assert 'myscheme1' in scs['schema_name'].tolist()

    db.drop_schema("myscheme1")
    time.sleep(0.5)
    assert 'myscheme1' not in db.list_schemas()['schema_name'].tolist()

def test_tables(db: DBManager):
    schema = "myscheme2"
    db.create_schema(schema)

    db.create_table(TableHandle("users", schema), {"*id":"VARCHAR", "name":"VARCHAR"})

    df = db.list_tables(schema)

    if not df.empty:
        alltables = df['table_name'].tolist()
        assert alltables == ['users']

    db.drop_table(TableHandle("users", schema))
    time.sleep(1)
    df = db.list_tables(schema)
    assert df.empty
    db.drop_schema(schema, cascade=True)

@pytest.mark.skip(reason="")
def test_df_tables(db: DBManager):
    schema = "myscheme3"
    db.drop_schema(schema, cascade=True)

    db.create_schema(schema)

    df = pd.DataFrame({"names": ["scott","alex","joe","mary"], "ages": [25, 35, 45, 55]})

    desc = 'the greatest people ever'
    db.write_dataframe_as_table(df, TableHandle("people", schema, {'description':desc}))

    info_tables = db.list_tables(schema)
    assert 'people' in info_tables['table_name'].tolist()

    meta_rec = info_tables.loc[info_tables['table_name'] == 'people'].to_dict('records')
    assert 'description' in meta_rec[0]
    assert meta_rec[0]['description'] == desc

    dbpeeps = db.execute(f"select * from {schema}.people")
    print(dbpeeps)
    assert sorted(dbpeeps['names'].tolist()) == sorted(df['names'].tolist())
    assert sorted(dbpeeps['ages'].tolist()) == sorted(df['ages'].tolist())

    db.drop_schema(schema, cascade=True)


from sqlalchemy.orm.session import Session
@pytest.mark.skip(reason="")
def test_connection_scans(db):
    from unify.db_wrapper import ConnectionScan

    values = {"key": "abc123", "opts": 5}

    session = Session(bind=db.engine)
    c = ConnectionScan(table_name="pulls",table_schema="github",connection="github")
    c.values = values
    session.add(c)

    scan = session.query(ConnectionScan).order_by('created').first()
    print("found scan: ", scan)
    assert c.values == values

@pytest.mark.skip(reason="")
def test_create_table_signal(db: DBManager):
    signals = []
    def on_table_create(dbmgr, table):
        signals.append((table, table.table_opts()))

    db.register_for_signal(DBSignals.TABLE_CREATE, on_table_create)

    db.create_schema("sch1")
    db.execute("drop table if exists sch1.users")
    db.execute("create table sch1.users (id VARCHAR, PRIMARY KEY(id))")

    assert len(signals) == 1

    db.execute("drop view if exists sch1.all_users")
    db.execute("create view sch1.all_users as select * from sch1.users")
    print(signals)
    assert len(signals) == 2

