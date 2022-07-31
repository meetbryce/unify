import duckdb
import glob
import pandas as pd
import os
from orator import DatabaseManager

if False:
	print(__file__)
	print(os.path.join(os.path.dirname(__file__), "tests"))

	duck = duckdb.connect()

	#duck.execute("create table orgs as select * from read_parquet('./data/github.orgs/*')")
	duck.execute("create schema github")
	#duck.execute("create table github.orgs as select * from read_parquet('./data/github.orgs/*')")

	print(duck.execute("select * from github.orgs").fetchall())


	df = pd.read_parquet(glob.glob('./data/github.orgs/*'))
	print(df)
	duck.execute("set search_path='github'")
	#table = duck.table("github.orgs")
	duck.append('orgs', df)
	duck.append('orgs', df)

	cols = df.columns.tolist()
	#cols = cols[-1:] + cols[:-1]

	if cols == df.columns.tolist():
		print("same")
	else:
		print("different!")
	df = df[cols]
	duck.append('orgs', df)

	print(duck.execute("select * from github.orgs").fetchall())

config = {
    'sqlite': {
        'driver': 'sqlite',
        'database': './data/sqlstore.db'
    }
}

db = DatabaseManager(config)
results = db.select("select 1")
print(list(results))

from orator import Model

Model.set_connection_resolver(db)

