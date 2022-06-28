import duckdb
import glob
import pandas as pd

duck = duckdb.connect()

#duck.execute("create table orgs as select * from read_parquet('./data/github.orgs/*')")
duck.execute("create schema github")
duck.execute("create table github.orgs as select * from read_parquet('./data/github.orgs/*')")

print(duck.execute("select * from github.orgs").fetchall())


df = pd.read_parquet(glob.glob('./data/github.orgs/*'))
duck.execute("set search_path='github'")
#table = duck.table("github.orgs")
duck.append('orgs', df)
duck.append('orgs', df)

print(duck.execute("select * from github.orgs").fetchall())


