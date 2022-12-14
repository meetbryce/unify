#!/usr/bin/env python3
import os
import sys
import re

if len(sys.argv) < 2:
    print("Please provide the table name")
    sys.exit(1)

table = sys.argv[1]
csv_file = table + ".csv"
if not os.path.exists(csv_file):
    print("Cannot find data file: ", csv_file)
    sys.exit(1)

line: str = ""
inside_cols: bool = False
inside_idxs: bool = False
columns: dict = {}
primary_key_cols = []

with open(f"{table}.schema") as f:
    for line in f.readlines():
        m = re.search(r"able \"(\w+)\.(\w+)", line)
        if m:
            schema = m.group(1)
            sc_table = m.group(2)
            if sc_table != table:
                print("Error, schema references the wrong table: ", sc_table)
        if line.startswith("Column|"):
            inside_cols = True
            continue
        if line.startswith("Indexes:"):
            inside_cols = False
            inside_idxs = True
            continue
        if inside_cols and line.count("|") >= 4:
            cname, ctype,collation,nullable,cdefault = line.split("|")
            columns[cname] = ctype
        if inside_idxs:
            m = re.search(r"PRIMARY KEY.*\((.*)\)", line)
            if m:
                primary_key_cols = m.group(1).split(",")
                print("Primary cols: ", primary_key_cols)

if not primary_key_cols:
    print("Cannot create table with no primary key found")
    sys.exit(1)

def ch_root_type(pgtype: str):
    if pgtype.endswith("_enum"):
        return "String"
    if pgtype.startswith("boolean"):
        return "Bool"
    if pgtype.startswith("character") or pgtype.startswith("jsonb") or pgtype == "text":
        return "String"
    if pgtype.startswith("time "):
        return "String"
    if pgtype.startswith("date"):
        return "Date32"
    if pgtype.startswith("timestamp"):
        return "DateTime64(3)"
    if pgtype.startswith("int") or pgtype.startswith("bigint"):
        return "Int64"
    if pgtype.startswith("smallint"):
        return "Int32"
    if pgtype.startswith("numeric") or pgtype.startswith("real") or pgtype.startswith("double"):
        return "Float64"
    raise RuntimeError("Unknown postgres type: " + pgtype)

def ch_type(pgtype: str):
    if pgtype.endswith("[]"):
        return "String" 
        # figure out how to parse CSV arrays. Need to sub '[' for '{' and then use JSONExtract(col,'Array(Int)')
        # "Array(" + ch_root_type(pgtype) + ")"
        # 
    else:
        return ch_root_type(pgtype)

import_structure = ", ".join([f"{col} {ch_type(ctype)}" for col, ctype in columns.items()])
sql = f"""
    DROP TABLE IF EXISTS {table};
    CREATE TABLE {table} ({import_structure}) ENGINE = MergeTree() ORDER BY ({", ".join(primary_key_cols)});
    INSERT INTO {table} SELECT * FROM file('{csv_file}', CSVWithNames, '{import_structure}') SETTINGS date_time_input_format='best_effort';
"""
print("Sending to clickhouse...")
for sline in sql.splitlines():
    if not sline.strip():
        continue
    print(sline)
    ret = os.system(f'clickhouse-client --query "{sline}"')
    if ret != 0:
        print(f"Command failed {ret}: {sline}")
        sys.exit(1)

#insert into tenant_scottp.pp_orders select * from file('orders.csv',CSVWithNamesAndTypes);