#!/usr/bin/env python3
import os
import sys
import re
import subprocess
from datetime import datetime, date

def verif_pg_env():
    if 'PGHOST' not in os.environ:
        print("PGHOST not set")
        sys.exit(1)
    if 'PGDATABASE' not in os.environ:
        print("PGDATABASE not set")
        sys.exit(1)
    if 'PGUSER' not in os.environ:
        print("PGUSER not set")
        sys.exit(1)
    if 'PGPASSWORD' not in os.environ:
        print("PGPASSWORD not set")
        sys.exit(1)
    if 'CLICKHOUSE_HOST' not in os.environ:
        print("CLICKHOUSE_HOST not set")
        sys.exit(1)
    if 'CLICKHOUSE_USER' not in os.environ:
        print("CLICKHOUSE_USER not set")
        sys.exit(1)
    if 'CLICKHOUSE_PASSWORD' not in os.environ:
        print("CLICKHOUSE_PASSWORD not set")
        sys.exit(1)

def dump_table(table, schema_file, csv_file):
    verif_pg_env()
    ret = os.system(f"psql --pset=format=unaligned -c \"\\d {table}\" > {schema_file}")
    if ret != 0:
        print("Error dumping schema")
        sys.exit(1)
    print(f"Saved schema to {schema_file}")
    print(datetime.now(), " Extracting table with COPY to csv...")
    ret = os.system(f'psql -c "\\copy (select * from {table}) to {csv_file} CSV HEADER\"')
    if ret != 0:
        print("Error dumping table")
        sys.exit(1)
    print(datetime.now(), f" Wrote {csv_file}")
    print("Done")

def parse_schema_file(schema_file):
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

    return {'columns': columns, 'primary_key_cols': primary_key_cols}

def pg_to_ch_root_type(pgtype: str):
    if pgtype.endswith("_enum"):
        return "String"
    if pgtype.startswith("boolean"):
        return "Bool"
    if pgtype.startswith("character") or pgtype.startswith("jsonb") or pgtype == "text":
        return "String"
    if pgtype.startswith("time "):
        return "String"
    if pgtype.startswith("date"):
        return "DateTime"
    if pgtype.startswith("timestamp"):
        return "DateTime64(3)"
    if pgtype.startswith("int") or pgtype.startswith("bigint"):
        return "Int64"
    if pgtype.startswith("smallint"):
        return "Int32"
    if pgtype.startswith("numeric") or pgtype.startswith("real") or pgtype.startswith("double"):
        return "Float64"
    raise RuntimeError("Unknown postgres type: " + pgtype)

def pg_to_clickhouse_type(pgtype: str):
    if pgtype.endswith("[]"):
        return "String" 
        # figure out how to parse CSV arrays. Need to sub '[' for '{' and then use JSONExtract(col,'Array(Int)')
        # "Array(" + ch_root_type(pgtype) + ")"
        # 
    else:
        return pg_to_ch_root_type(pgtype)

def clickclient(sql, cat_file = None, echo=False):
    host = os.environ['CLICKHOUSE_HOST']
    user = os.environ['CLICKHOUSE_USER']
    password = os.environ['CLICKHOUSE_PASSWORD']

    if echo:
        echo = "--echo"
    else:
        echo = ""

    if cat_file:
        ret = subprocess.run(f'cat {cat_file} | clickhouse-client -h {host} -u {user} --password "{password}" --query "{sql}"', 
                                shell=True, capture_output=True)
    else:
        cmd = f'clickhouse-client {echo} -h {host} -u {user} --password "{password}" --query "{sql}"'
        ret = subprocess.run(cmd, 
                            shell=True, capture_output=True)
    if ret.returncode != 0:
        print(f"Command failed {ret}: {sql}")
        sys.exit(1)

    return ret.stdout.decode('utf-8').strip()

def load_table(table, schema_file, csv_file, create_table=True):
    if not os.path.exists(csv_file):
        raise RuntimeError("Cannot find data file: ", csv_file)

    opts = parse_schema_file(schema_file)

    if not opts['primary_key_cols']:
        raise RuntimeError("Cannot create table with no primary key found")

    import_structure = ", ".join([f"{col} {pg_to_clickhouse_type(ctype)}" for col, ctype in opts['columns'].items()])

    print("Sending to clickhouse...")
    line_count = int(return_output(f"wc -l {csv_file}").split()[0])
    print(f"Loading {line_count} rows into {table}")
    
    if create_table:
        order_cols = ', '.join(opts['primary_key_cols'])
        clickclient(f"DROP TABLE IF EXISTS {table};")
        clickclient(f"CREATE TABLE {table} ({import_structure}) ENGINE = MergeTree() ORDER BY ({order_cols});")
    clickclient(f"""INSERT INTO {table} SELECT * FROM input('{import_structure}') 
        FORMAT CSVWithNames SETTINGS date_time_input_format='best_effort';""", csv_file)

def return_output(cmd):
    val = subprocess.run(cmd, shell=True, capture_output=True, check=True)
    return val.stdout.decode('utf-8').strip()

def update_table(table, schema_file):
    opts = parse_schema_file(schema_file)
    if not opts['primary_key_cols']:
        raise RuntimeError("No primary key for the table found, have to reload")
    if len(opts['primary_key_cols']) > 1:
        raise RuntimeError("Not sure how to incremental update with multiple primary key cols")
    primary_key = opts['primary_key_cols'][0]

    max_val = clickclient(f'SELECT max({primary_key}) FROM {table}')
    print("Max primary key value: ", max_val)

    csv_file = table + str(date.today()) + ".csv"
    # Now extract PG records where the primary key is greater than what's in Clickhouse
    ret = os.system(f'psql -c "\\copy (select * from {table} where {primary_key} > {max_val}) to {csv_file} CSV HEADER\"')
    load_table(table, schema_file, csv_file, create_table=False)


verif_pg_env()

if len(sys.argv) == 2 and sys.argv[1] == 'list':
    os.system("psql -c \"\\dt\"")
    sys.exit(0)

if len(sys.argv) == 2 and sys.argv[1] == 'chlist':
    clickclient("SHOW TABLES;", echo=True)
    sys.exit(0)

if len(sys.argv) < 3:
    print("Usage: repl.py <action> <table>")
    print("action: list, chlist, dump, load, update")
    sys.exit(1)

table = sys.argv[2]
csv_file = table + ".csv"
schema_file = table + ".schema"

if sys.argv[1] == 'dump':
    dump_table(table, schema_file, csv_file)
elif sys.argv[1] == 'load':
    load_table(table, schema_file, csv_file)
elif sys.argv[1] == 'update':
    update_table(table, schema_file)
else:
    print("Unknown action: ", sys.argv[1])

