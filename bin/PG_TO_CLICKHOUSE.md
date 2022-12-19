# Replicating Postgres data to a Clickhouse warehouse

1. Setup so that the Postgres and Clickhouse databases are reachable from "psql" and "clickhouse-client"
    without needing to provide any credentials:
        PGHOST, PGDATABASE, PGUSER, PGPASSWORD
        CLICKHOUSE_USER, CLICKHOUSE_PASSWORD

2. Use "COPY" to dump each Postgres table:
    ./dump_table <table>

This writes `table.schema` and `table.csv`. 

3. Now load the table into Clickhouse:

    ./load_ch_table.py <table>

This script:
    a. Maps the Postgres schema types to Clickhouse types
    b. Determines the Order by column or columns by inspecting the Postgres schema
    c. Loads the table into Clickhouse using INSERT INTO .. SELECT from input and piping the CSV into the client


## Incremental updates

After we do the initial load, we need to submit incremental updates to the warehouse. We don't intend
to synchronize deletes. For that the user should initial a full reload of the table. To synchronize
updates we need some timestamp on the row. Once we have idenified the "update" column then we can
query for all rows from the table with a greater value. On the Clickhouse side we need to rows that match
our updated records, and then insert the new records. 

A simpler strategy, good for very large tables, is ONLY to insert new records. In this case we just
need to find records with a greater numeric value than the greatest record we have in the warehouse, and
then insert those records.

The basic strategy is:
    - Select max(order col) from the warehouse
    - Use COPY (select .. where) to extract the new records to a CSV file
    - Use our loader script but don't try to create the table

### Error handling

If the COPY command fails then we can simply retry and rewrite the CSV file.
If the bulk INSERT into Clickhouse fails then we need to figure out the max order id which
DID get recorded, and insert all records AFTER that one from the CSV file.
