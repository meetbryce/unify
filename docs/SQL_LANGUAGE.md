# SQL Language

Unify implements standard SQL plus a number of extensions for working with connected
systems, adapters, and other features.

## Help and information
 
    help
    help [info, schemas, charts, import, export]

    show schemas
    show tables
    show tables from <schema>
    show columns from <schema>.<table>
    show columns from <schema>.<table> like 'pattern'
    describe 
    describe <schema>.<table>
    
## Standard language

    select [column refs] from ...
    create table <schema>.<table> ...
    create table <schema>.<table> ... as <select statement>
    create view <schema>.<table> ... as <select statement>
    insert into <schema>.<table> ...
    delete from <schema>.<table> [where ...]

    drop table <schema>.<table>
    drop schema <schema> ["cascade"]

## Charting

    create chart [<name>] [from <chart source>] as <chart type> where x = <col ref> and y = <col ref>

## Importing data

Generally importing data from connected systems is implicit. The system definition will define a set of logical tables that will appear inside the schema for the system. Querying any table will cause the data for the table to be imported from the connected system.

Some systems, like Google Sheets, have special commands for importing data. Use `<schema> help` to learn about the commands.

### Importing file data

You can import flat file data using the `import` command:

    import '<file url>' into <schema>.<table> ["overwrite"|"append"]

This command will create the indicated table according to the schema of the source file, which should be in
csv or parquet format. If the table exists then this command will return an error unless
you specify either the `overwrite` or `append` option.

The file url can either be a local path or an S3 file URL. Whether the file contains CSV or Parquet
format will be automatically detected.

## Writing to connected systems

    export <schema>.<table> to [adapter] 'file name'|expr ["overwrite"|"append"]

This will export all rows of the indicated table to the connected system specified by the "adapter" name. Only certain connected systems support exporting data. Use the "overwrite" option to allow overwriting an existing file
and its contents. Use the "append" option to append to any existing file.

    export hubspot.orders to s3 '/bucket1/order.csv'
    export hubspot.orders to s3 '/bucket1/order.parquet'

    export hubspot.orders to file '/tmp/orders.csv'
    export hubspot.orders to gsheets 'Hubspot Orders'

The file name argument can also be any expression enclosed in parenthesis. This allows constructing
the target file name dynamically:

    export hubspot.orders to gsheeets ('Hubspot Order as of ' || current_date)
