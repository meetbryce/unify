# Unify

Unify is a simple tool which allows you to use SQL to query information from any
cloud based system that supports a JSON-result REST API. 

System APIs are described in a configuration which models API endpoints as tables.
The JSON payload results from API calls are automatically flattened so that
result fields become columns in the table.

The tables representing the API endpoints for each system are grouped under a database schema.

After data is downloaded via API calls, it is stored to a local database (using DuckDB).

## Example

Querying the list of repositories you have access to on Github:

    > select id, name, owner_login, open_issues_count from github.repos;
    id                                  name owner_login  open_issues_count
    51189180                           philo   tatari-tv                 74
    51713527                        pytrends   tatari-tv                  0
    63492894                        philo-fe   tatari-tv                 30
    67756418                     grey-matter   tatari-tv                 34

## Configuration

REST API configurations are defined in YAML and stored in the `rest_specs` folder.
Consult [spec_doc.yaml](./rest_specs/spec_doc.yaml) for the specification. To
create access to a new system you just need to create a new YAML file describing
the system and restart the server.

## Connections

"Connections" define authentication into a particular account of a cloud system.
You need to define a connection for each cloud system you want to access. That
connection should reference the REST API spec (by name), and provide authentication
details.

The connections file holds a list of connection definitions, that looks like:

    - <schema name>
      spec: <REST API spec file prefix>
      options:
        <key>: <value>
        <key>: <value>

The "spec" field value should match the prefix of one of the "rest_spec" files. That
file will usually reference variables in its `auth` block. You should define values
for those variables in the `options` section of the connection.

You can store sensitive values directly into `connections.yml`, or you can use
`$<name>` syntax to reference an environment variable and store secrets in the
environment instead.

## Updates to the source system

Each API spec can define a "refresh strategy" which indicates how changes to the
system should be queried and merged into the local copy.

**Full Reload**

The default and simplest model is simply to perform a full table load again from
the source system.

**Incremental load**

In this model the REST API must support a filter which returns "all changes since
time t". The system will track the timestamp and provide it as a filter for the
next query.

**(future) Change data capture**

If the system support webhooks for broadcasting change events, then Unify can subscribe
to webhooks to be notified of changes to any records in the source system.

## Variables

Unify extends normal SQL syntax to support `$name` format variables:

    set $last_record = (select * from items order by created desc limit 1)
    ...
    select * from items where created > $last_record.created

By default variables will be evaluated into a result when the `set` operation occurs.
However, you can request "lazy" evaluation in which case the variable acts like
a VIEW which is evaluated when you reference it.

    lazy set $scotts_items = (select * from items where owner = 'scott')
    ...
    select * from $scotts_items  // query will be evaluated when referenced

Note that variables are automatically persisted across sessions. Use `unset` to
delete a variable:

    unset $scotts_items

## Exporting data

Unify supports the `>>` operator for exporting data out of the system:

    > select * from orders >> file:./orders.csv     // exported as CSV
    > select * from orders >> file:./orders.parquet // export as Parquet
    > select * from orders >> file[parquet]:./orders_data // export as Parquet
    > select * from orders >> s3:/bucket1/orders.parquet // export as Parquet

### Integration with Google sheets

Unify integrates to read and write data with Google Sheets.

To export a query to a Gsheets file, use this syntax:

    > select * from orders >> gsheets:<file name or sheetId>[/<tab name>]


To import from Gsheets, configure a Gsheets connection and use the custom
`gsheets` command to import data from your spreadsheets:

    > gsheets list files
    ...lists all Gsheet files
    > gsheets search <query>
    ...searches for Gsheet files whose title matches the query
    > gsheets info <file name or gsheet Id>
    ...lists the tabs of the idicated gsheet file
    > gsheets import <file name or gsheet Id> 
    ...imports the first sheet from the indicated Gsheet file. This will create
    a new table in the gsheets connection schema with a name derived from the file name
    > gsheets import <file name or gsheet Id> sheet <sheet name or number>
    ...indicates the specific sheet to import (by sheet name or numeric index starting with 1)
    > gsheets import <file> sheet <sheet> as table <name>
    ...imports the indicated sheet into a table with the indicated table name. If the
    table already exists then data from the sheet will be appended to the table

## TODO

1. Implement unit tests
1. Implement table refresh, with support for strategies
1. Implement GSheets adapter
1. Implement Lark parser for more complex syntax support
1. Implement full `show` commands
1. Implement dollar variables
1. Unobtrusive table loading status supporting interrupts
1. Implement "script UI" like notebooks




