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

1. [done] Implement unit tests
1. Implement table refresh, with support for strategies
1. Implement GSheets adapter
1. Implement AWS Cost Reporting adapter
1. [done] Implement Lark parser for more complex syntax support
1. [done] Implement full `show` commands
1. Implement dollar variables
1. Unobtrusive table loading status supporting interrupts
1. Jupyter UI integration: 
    1. Schema browser tree panel
    1. Custom charting command
    1. Implement more autocompletions

### GUI

Instead of creating a custom GUI, we are integrating with Jupyter and Jupyterlab instead.

The basic SQL command line integration is straightforward, except that we want to 
offer intelligent autocompletion for schemas, tables, and columns.

Beyond SQL, we want to make it easy to construct charts, and eventually dashboards,from
the results of queries.

There are many charting libraries for Jupyter, but most of them are extremely complicated.
So for now we are implementing our own charting commands, and internally mapping those
onto MatPlotLib.

    select count(*) as count, user_login from github.pulls group by user_login
    create chart prs_by_user as bar_chart where x = user_login and y = count

    select sum(spend) as revenue, date_trunc('month','timestamp') as month
    create chart rev_by_month as line_chart where x = month and y = revenue

The full chart syntax should look like:

    create chart [<name>] [from <chart source>] as <chart type> where x = <column> and y = <column>

<name> - any identifier
<chart source> - $var, table name, chart name list, or a sub-query in parens
<chart type> - bar_chart, line_chart, pie_chart
<column> - column reference

More parameters to the chart can be captured in more k=<val> stanzas in the where clause.

Multiple charts can be combined as:

    create chart from <chart1>, <chart2>

So:
    create chart chart1 as ...
    create chart chart2 as ...
    create chart combo as chart1, chart2
        
## Developing

Make sure to setup the environment with:

    export $(cat .env)
    

