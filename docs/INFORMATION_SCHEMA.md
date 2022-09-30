# The Unify information schema

Unify cribs from the standard SQL db mechanism of the `information_schema` system tables.

We create in the target database an `information_schema` catalog with tables that reflect
configuration information about the Unify database.

`schemata` table

Lists schemas defined in the tenant database and annoates the adapter information for them.

type      | name                | type_or_spec    | comment
---------- --------------------- -----------------  ---------------------------------------------
adapter     github                github_spec.yaml  Help on the Github adapter
connection  github                github
connection  jira                  jira              Connection to our JIRA instance
connnection files                 LocalFileAdapter  Connection for importing/exporting files

`tables` table

table_name    | table_schema | connection      | refresh_schedule   | source  | provenance  | comment
-------------   -------------  ----------------  -------------------  --------  ------------  ----------
pulls           github         github            daily at 08:00       YAML from spec
projects_csv    files          files             daily at 07:00       'projects.csv'  

This table stores critical metadata about each Unify managed table. This information could include:

- The adapter and API/file source that created the table
- The refresh interval
- The last refresh time and the record count from the refresh
- The last refresh message if there was an error
- A help comment describing the table
- The "source" of the table, either a file URI or a REST adapter config block

## Non-adapter tables

Unify maintains table metadata even for tables and views that are NOT created by
adapters. Any `SELECT .. INTO` or `CREATE VIEW` operation will update a record
in `information_schema.tables` that tracks the population event for the table.

## Implementation

The information schema is maintained by SQLAlchemy classes that are configured

