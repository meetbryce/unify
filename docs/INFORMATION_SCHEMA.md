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

`connectionscans` table

This meta table keeps a history of "tablescans" (pull data from an API) performed against a 
given connected table.

id   | created    | table_name  | table_schema  | connection  | values (JSON blog)
-----  ----------   -----------   -------------   -----------   --------------------
guid   timestamp    pulls         github          github        json blob


## Non-adapter tables

Unify maintains table metadata even for tables and views that are NOT created by
adapters. Any `SELECT .. INTO` or `CREATE VIEW` operation will update a record
in `information_schema.tables` that tracks the population event for the table.

## Implementation

The information schema is maintained by SQLAlchemy classes that are configured to use the active
tenant's schema.

## Data provenance

We want to keep a complete record of the "provenance" of each table. So as both to understand
the contents of the table, but also to automatically execute pipelines which maintaing derived
data sets.

Examples:

The "prs_and_tickets" view collates information from Github and JIRA together into a single view.
It's provenance, in reverse order, looks like:

  prs_and_tickets view
  -> create view as (select from jira_issues, pr_counts JOINED to github.coders)
     ancestor tables: jira_isses, pr_counts, github.coders
        
      jira_issues
      -> loaded by the JIRA adapter, issues table

      pr_counts
      -> CTE loaded by the Github adapter, pulls table

      github.coders
      -> create view as (select from $emps)
      ancestor tables: $emps

        $emps:
        -> select from github.users, gsheets.employees
           ancestor tables: github.users, gsheets.employees

           github.users
           -> loaded by the Github adapter
           
           gsheets.employees
           -> loaded from a spreadsheet by the GSheets adapter

Now, whenever the system observes new data arriving at any of the source tables
(employees, github.user, github.pulls, jira.issues) then it should be able to
re-calculate the pr_and_tickets view by re-executing the dependency tree.

### Tables or scripts

Should we store "scripts", and maintain the depedendency between them, and automatically
re-execute them? This has the advantage of supporting non-table creating commands inside
scripts like building dashboards or sending emails. But it requires us to save and
manage "script" models in the database. We also have to assume that scripts are correct
when executed in order.

The advantage of "table dependence" is that we just have to record data for each table
and then traverse the relationships to understand the provenance.

## Pull or push

We really want to be able to 'pull' results through the system. Basically I have an output
target that I want created:

- Construct a table of data (for use by someone else)
- Construct a chart report (and email or display it)
- Construct a tabular report (and email or display it)
- Construct a "composite" report of other reports

And I want to generate these outputs on some kind of schedule (daily report, or dashboard
updated hourly).

The generation of these reports should "pull" the dependent data through the system. That is,
I should trace through the data sets used by the reports back ultimately to adapter tables, refresh the data from those adapters, then run any intermediate transformation steps, and
then re-generate the report.

So I have:

**report** This is the fundamental thing that I want to create.


    