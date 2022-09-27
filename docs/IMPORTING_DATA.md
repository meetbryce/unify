# Importing data

The core functionality of Unify is to import data from connected cloud systems via Adapters.
We also support importing file data from local or cloud hosted file systems.

Basic importing is handled either implicitly by `select`ing from a mapped table or by
using `import` to import from a file.

## Refreshing data

For purporses other than ad-hoc analysis we will want to refresh our local copy of imported
data. Generally this can happen either on a fixed schedule or in response to some external
trigger. Repeated schedule should be the "default" mechanism.

The refresh time and interval is configured on the local `table`. This config can be setup
by default in the adapter:

tables:
  - name: orders
    refresh:
        strategy: updates
        interval: daily
        time: 08:00

But that interval can be overidden on the live instance:

    > refresh table <table> every [hour|day|week] at <datetime or time>

and we can see the schedule on a table with `show info`:

    > refresh table orders every day at 12:05
    > show info for orders
    ...
    refresh: daily at 12:05

### Refreshing file imports

The `refresh` command also works on tables imported from files:

    > import 'projects.csv'
    ... created table files.projects

    > refresh table files.projects every day at 07:00

With this set then the file data will be re-imported every day. For now file imports
don't support any refresh strategy other than to reload the entire table.


