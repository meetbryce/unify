## Testing

Tests are in the [tests](./tests) folder. They use `pytest`.
They may need some environment variables to be set. To be
compatible with VSCode I keep them in a local .env file and
use:

    export $(cat .env)

To source them into my environment.

Simply run `pytest` to run the tests.

## Parser

Unify builds its own command interpreter on top of DuckDB, so that it can offer extended operations and syntax without modifying DuckDB.

The parser uses the `lark` package. Find the grammer in [grammark.lark](grammar.lark). Tests for the gammar are in [tests/test_parser.py](tests/test_parser.py).



## Class model

An `Adapter` is the manager object which implements a connection to a particular
cloud system. The Adapter is a logical entity that is configured via the
`adapter spec` YAML file. Each adapter has a name which identifies the cloud
system that it connects to, like "Github" or "Salesforce".

Some adapters will be implemented by a dedicated class, but most adapters are just
configured instances of the generic RESTAdapter. This allows us to implement
new adapters just by creating a new adapter spec YAML file. Spec files that omit
the `class` property will use `RESTAdapter` by default.

Our list of `Adapters` is created by enumerating the spec files in the `rest_specs`
directory and constructing an adapter instance for each one. Each adapter instance
will have a class which inherits from the `Adapter` base class. Adapters define
an authentication scheme including referencing a set of "auth variables" that
must be supplied by the Connection to configure the adapter.

A `Connection` represents an active, authorized connection to a cloud system. A
connection has a name which is also used as the schema name to organize the tables
that pull data from that system. Each Connection references the Adapter which is
used to talk to the source system. The Connection supplies account-specific authentication
information to the Adapter.

There is a singular `Connection` class whose instances represent the list of active
connections. These instances are configured via the `connections.yaml` file. The configuration
also supplies values for the "auth vars" needed by the adapter. These can either be
hard-coded values or references on env vars.

We don't ever use "Connector" to avoid confusion!

Each Adapter represents the data sets that it can produce via the `TableDef` class.
So we ask the adapter for its "virtual tables" via `list_tables` and get a list of
TableDefs back. To load the table (pull data from the underlying API and populate
our local db) we use the `TableLoader` class from the `loading` module. This class implements
re-usable logic for loading data from APIs, where the adapter's TableDef is responsible
for talking to the specific API.

Most connections will re-use our RESTAdapter.

## Background tasks

Loading tables can take a long time. However, the process can also be error-prone, and running
loading jobs async is much harder to debug.

Therefore by default the interpreter runs loading jobs in the command process, but on a
background thread. The command process waits by default and shows a progress bar while
the lob job is running. The user can "push" the job to background in which case the command
process simply stops waiting for the loading thread.

This works well, and allows the user to load multiple tables simultaneously. 

    command process 
          | starts loading thread
          |    wrapped by a ProgressBar    -> loading thread
          |                                 |   starts loading data
          |    waits on thread       ->     |
          |    <User escape>                |
          | <- return to command prompt     |   data continues loading

However, we still want "background loading" for table refresh jobs. So we can run
the loader process as a separate daemon. The daemon simply cycles through all 
tables and attempts to keep them up to date. The daemon also observes the current
system load and tries to only run load jobs when load is low.

## Loading log

The system maintains two tables which record an audit history of table loading.

`information_schema.loading_jobs` - this table keeps a log of table creation+loading
jobs. Long running jobs create a _start and _end record.
    id 
    parent_id   (links _end records to _start records)
    timestamp
    table_schema
    table_name
    adapter_name
    action (create, load_start, load_end, refresh_start, refresh_end, truncate, drop)
    numrows
    status (success, error)
    error_message

`information_schema.loading_log`
    loading_job_id - reference to a loading_jobs record
    timestamp
    table_schema
    table_name
    adapter_name
    message
    level (matches python.logging levels)
