## Testing

Tests are in the [tests](./unify/tests) folder. They use `pytest`.
They may need some environment variables to be set. To be
compatible with VSCode I keep them in a local .env file and
use:

    export $(cat .env)

To source them into my environment.

Simply run `pytest` to run the tests.

## Parser

The parser uses the `lark` package. Find the grammer in [grammark.lark](grammar.lark). Tests for the gammar are in [unify/tests/test_parser.py](unify/tests/test_parser.py).


## Class model

An `Connector` is the manager object which implements a connection to a particular
cloud system. Most connectors re-use the generic `RESTConnector` class and configure
an instance from their yaml file. 

It is possible to implement new connectors in code by implementing the 
[Connector](./unify/connectors.py) and `TableDef` interfaces. 

Each Connector represents the data sets that it can produce via the `TableDef` class.
So we ask the connector for its "virtual tables" via `list_tables` and get a list of
TableDefs back. To load the table (pull data from the underlying API and populate
our local db) we use the `TableLoader` class from the `loading` module. This class implements
re-usable logic for loading data from APIs, where the connector's TableDef is responsible
for talking to the specific API.

Our list of `Connectors` is created by enumerating the spec files in the `rest_specs`
directory and constructing an connector instance for each one. Each connector instance
will have a class which inherits from the `Connector` base class. Connectors define
an authentication scheme including referencing a set of "auth variables" that
must be supplied by the Connection to configure the connector.

A `Connection` represents an active, authorized connection to a cloud system. A
connection has a name which is also used as the schema name to organize the tables
that pull data from that system. Each Connection references the Connector which is
used to talk to the source system. The Connection supplies account-specific authentication
information to the Connector. In this way we can have multiple **connections** to the
same type of system if desired.

Most connections will re-use our RESTConnector.

## Background tasks

Loading tables can take a long time. However, the process can also be error-prone, and running
loading jobs async is much harder to debug.

Therefore by default the interpreter runs loading jobs in the command process, but on a
background thread. The command process waits by default and shows a progress bar while
the lob job is running. The user can "push" the job to the background in which case the command
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
    connector_name
    action (create, load_start, load_end, refresh_start, refresh_end, truncate, drop)
    numrows
    status (success, error)
    error_message

`information_schema.loading_log`
    loading_job_id - reference to a loading_jobs record
    timestamp
    table_schema
    table_name
    connector_name
    message
    level (matches python.logging levels)
