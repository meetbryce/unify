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

Unify offloads long-running tasks which load data from connected systems onto a background task
runner. This lets the user continue interacting with the database while data is being loaded.

We don't want to use a simple sub-process of the main process, because then the loader will
die if the user exits the command process. Instead we want a daemon process which is started
automatically and keeps running even if the command process exits. The daemon process needs
to communicate detailed job status back to the command process. We use a simple ZeroMQ
queue to do this.

We use a SQLite database to keep our loading task queue. This allows those tasks to be persistent.
If the loader daemon is offline, we can still queue tasks and they will get processed when
the daemon starts again. The daemon also implements a simple scheduler so that "table refresh"
tasks can get scheduled. The intention is that as long as you let the daemon run then your
tables will automatically get refreshed.

The daemon process uses the multiprocessing package to manage a pool of "loading task" workers.
This allows us to manage the parallelism of the loaders.

So the table loading flow looks like:

    command process 
          | starts daemon            -> daemon process
          |                                 | starts workers  --> worker(s)
    "select * from table"                   |                       |
          | -> enqueue job to sqlite        | poll sqlite           |
          |                                 | grab task             |
          |                                 |   send to worker ->   |
          |                                 |                       | starts loading data
          |                                 |   <-- loading task status (via multiprocessing.Queue)
          | <- loading status (via ZeroMQ)  |                       |

Loader requests are serialized as `LoaderJob` instances. Each job identifies the tenant, the
action type, and the relevant table. The daemon will publish `LoaderStatus` events that reference
a loader job so the Command process can track status.

The action `cancel` can be sent by the Command process to cancel a loading job. This action will
be interpreted by the daemon and the relevant loader job will be canceled.

We have "system" commands in the interpreter for controlling the daemon:

    > system status
    ..Data loading daemon is running
    ..Tasks:
       Loading table: github.pulls
       Refreshing table: jira.issues

    > system stop daemon
    ..Data loading daemon stopping.


