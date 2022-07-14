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

## Jupyter intergration

Unify integrates with Jupyter as a "kernel", as implemented
in the [unify_kernel](./unify_kernel) directory.

The [kernel](./unify_kernel/kernel.py) file implements a class which supports execution of SQL script against the Unify database.

Install Jupyter, and run `jupyter-lab` to open the web interface.

Install the Unify kernel for development (sometimes have to do this when env restarts):

    jupyter kernelspec install ./unify_kernel

To test with the Jupyter console:

    jupyter console --kernel unify_kernel

### Autocomplete

The jupyter kernel implements autocompetion hints for schemas,
tables and columns. Generally we can guess that if the user hits
tab right after a ".", then we should suggest a table name. 

Without a preceding period then we try to guess the command by looking for the root command word

Take these examples:

    show <tab>
        [schemas, tables, columns]

    show tables for <tab>
        [schemas]

    show tables for sc<tab>
        [schemas matching prefix "sc"]

    show columns for <tab>
        [qualified tables]

    show columns for sch1.<tab>
        [tables in sch1]

    show columns for sch1.us<tab>
        [tables matching "us*" in sch1]

The most complex example is a SELECT statement, since we want to autocomplete for both table references and column references.

To do this, we implement a few grammar rules that match an incomplete query, so that we can infer which info has been provided so far.

    select <tab>
        [suggestions are all column first letters]

    select us<tab>
        [suggests any column with the given prefix]

    select * from <tab>
        [all qualified tables]

    select * from g<tab>
        [schemas matching "g" prefix]

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