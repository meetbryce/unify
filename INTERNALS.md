## Testing

Tests are in the [tests](./tests) folder. They use `pytest`.
They may need some environment variables to be set. To be
compatible with VSCode I keep them in a local .env file and
use:

    export $(cat .env)

To source them into my environment.

Simply run `pytest` to run the tests.

## Parser

Unify builds its own command interpreter on top of DuckDB, so that it can offer extended operations and syntax without modifying
DuckDB.

The parser uses the `lark` package. Find the grammer in [grammark.lark](grammar.lark). Tests for the gammar are in [tests/test_parser.py](tests/test_parser.py).

## Jupyter intergration

Unify integrates with Jupyter as a "kernel", as implemented
in the [unify_kernel](./unify_kernel) directory.

The [kernel](./unify_kernel/kernel.py) file implements a class which supports execution of SQL script against the Unify database.

Install Jupyter, and run `jupyter-lab` to open the web interface.

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

