# Unify

Unify is an experiment in building a "personal data warehouse". It integrates Extract-Transform-Load,
plus analysis, into a single app and database which runs on your local machine.

The primary interface to Unify is a command interface which mixes standard SQL and
meta commands. Use `select` to query data, but you also have commands available for
easily importing and exporting data, sending email, and drawing charts.

Unify includes a set of *connectors* to popular SaaS systems, and allows you to easily
create new connectors to systems with REST APIs via a simple configuation file.

Connectors automatically flatten JSON responses into relational tables that are easy
to query with SQL.

Unify offers a columnar store 'database backend' (either DuckDB or Clickouse) which can
efficiently store and analyze tens of millions of rows of data.

## Example

After creating a connection to the Github API, you can query your list of repos:

    > select id, name, owner_login, open_issues_count from github.repos;
    id                                  name owner_login  open_issues_count
    51189180                           philo   tatari-tv                 74
    51713527                        pytrends   tatari-tv                  0
    63492894                        philo-fe   tatari-tv                 30
    67756418                     grey-matter   tatari-tv                 34

## Getting started

Install Unify:

    $ pip install unifydb

And run:

    $ unify

When you first run you need to choose your database backend. DuckDB is simpler to get started with,
but doesn't handle access from multiple processes well. Clickhouse is a little more work to setup,
but works a lot better with other tools like BI tools.

All configuration data is stored into `$HOME/unify`.

Checkout the [tutorial](docs/TUTORIAL.md) to get an overview of using Unify to work with your data.

## Learning more

* Read about the list of [current connectors](docs/ADAPTERS.md).
* Learn about [building](docs/BUILDING_ADAPTERS.md) new connectors.
* Get an overview of [all commands](docs/SQL_LANGUAGE.md).

## Developing

Make sure to setup the environment with:

    export $(cat .env)
