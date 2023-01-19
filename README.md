# Unify

Unify is an experiment in building a "personal data warehouse". It integrates Extract-Transform-Load,
plus analysis, into a single app and database which runs on your local machine.

The primary interface to Unify is a command interface which mixes [standard SQL and
meta](./docs/SQL_LANGUAGE.md) commands. Use `select` to query data, but you also have commands available for
easily importing and exporting data, sending email, and drawing charts.

Unify includes a set of *connectors* to popular SaaS systems, and allows you to easily
create new connectors to systems with REST APIs via a simple configuation file. Connectors 
automatically flatten JSON responses into relational tables that are easy to query with SQL. 

Unify offers a columnar store 'database backend' (either DuckDB or Clickouse) which can
efficiently store and analyze tens of millions of rows of data.

Unify **should** be the easiest way to pull data from a REST API and materialize
it into a database suitable for SQL analysis.

## Example

Establish a connection to the Github API:

    > connect
    1: aws_costs
    2: coda
    3: datadog
    4: github
    Pick a connector: 4
    Ok! Let's setup a new github connection.
    Specify the schema name (github):
    Please provide the configuration parameters:
    username (Github user email): johnsmith@example.com
    password (Github personal access token): ghp_Jik22xkF88wmzzj8xxks2x2jz
    Testing connection...
    New github connection created in schema github
    The following tables are available, use peek or select to load data:
    8 rows
    table_name table_schema                                            comment materialized
        pulls       github                                               None            ☐
        repos       github                                               None            ☐
        users       github                                               None            ☐
    >

After creating the connection, you can query your list of repos:

    > select id, name, owner_login, open_issues_count from github.repos;
    Loading data for table: github.repos
    ...
    id                                  name owner_login      open_issues_count
    51189180                     dep-checker linuxfoundation             74
    51713527                    code-janitor linuxfoundation              0
    63492894                    foss-barcode linuxfoundation             30

## Getting started

Install Unify:

    $ pip install unifydb

And run:

    $ unify

When you first run you need to choose your database backend. DuckDB is simpler to get started with
(and supported on Windows), but it doesn't handle access from multiple processes well. 
Clickhouse is a little more work to setup, but works a lot better with other tools like BI tools.
If you are running on Windows, you can get a cloud Clickhouse instance at [clickhouse.com](https://clickhouse.com).

All configuration data is stored into `$HOME/unify`. Set `UNIFY_HOME` in your environment
if you want to store config data somewhere else.

**Checkout the [tutorial](docs/TUTORIAL.md) to get an overview of using Unify to work with your data.**

## Learning more

* Read about the list of [current connectors](docs/CONNECTORS.md).
* Learn about [building](docs/BUILDING_CONNECTORS.md) new connectors.
* Get an overview of [all commands](docs/SQL_LANGUAGE.md).

## Limitations

* This is **alpha** software. Use at your own risk! Good news is that API connectors only read data.
If you want to examine the code which interacts with your systems, checkout the [RESTConnector](https://github.com/scottpersinger/unify/blob/main/unify/rest_connector.py).
* The schema and table layout are not stable yet, so upgrading Unify may change the database layout.
* Test coverage is poor and bugs are numerous.

## Developing

Tests use `pytest` and the project overall uses `poetry`. 
