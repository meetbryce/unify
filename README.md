# Unify

Unify is a simple tool which allows you to use SQL to query information from any
cloud based system that supports a REST API. 

System APIs are described in a configuration which models API endpoints as tables.
The JSON payload results from API calls are automatically flattened so that
result fields become columns in the table.

The tables representing the API endpoints for each system are grouped under a database schema.

After data is downloaded via API calls, it is stored to a local database (using DuckDB).

## Example

Querying the list of repositories you have access to on Github:

    > select id, name, owner_login, open_issues_count from github.repos;
    id                                  name owner_login  open_issues_count
    51189180                           philo   tatari-tv                 74
    51713527                        pytrends   tatari-tv                  0
    63492894                        philo-fe   tatari-tv                 30
    67756418                     grey-matter   tatari-tv                 34

## Getting started

Copy [dotenv_template](dotenv_template) to `.env` and edit values to match your environment.

The easiest way to start is by using DuckDB which provides an embedded database. Just
set the `DATABASE_BACKEND=duckdb`.

Now run the command interpreter:

    python -m unify

To add connections to external systems, edit the `unify_connections.yaml` file in your
home or current directory. See [connections_example.yaml](connections_example.yaml) for the format.

## Configuration

REST API configurations are defined in YAML and stored in the `rest_specs` folder.
Consult [spec_doc.yaml](./rest_specs/spec_doc.yaml) for the specification. To
create access to a new system you just need to create a new YAML file describing
the system and restart the server.

## Connections

"Connections" define authentication into a particular account of a cloud system.
You need to define a connection for each cloud system you want to access. That
connection should reference the REST API spec (by name), and provide authentication
details.

The connections file holds a list of connection definitions, that looks like:

    - <schema name>
      spec: <REST API spec file prefix>
      options:
        <key>: <value>
        <key>: <value>

The "spec" field value should match the prefix of one of the "rest_spec" files. That
file will usually reference variables in its `auth` block. You should define values
for those variables in the `options` section of the connection.

You can store sensitive values directly into `connections.yml`, or you can use
`$<name>` syntax to reference an environment variable and store secrets in the
environment instead.

## Variables

Unify extends normal SQL syntax to support `$name` format variables:

    $last_record = <expr>
    $VAR1 = <expr>
    
where <expr> will generally be either a scalar expression, including some literal value, or
the result of a SELECT query.

Examples:

    $current = select current_date
    $maxval = 100

    $last_item = select * from items order by created desc limit 1

Variables that use all upper case letters are automatically treated as globals, meaning
they are globally defined, available, and persistent. Other variables are considered transient
and only live the lifetime a session.

Creating a variable as the result of a select statement is equivalent to creating
a _materialized_ view - the query is evaulated immediately and the results are stored
as the present and future value of the variable. To create a variable which executes the
query each time it is used you should just create a normal `view` using `create view`.

Note that variables are automatically persisted across sessions. Use `unset` to
delete a variable:

    unset $scotts_items

## Exporting data

Some Adapters can support writing results to their connected system. Generally we don't
try to follow `create table` semantics as much as "data export" semantics. This implies
that data is "exported" to the target system as "append only". Typical use cases
are exporting data to files on S3 or Google Sheets. 

See the [SQL LANGUAGE](docs/SQL_LANGUAGE.md) docs for syntax.


## TODO

See [TODO](docs/TODO.md)


### Adapters

- Implement some interesting new adapters: Gmail, Salesforce, S3, GDrive
- Implement YAMLSchema validation for adapter specs
- A Web based "adapter builder" for interactive building/modifying of adapter specs
- Generalizeed Oauth support

### Interpreter

- Proper command/table/column completion

### Web interface

- Build a centralized web admin front-end for interacting with the interpreter
- Data exploration interfaces
    - Create connection interface
    - Schema browser
    - Data "peeking" (both tables and API endpoints)
    - Adapter builder
    - Command console
- Proprietary "notebook" support. A notebook is a similar to Jupyter - a sequential set of cells
where cells are either code or right text. Notebooks also include "input" and "output" metadata.
That is, notebooks automatically describe which tables/vars they depend on for input, and which ones
they generate as output. This allows the user to decompose a complex pipeline into multiple notebooks,
where the system automatically understands the dependencies between the pieces. The user can annotate
a notebook to "auto execute" if any of its inputs change. So if I imported a spreadsheet into a table,
then created a notebook to transform the sheet data into another table, then I could specify that
the notebook should be executed automatically if the original sheet data was updated.
- Implement support for "reports" and "dashboards" either as new first-class objects or specializations
of notebooks. Likely to be the latter. You create a notebook and then say "render this notebook as
a dashboard" which basically extracts and displays only "marked reportable" cells. By default a code
cell which makes a chart will be marked "reportable". But the user could elect to report via tables or
text cells as well.
- Notebook "dev/publish" mode should be very clear. A notebook can only be run automatically in the background
once it is "published" which creates a "hard version" of the notebook contents. This let's someone play
with notebook edits without affecting published notebooks. (Maybe "published" notebooks are commited into
the interpreter, while dev notebooks are only held in the admin layer?)

## Developing

Make sure to setup the environment with:

    export $(cat .env)
    
## Metrics use cases

- Quality metrics
  - Count of P1/P2 incidents by month
  - Tickets opened vs. closed
  - Revenue chart
  - Count of DQ tickets (by tag)
  - Costs
    - 30 day lookback for
      - AWS costs by service
      - Datadog
      - Papertrail

## Running Metabase

Metabase supports Clickhouse, so you can run Metabase and connect it directly to the CH db.
Running Metabase directly from the jar is easy as long as you get the right (OpenJDK) version.

Install `clickhouse.metabase-driver.jar` in the plugins directory and that will let you 
add the Clickhouse connection.
