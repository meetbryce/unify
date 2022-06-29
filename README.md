# Unify

Unify is a simple tool which allows you to use SQL to query information from any
cloud based system that supports a JSON-result REST API. 

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
