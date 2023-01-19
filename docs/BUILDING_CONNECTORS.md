# Building Unify connectors

Unify expects most "connectors" to be implemented by the generic [RESTConnector](./unify/rest_connector.py), and configured via the YAML spec file. 

## Building a new RESTConnector

Create a new yaml file called `<system>_spec.yaml` and place it in `$HOME/unify/connectors`.
The easiest approach is to copy an existing connector spec and modify it. The
[papertrail_spec.yaml](./unify/rest_specs/papertrail_spec.yaml) is a good, simple example.

The required parts of the spec definition include:

* **enabled**: Boolean flag. Must be set to `true` or this spec will be ignored.
* **name**: The name of the system we are connecting to. Should be simple as it will generally
get mapped to the database schema.
* **base_url**: The root address for the system's API
* **tables**: The list of tables exported by the connector
  * **-name**: The name of the mapped table
  * **resource_path**: The path to the specific REST API for this data set

That's it. Once you restart Unify it will recognize the new connector. Use the `connect` command
to create a connection. 

With the configured info the connector will make GET requests to the indicated API paths and map JSON data to tables.
Most systems, however, will require authentication. Add the `auth` stanza as described below to
specify authentication.

If API endpoints support paging, you should add the `paging` strategy described below.

This is enough to build a connector that reads data into a set of tables. To get fresh content 
from the system Unify will do a complete reload every cycle. This is sufficient for lots of
cases! But look at the *update stragies* section for information on how to support incremental
updates in your connector.

## Authentication

The RESTConnector supports a number of strategies for supplying an API key or token for authentication to an API. It also supports a configurable Oauth strategy for services that offer Oauth authentication.

We expect that some complex services (such as Amazon AWS) may require custom code to implement their authentication scheme. In this case the RESTConnector can be configured to access a custom library for authentication. This custom library will have to be included a priori in the Unify source code.

Example auth strategies:

    auth:
      type: BASIC
      params:
        username: Github user email
        password: Github personal access token 

    auth:
      type: PARAMS
      params:
        hapikey: HUBSPOT_API_KEY

    auth:
      type: BEARER
      params:
        bearer_token: Hubspot private app token

    auth:
      type: CUSTOM_AUTH

Connector auth parameters are stored in the `unify_connections.yaml` file. 

## Paging strategies

The RESTConnector supports different strategies for paging through multiple results:

    pageAndCount - a page number and count parameters are supplied. Paging continues
    until a result page smaller than the count (page size) is returned.

    offsetAndCount - an offset number and count parameters are supplied. Paging continues
    until a result page smaller than the count is returned.

    pagerToken - Each page includes a "next page" token which should be supplied as a parameter
    to subsquent fetches. The token is defined by a path (`pager_token_path`) into the result doc.
    You should indicate the `token_param` and the `count_param` to use in requests. If you need
    to pass the page token in the POST body, then specify any parameter name for `token_param`
    and then reference that value using ${param} in the `post` dictionary.

Paging options can be specified at the level of the connector spec, or specified individually
on a table spec. Example:

    paging:
      strategy: pageAndCount
      page_param: page
      count_param: per_page
      page_size: 100

    paging:
      strategy: offsetAndCount
      offset_param: offset
      count_param: per_page
      page_size: 100

    paging:
      strategy: pagerToken
      pager_token_path: folder.folder.item
      count_param: limit
      token_param: after
      page_size: 100

## Update strategies

Each API spec can define a "refresh strategy" which indicates how changes to the
system should be queried and merged into the local copy.

**Full Reload**

The default and simplest model is simply to perform a full table load again from
the source system.

    refresh:
      strategy: reload

**Incremental load**

In this model the REST API must support a filter which returns "all changes since
time t". The system will track the timestamp and provide it as a filter for the
next query. The REST resource must also have a unique identifying key. This key is
used to delete the old record before the new recorded is inserted into the database.

    refresh:
      strategy: updates
      params:
        name: <value expr>
    key: <column>

The `params` should refer to query parameters for passing the date filter to the
REST API, using`{timestamp}` to reference the value. For example:
      params: 
        filter: updated_at>{timestamp}
    
**(future) Change data capture**

If the system support webhooks for broadcasting change events, then Unify can subscribe
to webhooks to be notified of changes to any records in the source system.

## The Connector interface

All connectors support this interface, defined in `Connector`:

    name - return the name of the service
    resolve_auth - apply the local connection settings to the authentication configuration
    validate - validate the authentication to the service
    list_tables - return the list of 'available' table definitions for the service
    lookupTable(tableName) - return the table spec for the table with the given name
    supports_commands - returns True if the Connector implements its own special commands
    run_command - execute a command targeted at the Connector

The `list_tables` method should return objects which implement the `TableDef` interface.
This interface declares "potential" tables for which the Connector can return data and which
can be materialized into physical tables by Unify. 

The `TableDef` interface includes:

`name` - The name of the table

`select_list` - Returns a list of fields to request from the service, rather than all

`query_resource` - This method returns a generator which yields the rows of
data from the source system. This method will be called repeatedly to request
all pages of data from the system. For each page, the function should yield
the page of results as a list of dicts, and an empty array which is used
to return the number of rows parsed by Unify. The Connector should compare
this row result with the size of the page requested and generally finish
once an incomplete page is encountered. 

`create_output_table` - This method will be called in order to write results
**to** the connected system. Throw an Exception if writing is not supported
by the Connector. This method will be called once at the start of writing a table,
and it should return an opaque identifier for the output result.

`write_page` - This method is called with each page of rows to write to the
connected system. The output handle from `create_output_table` will be passed
in as well as the source data as a DataFrame.

`close_output_table` - Invoked when all data has been written


## Dynamic tables

By default Unify expects to be able to extract all data from an API resource and
cache it to the local database, then apply the update strategy to retrieve updates later.

However, it can be useful to support a 'dynamic' table definition where we allow
the API to potentially return different results each time it is called. In this case
we don't want to cache the results, but rather invoke the API live every time
the table is referenced.

One use case could be a "Google search" connector, which returns the results of a
Google search query. It's infeasible to retrieve "all" results, so instead we
need to call the API every time a query is made.