Unify expects most "adapters" to be implemented by the generic RESTAdapter, and configured via the YAML spec file. 

## Authentication

The RESTAdapter supports a number of strategies for supplying an API key or token for authentication to an API. It also supports a configurable Oauth strategy for services that offer Oauth authentication.

We expect that some complex services (such as Amazon AWS) may require custom code to implement their authentication scheme. In this case the RESTAdapter can be configured to access a custom library for authentication. This custom library will have to be included a priori in the Unify source code.

## The Adapter interface

All adapters support this interface, defined in `Adapter`:

    name - return the name of the service
    resolve_auth - apply the local connection settings to the authentication configuration
    validate - validate the authentication to the service
    list_tables - return the list of 'potential' table definitions for the service
    lookupTable(tableName) - return the table spec for the table with the given name
    supports_commands - returns True if the Adapter implements its own special commands
    run_command - execute a command targeted at the Adapter

The `list_tables` method should return objects which implement the `TableDef` interface.
This interface declares "potential" tables for which the Adapter can return data and which
can be materialized into physical tables by Unify. 

The `TableDef` interface includes:

`name` - The name of the table

`select_list` - Returns a list of fields to request from the service, rather than all

`query_resource` - This method returns a generator which yields the rows of
data from the source system. This method will be called repeatedly to request
all pages of data from the system. For each page, the function should yield
the page of results as a list of dicts, and an empty array which is used
to return the number of rows parsed by Unify. The Adapter should compare
this row result with the size of the page requested and generally finish
once an incomplete page is encountered. 

`create_output_table` - This method will be called in order to write results
**to** the connected system. Throw an Exception if writing is not supported
by the Adapter. This method will be called once at the start of writing a table,
and it should return an opaque identifier for the output result.

`write_page` - This method is called with each page of rows to write to the
connected system. The output handle from `create_output_table` will be passed
in as well as the source data as a DataFrame.

`close_output_table` - Invoked when all data has been written
