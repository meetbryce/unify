# Multi-tenancy

We anticipate handling multi-tenancy in three forms:

1. When using DuckDB we will create a separate database file for each tenant.

2. For efficient multi-tenancy with Clickhouse we will package multiple tenants onto
the same database server instance.

3. For enterprise cases we can devote a dedicated Clickhouse server.

## Clickhouse multi-tenant

This case is complicated by the fact that a Clickhouse server only supports a single
level of "database" scope. There is no notion of separate schemas or catalogs.

So our work-around is to pack ALL tables for a single tenant within a single
Clickhouse database (schema). We use a prefix naming scheme to encode the logical
schema into the table name.

So if we have 2 connections, github and jira, with tables:

    github.users
    github.orgs
    jira.issues

Then in the db we will create a single database, `tenant1`, and place all tables
in this database:

    github___users
    github___orgs
    jira___issues

Now during SQL parsing we have to map "schema dot" references to table name
prefixes:

    select * from github.users

becomes:

    select * from tenant1.github___users

Similarly we need to map all informational queries, like "show schemas" and "show tables"
to observe this mapping scheme.

The problem with this approach is that ALL queries have to be rewritten on input and output,
including if someone wants to connect directly to the Clickhouse server. For development
expediancy it would be better if we can use Clickhouse "as is" without having to resort to
naming schemes.

So, we could choose one of:

1. Pack all user tables into a single database, and use a user-visible prefix to "namespace"
the tables within a system, like:
   github_users
   github_orgs
   jira_issues

Obviously with enough systems connected this will get pretty messy. Also makes something like
"show tables from github" needs to observe the prefix. However, this approach would work and
is easy to implement.

2. Let each user have a dedicated database per system. The problem is that database names
have to be unique, so only one user on a system can use "github" as the schema name. In this
case we probably need a per-customer prefix to keep the database names unique:

    tenant1_github
    tenant1_jira
    tenant1_salesforce
    tenant1_files

This leads to queries like:
    select * from tenant1_github.users

which is pretty gross.

3. Adopt a "table batch" approach. This would be a compromise where a given tenant
could own multiple databases, and stripe their tables across them. But again the
databases would need "constructed" names:

    tenant1abc01
    tenant2abc02
    tenant3abc03

Not clear that querying "select * from tenant1abc01.users" is any better.

At the end of the day I think approach #1 is the best for now. Name prefix all the
tables with the system name. Maybe for command line usage we can support some
syntactic sugar like "show tables from <system>" which understands the prefixes.

### System tables

We have various needs to store "system" information inside the database. Generally
we will do this by creating similar "virtual" schemas in the tenant database
and storing our data there. This makes these tables visible and useful by the tenant
and avoids us having to separately segregate tenant information.

### The `meta` schema

The `meta` schema is where we store special system tables that are used by Unify, like
tables to store materialized query results.

### The `information` schema

Most databases support the `information_schema` with a special set of tables that
describes the contents of the database. We will create our own version of those
tables in a virtual schema. These tables will be maintained by the Unify engine
itself.

So after some setup, this is what the user will see in their database:

    > show schemas
    github
    jira
    meta
    information_schema

    > show tables from information_schema
    schemata
    tables

    > select table_name, table_schema from information_schema.schemata

    table_name    | table_schema | connection      | refresh_schedule   | source  | provenance  | comment
    -------------   -------------  ----------------  -------------------  --------  ------------  ----------
    users           github         github            daily at 08:00       YAML from spec
    orgs            github         github            daily at 08:00       YAML from spec
    issues          jira           jira

But if we look at the ACTUAL contents of the Clickhouse server:

    >> show schemas
    informatio_schema
    tenant1

    >> show tables from tenant1
    github___users
    github___orgs
    jira___issues
    information_schema___schemata
    information_schema___tables

    >> select table_name, table_schema from information_schema.schemata

    table_name                      | table_schema
    -------------------------------------------------
    github___users                    tenant1
    github___orgs                     tenant1
    jira___issues                     tenant1
    information_schema___schemata     tenant1
    information_schema___tables       tenant1
    schemata                          information_schema
    tables                            information_schema

### Rationale

This may seem like a lot of work to maintain our information tables, but it gives
us a good place to add Unfiy specific capabilities like supporting descriptions for
all tables and supporting "provenance" information. We can just keep extending
the `information_schema`.





