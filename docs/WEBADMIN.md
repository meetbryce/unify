# Unify Admin

The Unify Admin web app is built as a new front-end to the Unify database to replace
the use of Jupyter. The new UI should be easier to use, more powerful, and built for
purpose to use with Unify.

## Architecture

UnifyAdmin is a centralized web service. It keeps track of all tenants and users and
manages authn and authz. It also implements its own data model for users to code
ELT scripts, reports and Dashboards. Each Admin Tenant is paired with a backend
Unify database. The Admin interacts with the Unify interpreter to run commands against
the database and display the results to the user.

           /---------------\
           |Admin          |
           |  Tenant       |
           |   User        |
           |     Notebook  |
           |        ------------------------> /-------------------\
           \------|--------/                  | Unify interpreter |
             ( Admin database)                | (tenant database) |
                                              |          -----------------> Clickhouse or DuckDB database
                                              |                   |
                                              \-------------------/