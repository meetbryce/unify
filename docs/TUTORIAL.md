# Unify Tutorial

This tutorial will walk you through the basics of installing **Unify** and working with its core features
for importing and analyzing data.

## Getting started

    pip install unifydb

Choose your database backend. Unify supports Clickhouse or DuckDB. DuckDB is simpler to use
as it runs in the main process. However, if you want to connect a BI tool to your warehouse
then Clickhouse will be better supported.

Set `DATABASE_BACKEND` in your environment:

    export DATABASE_BACKEND=duckdb

or if you installed Clickhouse:

    export DATABASE_BACKEND=clickhouse
    export DATABASE_HOST=localhost
    export DATABASE_USER=default
    export DATABASE_PASSWORD=""

## Tutorial

Start Unify:

    $ python -m unify
    Welcome to Unify - your personal data warehouse. Use 'help' for help.
    >

Get an overview of available commands:

    > help
    help - show this message
    help schemas - overview of schemas
    ...

Let's start by adding some data to the warehouse. Let's import some CSV data that contains information restaurant menu items:

    > import https://github.com/scottpersinger/unify/blob/3914c19bc723ddfca3a0d4ae7f7a8219b9ed3c6c/sample_data/menu_dishes.csv

[TODO: import by url]

The data import happens in the background, but the first batch of data gets echoed to the screen:

     id                             name  menus_appeared  times_appeared  first_appeared  last_appeared  lowest_price  highest_price
      1       Consomme printaniere royal               8               8            1897           1927          0.20           0.40
    2                    Chicken gumbo             111             117            1895           1960          0.10           0.80
    3              Tomato aux croutons              14              14            1893           1917          0.25           0.40
    4                  Onion au gratin              41              41            1900           1971          0.25           1.00

Now let's explore the data a little bit:

    > count files.table_dish
    1 row
        count_
        428146

```sql
> show columns from files.table_dish
    8 rows
    column_name column_type default_type default_expression comment codec_expression ttl_expression
    first_appeared    Int64                                                                        
     highest_price    Float64                                                                        
                id    Int64                                                                        
     last_appeared    Int64                                                                        
      lowest_price    Float64                                                                        
    menus_appeared    Int64                                                                        
              name    String                                                                        
    times_appeared    Int64                     
```
The `import` command has loaded our CSV into a new table. It places the table under the `files`
schema because the FileAdapter was used to load the data.

The column names and types were inferred from the data in the CSV file. We can use any SQL to analyze
the data:

```sql
> select name, highest_price from files.table_dish order by highest_price desc limit 1
1 row
                            name  highest_price
Cream cheese with bar-le-duc jelly        3050.00
```
Apparently the highest priced item in our data set is the $3000 cream cheese! 

## Creating connections

The `files` adapter is automatically configured, but now lets connect to a real API and load some more 
interesting data. Unify comes with a set of **adapters** for pulling table from different systems.

The site [Public APIs](https://publicapis.dev/) publishes a simple API which returns their curated list
of publicly usable system APIs.  

We have created a simple adapter for querying from this API. To use it, you first need to create
a **connection**, which configures the adapter in your environment and binds it to a `schema` in your
database:

```sql
> connect
1: aws_costs
2: coda
3: datadog
4: publicapis
...
```
Choose the `publicapis` adapter and finish the setup:
```
Pick an adapter: 4
Ok! Let's setup a new publicapis connection.
Specify the schema name (publicapis):
Please provide the configuration parameters:
Testing connection...
New publicapis connection created in schema 'publicapis'
The following tables are available, use peek or select to load data:
1 row
table_name table_schema comment materialized
 entries     publicapis    None            ☐
>
```

Because this adapter is so simple, there is no additional configuration other than specifying the adapter. 
But for most adapters you will need to provide at least authentication credentials.

Once the connection is created, then the schema will be created in your database, but it will be empty:

```sql
> show schemas
files
publicapis
```
However, if you use `show tables` you will see the **available** tables from the adapter:
```sql
> show tables from publicapis
table_name table_schema comment materialized
 entries     publicapis    None            ☐
```
The *materialized* column shows that the `entries` table is available but no data has been loaded.
If we `select` from the table then it will automatically load the data from the API:

```sql
> select * from publisapis.entries
Loading table...
    API                                        Description          Auth  HTTPS    
    0x API for querying token and pool stats across va...                    1   
    18F   Unofficial US Federal Government API Development                   0 
    1Forge                      Forex currency market data        apiKey     1 
...
```
and now we can inspect the table loaded with data from the API:
```sql
> show columns from publicapis.entries
7 rows
column_name column_type default_type default_expression comment codec_expression ttl_expression
        API      String                                                                        
       Auth      String                                                                        
   Category      String                                                                        
       Cors      String                                                                        
Description      String                                                                        
      HTTPS       UInt8                                                                        
       Link      String                                                                        

> select * from publicapis.entries where API ilike '%JIRA%'
1 row
 API                                        Description  Auth                                  Link                 Category
JIRA JIRA is a proprietary issue tracking product th... OAuth      1 unknown https://developer.atlassian.com/server/jira/pla... Documents & Productivity
```
```sql
> select count(*) as count, Auth from publicapis.entries group by Auth order by count desc
5 rows
 count          Auth
   669              
   600        apiKey
   149         OAuth
     6 X-Mashape-Key
     1    User-Agent
>
```
Once we create a query with some interesting results we work with it easily.
We can *export the table* to a file (or Google Sheet if we create a GSheet connection):
```sql
> export publicapis.entries to files 'api_entries.csv'
> show files
api_entries.csv
```
Or we can *email the data* to ourselves (this requires SMTP config):
```sql
> email publicapis.entries to 'scottpersinger@gmail.com'
Sent data to scottpersinger@gmail.com
```
Or we can even draw our data as a chart:
```sql
> create chart as bar_chart where x = Auth and y = count
```

### Graphical analysis

Unify has only primitive built-in charting features. But if you use Clickhouse as your database 
backend then you can connect lots of popular Business Intelligence tools to the database.

The easiest way to start is to use [Metabase](https://www.metabase.com/) - a popular open source
BI tool.

Unify makes it super easy to try it out:
```sql
> open metabase
Do you want to install and run Metabase (y/n)? y
Downloading Metabase
 100.0% [======================================================================================================================>] 265267/265266 eta [00:00]
Downloading Clickhouse driver
 100.1% [==========================================================================================================================>] 1318/1317 eta [00:00]
Please enter info to create your local Metabase account:
Enter your email address: scottp@example.com
Choose a Metabase login password: *****
Confirm the password: *****
Metabase setup succeeded
```
**Exploring your warehouse in Metabase**

![metabase screenshot](metabase_screenshot.png)

## Learning more






