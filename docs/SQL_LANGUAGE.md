# SQL Language

Unify implements standard SQL plus a number of extensions for working with connected
systems, adapters, and other features.

## Help and information
 
    help
    help [info, schemas, charts, import, export]

    show schemas
    show tables
    show tables from <schema>
    show columns from <schema>.<table>
    show columns from <schema>.<table> like 'pattern'
    describe 
    describe <schema>.<table>
    peek <schema>.<table>
    
## Standard language

    select [column refs] from ...
    create table <schema>.<table> ...
    create table <schema>.<table> ... as <select statement>
    create view <schema>.<table> ... as <select statement>
    insert into <schema>.<table> ...
    delete from <schema>.<table> [where ...]

    drop table <schema>.<table>
    drop schema <schema> ["cascade"]

## Convenience syntax

    > ??

    Describe the columns from the last referenced table.

    > select * from  ??

    Short-cut to reference the most recent table.

    > count <table>

    Short-cut for 'select count(*) from <table>'

## Peeking at tables

The special `peek` command makes it easy to see "interesting" data from a table without having
to know columns names ahead of time. It is similar to using "select * from..." except that
it applies some heurists to determine which columns to show:

    > peek github.pulls
    id     number       title             user_login
    ------ ------------ ---------------   --------------
    a234   7002         Update the Readme scooter

When data is first loaded into a table, the columns of the table are analyzed to determine
the "peek" columns using these heuristics:

- shorter named columns 
- shorter width text columns
- preference dates
- preference enums
- don't show URLs
- don't show floating point values

Peek assigns weights and expected widths to each column. When you peek at the table it
picks the highest weighted columns and attemps to fill (but not overfill) the page width.

You can also use `peek*` as a special syntax in select statements to include the peek
columns plus additional ones:

    > select peek*, base_repo_url from github.pulls

## Charting

    create chart [<name>] [from <chart source>] as <chart type> where x = <col ref> and y = <col ref>

## Importing data

Generally importing data from connected systems is implicit. The system definition will define a set of logical tables that will appear inside the schema for the system. Querying any table will cause the data for the table to be imported from the connected system.

Some systems, like Google Sheets, have special commands for importing data. Use `<schema> help` to learn about the commands.

### Importing file data

You can import flat file data using the `import` command:

    import '<file url>' [into <schema>.<table> ["overwrite"|"append"]]

This command will create the indicated table according to the schema of the source file, which should be in
csv or parquet format. If the table exists then this command will return an error unless
you specify either the `overwrite` or `append` option.

The file url can either be a local path or an S3 file URL. Whether the file contains CSV or Parquet
format will be automatically detected.

## Writing to connected systems

    export <schema>.<table> to [adapter] 'file name'|expr ["overwrite"|"append"]

This will export all rows of the indicated table to the connected system specified by the "adapter" name. Only certain connected systems support exporting data. Use the "overwrite" option to allow overwriting an existing file
and its contents. Use the "append" option to append to any existing file.

    export hubspot.orders to s3 '/bucket1/order.csv'
    export hubspot.orders to s3 '/bucket1/order.parquet'

    export hubspot.orders to file '/tmp/orders.csv'
    export hubspot.orders to gsheets 'Hubspot Orders'

The file name argument can also be any expression enclosed in parenthesis. This allows constructing
the target file name dynamically:

    export hubspot.orders to gsheeets ('Hubspot Order as of ' || current_date)

## Email

You can request to render a particular chart or a whole notebook into an email:

    email <"notebook"|chart name> to '<recipient list>' [every "day"|"week"|"month"] starting at <DD-MM-YYYY HH:MM>]
       [subject <'' or expr>]

Running `email notebook to` will send the email immediately. If you specify `every ...` then a schedule
will be created to execute the full notebook and send the email according to the schedule.

## Scheduled tasks

You can use the `run` command to execute a notebook in the future:

    run at '2022-10-01 08:00'

Schedules to execute this notebook once at the indicated day and time.

You can also run the notebook periodically:

    run every ['day','week','weekday', 'month'] starting at <date time>

This syntax creates a schedule to run the notebook periodically, starting at the
indicated day and time. For convience you can use a shorthand for the
starting time: `now` or `tomorrow`.

The date-time value can either by a date, a time, or both:

    2022-08-10
    16:30
    2022-08-10 14:30

If the time is omitted then "current time" is assumed.

Examples:

    > run every day starting now

Will immediately execute the notebook and schedule it to run at the same time each day.

    > run every week starting 2022-10-05

Will run the notebooke every week starting on Oct 5, at the current wall clock time.

    > run every month starting 2022-11-01

Run the notebook on the first day of every month starting November 1.

To see the list of notebook schedules, use:

    > run schedule
    id           time              repeat        notebook
    ------------ ----------------  ------------  --------------
    05832        2022-10-02 08:00  daily         'Latest PR list'

and to delete a schedule:

    > run delete 05832



