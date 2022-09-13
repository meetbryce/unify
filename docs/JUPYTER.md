## Jupyter intergration

Unify integrates with Jupyter as a "kernel", as implemented
in the [unify_kernel](./unify_kernel) directory.

The [kernel](./unify_kernel/kernel.py) file implements a class which supports execution of SQL script against the Unify database.

Install Jupyter, and run `jupyter-lab` to open the web interface.

Install the Unify kernel for development (sometimes have to do this when env restarts):

    jupyter kernelspec install ./unify_kernel --user

To test with the Jupyter console:

    jupyter console --kernel unify_kernel

### GUI

Instead of creating a custom GUI, we are integrating with Jupyter and Jupyterlab instead.

The basic SQL command line integration is straightforward, except that we want to 
offer intelligent autocompletion for schemas, tables, and columns.

Beyond SQL, we want to make it easy to construct charts, and eventually dashboards,from
the results of queries.

There are many charting libraries for Jupyter, but most of them are extremely complicated.
So for now we are implementing our own charting commands, and internally mapping those
onto MatPlotLib.

    select count(*) as count, user_login from github.pulls group by user_login
    create chart prs_by_user as bar_chart where x = user_login and y = count

    select sum(spend) as revenue, date_trunc('month','timestamp') as month
    create chart rev_by_month as line_chart where x = month and y = revenue

The full chart syntax should look like:

    create chart [<name>] [from <chart source>] as <chart type> where x = <column> and y = <column>

<name> - any identifier
<chart source> - $var, table name, chart name list, or a sub-query in parens
<chart type> - bar_chart, line_chart, pie_chart
<column> - column reference

# incidents chart
alt.Chart(df).mark_bar().encode(x='month',y='count', color='priority',order=order).show()

# AWS costs chart
alt.Chart(df).mark_bar().encode(x='end_date',y='total',color='svc_name').show()

More parameters to the chart can be captured in more k=<val> stanzas in the where clause.

Multiple charts can be combined as:

    create chart from <chart1>, <chart2>

So:
    create chart chart1 as ...
    create chart chart2 as ...
    create chart combo as chart1, chart2
        
### Autocomplete

The jupyter kernel implements autocompetion hints for schemas,
tables and columns. Generally we can guess that if the user hits
tab right after a ".", then we should suggest a table name. 

Without a preceding period then we try to guess the command by looking for the root command word

Take these examples:

    show <tab>
        [schemas, tables, columns]

    show tables for <tab>
        [schemas]

    show tables for sc<tab>
        [schemas matching prefix "sc"]

    show columns for <tab>
        [qualified tables]

    show columns for sch1.<tab>
        [tables in sch1]

    show columns for sch1.us<tab>
        [tables matching "us*" in sch1]

The most complex example is a SELECT statement, since we want to autocomplete for both table references and column references.

To do this, we implement a few grammar rules that match an incomplete query, so that we can infer which info has been provided so far.

    select <tab>
        [suggestions are all column first letters]

    select us<tab>
        [suggests any column with the given prefix]

    select * from <tab>
        [all qualified tables]

    select * from g<tab>
        [schemas matching "g" prefix]