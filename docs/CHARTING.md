We have converted to using [Altair](https://altair-viz.github.io/) for charting instead
of Matplotlib. It looks a lot nicer.

Only downside is that Altair is Javascript based. In interactive use this isn't a problem
but when rending server-side for email we have to use a headless browser instance to
render the chart. This works but is quite slow.

## Setup

Setting up for Altair requires installing a bunch of pieces:

    pip install altair
    pip install altair-saver==0.5.0
    # install NodeJS
    npm install package-lock.json

## Creating charts

Use the `create chart` command to build a chart.

    create chart [from <var or table>] as <chart type> where x = <col> and y = <col> [...more options]

### Adding a trendline

Use the `trendline` parameter to add a trendline to the chart. It can take a value of 'average', 'mean',
'rolling' or a fixed value:

    create chart as bar_chart where y = total and trendline=average

Shows the average value of "total" as a trendline.

    create chart as bar_chart where y = total and trendline=50

Shows a fixed value line at "50" on the y axis.

### Creating stacked charts

You can use multiple `create chart` commands to create separate charts that will appear in
series. Or you can plot charts together over the same axis. To create a compound chart
use this syntax:

    create chart as (
        create chart from ...,
        create chart from ...
    )

## Saving charts

When we save a chart, we could save the image for display later, or we could save the 
chart definition and evaluate that later. If the definition refers directly to a table,
then that's fine and we can query the table at the time the chart is rendered. If the
defintion refers to "last chart" then we won't have that. So we can reserve that for
"anonymous charts" which you can't render later.

If the chart refers to a $ variable then either it's a materialized result, which means
we can just query the table when we render the chart, or its a "local" variable. In
that case we can either error because the variable isn't defined, or we can run the
query which defines the variable. This would be preferable as long as we can do it in
a chain, like:

   $users = select id, name, age from users where created > (now() - interval 1 week)
   $counts = select count(name), age from $users group by age

   create chart users_by_age from $counts as bar_chart where x = age

In this approach we would have to persist the query definition of $counts, and the
query definition of $users, and evaluate each in turn. That is either a cool feature
or we should just evaluate the whole notebook in that case...
