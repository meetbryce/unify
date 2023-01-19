Unify uses [Altair](https://altair-viz.github.io/) for built-in charting.

Only downside is that Altair is Javascript based. In interactive use this isn't a problem
but when rending server-side for email we have to use a headless browser instance to
render the chart. This works but is quite slow.

## Setup

To enable rendering charts into emails:

    > chart install

This installs some Altair packages AND some NodeJS packages for backend rendering.

[TODO: implement install command:
    pip install altair-saver==0.5.0
    # install NodeJS
    npm install package-lock.json
]

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

TODO
