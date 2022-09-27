# Notebooks

Unify cribs the concept of "notebooks" as executable scripts from Jupyter. But we introduce our
own notebook format for internal storage.

A notebook is essentially just an ordered list of cells. Each cell has a "type" and so
we can support both executable command cells as well as markdown cell for annotation.
We also support the notion of inlining the most recent results of a cell execution.

## Execution

Unify natively supports the execution of a notebook by providing the list of cells or
a serialized version of the notebook.

### Scheduled exection

Unify supports running notebooks either interactively or asynchronously. The basic
asynchronous case is to run the notebook on a periodic schedule.

### Dependent execution

In practice simple scheduling isn't super useful. Instead what we want is to execute
a notebook based on some trigger, which will generally be the fact of some table
being updated.

So the basic form of dependent execution is to run a notebook after some external service
table has been updated:

    run after update on github.pulls

now when the `github.pulls` table is updated, our notebook will be executed immediately after.
Note that this logic is implemented at the _system_ level, so no database triggers are created
and raw database-level updates will not trigger our notebook. We support the system `signal`
command to allow a user to indicate an update has occurred.

### Chaining notebooks

We can also indicate that a notebook should execute after some other notebook has run:

    run after notebook 'update_costs'

This instructs to run the current notebook whenever the "update_costs" notebook runs
non-interactively. In this way we can chain dependent notebook executions together.
More practically it means that a complex notebook sequence can be broken up into
small notebooks.

We can also indicate dependence on multiple upstream notebooks:

    run after notebooks 'enrich_cost_data', 'update_costs'

In this case our notebook will only run after both upstream notebooks have completed.

