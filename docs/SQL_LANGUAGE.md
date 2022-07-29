# SQL Language

Unify implements standard SQL plus a number of extensions for working with connected
systems, adapters, and other features.



## Writing to connected systems

Generally the results of a `select` statement can be written to an output destination
using the `>>` operator.

    > select * from orders >> file: './orders.csv'     // exported as CSV
    > select * from orders >> file: './orders.parquet' // export as Parquet
    > select * from orders >> file: './orders_data as parquet' // export as Parquet
    > select * from orders >> s3: '/bucket1/orders.parquet' // export as Parquet
    > select * from orders >> gsheets 'Monthly orders'
