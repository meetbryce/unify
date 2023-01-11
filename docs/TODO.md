## OSS readiness

1. [done] Convert to poetry
1. [done] Add "connect" interactive flow
1. Create proper install package
1. Create a basic usage tutorial
1. Add background task processing
   - Persist log messages and support "show logs ..." command to view them
1. Add command completion based on current schemas and tables

### FileAdapter

1. Add support for remote URLs and TAR file listing
1. Import remote files, including a JSON API result

## Unsorted

1. [done] Implement unit tests
1. [done] Implement table refresh, with support for strategies
1. [done] Implement scheduled automatic table refresh
1. [done] Implement GSheets adapter
1. Implement AWS Cost Reporting adapter
   - [done] Need to support AWS authentication
   - [done] Support POSTing for REST API calls
   - [done] Support pulling the right values out of the result
   - Implement updates
1. [done] Implement Lark parser for more complex syntax support
1. [done] Implement full `show` commands
1. [done] Implement dollar variables
1. Background table loading status supporting interrupts
1. [done] Pluggable database, plus Clickhouse backend
1. [done] SQL expressions for rest specs
1. [done] Implement "peek <table>" command which automatically selects interesting columns
1. [done] Implement "run [<notebook>] at <schedule>" command
1. [done] Have "show tables" indicate table comment and if data has been loaded from the table yet
1. [done] Automagically show hyperlinks for URL content from tables
1. Start working on "lineage" features that can show which notebook/code produced a table
1. [done] Make sure date/float typing works properly on table load
1. Add S3 adapter
1. Add GDrive adapter
1. [done] Add annotation layer for tables/columns to schema support. 
1. [done] implement CSV and parquest file import/export
1. Fix date handling in charts (requires strings right now)
1. Automatically unpack TAR and compressed files
1. Add support for PRSQL syntax
1. Re-implement Postgres adapter to use EXPORT and COPY instead of native Clickhouse support

## Jupyter integration

1. Schema browser tree panel
1. [done] Custom charting command
1. Implement more autocompletions based on scanning help messages
1. Async responses for slow actions (like emailing)

## Roadmap
