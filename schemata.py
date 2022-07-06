# FIXME: We should replace use of this class with
# the programmatic interface to DuckDB queries.

class Queries:
    list_schemas = "select schema_name from information_schema.schemata"
    _ordering = "order by table_schema, table_name"
    _list_tables = "select concat(table_schema, '.', table_name) from information_schema.tables "
    list_tables = _list_tables + _ordering
    list_tables_filtered = _list_tables + " where table_schema = '{}' " + _ordering
