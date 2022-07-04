class Queries:
    list_schemas = "select schema_name from information_schema.schemata"
    list_tables = "select concat(table_schema, '.', table_name) from information_schema.tables "
    list_tables_filter = " where table_schema = '{}' order by table_schema, table_name"
