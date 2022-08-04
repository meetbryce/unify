### Integration with Google sheets

Unify integrates to read and write data with Google Sheets.

To export a query to a Gsheets file, use this syntax:

    > select * from orders >> gsheets:<file name or sheetId>[/<tab name>]


To import from Gsheets, configure a Gsheets connection and use the custom
`gsheets` command to import data from your spreadsheets:

    > gsheets list files
    ...lists all Gsheet files
    > gsheets search <query>
    ...searches for Gsheet files whose title matches the query
    > gsheets info <file name or gsheet Id>
    ...lists the tabs of the idicated gsheet file
    > gsheets import <file name or gsheet Id> 
    ...imports the first sheet from the indicated Gsheet file. This will create
    a new table in the gsheets connection schema with a name derived from the file name
    > gsheets import <file name or gsheet Id> sheet <sheet name or number>
    ...indicates the specific sheet to import (by sheet name or numeric index starting with 1)
    > gsheets import <file> sheet <sheet> as table <name>
    ...imports the indicated sheet into a table with the indicated table name. If the
    table already exists then data from the sheet will be appended to the table
