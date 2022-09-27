# Mac

Use homebrew to install, add:

    brew services start clickhouse

Edit config in:

    cd $(brew --prefix)/etc/clickhouse-server/

Create a local user different than 'default':

    CREATE USER scottp IDENTIFIED WITH plaintext_password BY 'egegeeg'

Let that user do anything for ease of use:

    GRANT ALL ON *.* TO scottp WITH GRANT OPTION;

and set the user in the env:

    DATABASE_USER=scottp
    