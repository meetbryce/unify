# Mac

**Don't install from Homebrew as the first is very old.**:

Install Clickhouse per their instructions: https://clickhouse.com/docs/en/install/#self-managed-install

Edit ~/unify/unify_config to set your Clickhouse connection parameters:

    DATABASE_BACKEND=clickhouse
    DATABASE_HOST=??
    DATABASE_USER=??
    DATABASE_PASSWORD=??
