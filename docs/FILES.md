# Files

Unify attempts to provide a "unified" file system which co-exists alongside the integrated database.

This file system is available to all Unify commands. It has a physical backing in the Unify server
space, but other filesystems can be "mounted" into it.

These include:

### S3

One or more S3 buckets can be mounted into the file system. This makes files in those buckets visible to the
Unify service, and Unify data can be easily written out to S3.

### Google Drive

You can mount one or more folders from Google Drive into the file system.

### Local files

Unify provides a local application which mirrors local files into the file system. It allows you
to mount a folder from your local system into the Unify file system, and to read and write files
in the folder. This allows an easy path for importing file data into Unify, or for exporting
Unify data back to your local system.

## Using files

(See here for help: https://stackoverflow.com/questions/33713084/download-link-for-google-spreadsheets-csv-export-with-multiple-sheets)

The simplest way to import a file is to use the `import` command and a URL:

    > import https://somecoolhost.com/data.csv

The system will attempt to download the file and infer its data format, then it will import the
data into a local table under the files schema:

    ...created 'files.data_csv' table

This also works with Google Sheets URLs:

    > https://docs.google.com/spreadsheets/d/1sfkehaHQ2IOGNVOZQjyDF_6mCGO0ZEBru5WBsQU6_D8/edit#gid=0

If a GSheets connection is available we will try to download the file using the API and the table
will be created in the gsheets schema. If that fails then we will fall back to using the "Chart tools
protocol" and attempt to download the sheet with no authentication. This will only work if the sheet
has been made publicly accessible.

### The FILE function

We generalize access to files using the special `FILE` table function. You can read or write
data from a "file table":

Reading files:

    select * from FILE('test_data.csv')
    insert into orders select id, product, amount from FILE('/inbox/orders.parquet')

Writing out to files:

    insert into FILE('lastest_prs', 'csv') select * from github.pulls where created_at = today()

You can specify the file format with the second argument to the function. If omitted then the
function will attempt to guess the right file format based on the file name extension or the
file contents.

### Listing files

We support the `ls` command similar to a Unix prompt:

    ls           # lists all files in the root directory. This will show all the
                 # mounted file system folders
    ls '/inbox'  # lists files in the /inbox directory
    ls *.csv     # uses file globbing to list matching files

and the `rm` command for deleting files matching a file name or pattern

    rm /inbox/*.csv

## Mounting file systems

For now you have to specify file mounts in the `connections.yaml` config file:

    - filemount:
        adapter: s3
        path: /s3
        options:
            BUCKET_NAME: unify.bucket1
            AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID
            AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY
            AWS_DEFAULT_REGION: $AWS_DEFAULT_REGION
    - filemount:
        adapter: local
        mount: /{user}

# Implementation

Our `FileSystem` class implements the virtual filesystem. It gets configured with a set
of `FileAdapters` which provide file interfaces to specific backends.

By default the server will add the `LocalFileAdapter` configured with a root unique
to the current tenant. 

For now we are using 'unison' to power the local file mounting. We will automatically
run unison locally and ssh into the DATABASE_HOST box, and sync $HOME/unify/data between
local host and server.

Additional file systems can be mounted by configuring additional file adapters in the
config file.

## Installing Unison

Unison is a simple-to-use sync program written in ocaml. It can watch a directory
and keep it synchronized between client and server. The biggest hassle is that it has
to be the same version running on both ends. And you also have to get the unify-fsmonitor
program installed on both ends to be able to watch for file changes. Once you have that
then running unison is pretty easy:

unison local_path ssh://unifyserver16/remote_path -batch -repeat watch

This says to sync local_path over ssh with remote_path, always auto-answer any conflict
prompts (-batch), and to use the filesystem watcher to watch for changes.

