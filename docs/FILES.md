# Files

Unify provides powerful `import` and `export` commands which make it easy to work with
data in files. 

Examples:

    import https://docs.google.com/spreadsheets/d/16YgB5XykiMBMXQfHk8lI9hYilJ2ctn6madVAJoKt12Q/edit

Imports a Google Sheet.

    import order.csv

Imports a CSV file.

    export github.pulls as 'pulls.csv'

Exports the indicated table to a csv file.


## Use of adapters

File import and export are implemented by Adapters which implement the `create_output_table` and
the `import_file` methods. The Adapter is responsible for reading/writing the files and converting
between DataFrames for saving in the database.

The special `LocalFileAdapter` implements reading and writing to files on the server filesystem.

Other sources of files are supported by other adapters:

### S3

One or more S3 buckets can be mounted into the file system. This makes files in those buckets visible to the
Unify service, and Unify data can be easily written out to S3.

### Google Drive

You can mount one or more folders from Google Drive into the file system.

### Listing files

We support `show files` to show available files

    show files
    
By default this shows files from the LocalFile adapter.

Use:

    show files <adapter>

to show a file listing from another adapter.

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

