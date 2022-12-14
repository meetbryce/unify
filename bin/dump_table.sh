echo "Make sure to set PGHOST, PGUSER, PGDATABASE and PGPASSWORD in the environment"

psql --pset=format=unaligned -c "\d $1" > $1.schema

echo "Saved schema to $1.schema"
echo `date` " Extracting table with COPY to csv..."
psql -c "\copy (select * from $1) to $1.csv CSV HEADER"
echo `date` " Wrote $1.csv"
