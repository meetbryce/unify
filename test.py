from tkinter import E
import duckdb
import glob
import pandas as pd
import os
import base64
import pyarrow as pa

if False:
	print(__file__)
	print(os.path.join(os.path.dirname(__file__), "tests"))

	duck = duckdb.connect()

	#duck.execute("create table orgs as select * from read_parquet('./data/github.orgs/*')")
	duck.execute("create schema github")
	duck.execute("create table github.orgs (id VARCHAR, created DATETIME)")
	print(duck.execute("select * from github.orgs").fetchall())
	cols = duck.execute(
		"select column_name from information_schema.columns where table_schema = ? and table_name = ?",
		["github", "orgs"]
		).fetchall()
	print(cols)

	# df = pd.read_parquet(glob.glob('./data/github.orgs/*'))
	# print(df)
	# duck.execute("set search_path='github'")
	# #table = duck.table("github.orgs")
	# duck.append('orgs', df)
	# duck.append('orgs', df)

	# cols = df.columns.tolist()
	# #cols = cols[-1:] + cols[:-1]

	# if cols == df.columns.tolist():
	# 	print("same")
	# else:
	# 	print("different!")
	# df = df[cols]
	# duck.append('orgs', df)

	# print(duck.execute("select * from github.orgs").fetchall())

from redmail import EmailSender
import smtplib
import io
import re

def test_redmail(nb_email):
	email = EmailSender(
		host='smtp.gmail.com',
		port='465',
		username=os.environ['EMAIL_USERNAME'],
		password=os.environ['EMAIL_PASSWORD'],
		cls_smtp=smtplib.SMTP_SSL,
		use_starttls=False
	)

	html = nb_email.get_payload()[0].get_payload()
	image_tags = []
	image_names = ["image1", "image2", "image3", "image4", "image5"]

	def replace_tag(match):
		cid = match.group(1).replace("-", "")
		cid = image_names.pop(0)
		image_tags.append(cid)
		return "{{ " + cid + " }}"

	html = re.sub(r'<img src="cid:(.*)"/>', replace_tag, html)

	image = nb_email.get_payload()[1].get_payload(decode=True)

	email.send(
		subject='Another inline image testing',
		sender="scottp@berkeleyzone.net",
		receivers=['scottp@berkeleyzone.net'],
		html=html,
			body_images={
				image_tags[0]: image,
			}
		)


def convert_nb():
	from nb2mail import MailExporter
	import nbformat
	from nbconvert.nbconvertapp import NbConvertApp
	from email.parser import Parser
	from traitlets.config import Config
	from nbconvert.preprocessors import ExecutePreprocessor

	mail_exporter = MailExporter(template_name="mail")
	notebook = nbformat.reads(open("notebooks/Incident Stats.ipynb").read(), as_version=4)
	ep = ExecutePreprocessor(timeout=600, kernel_name='unify_kernel')

	# Trick is to strip out any email command from the notebook
	for cell in notebook['cells']:
		if cell['cell_type'] == 'code':
			src = cell['source']
			if src and src.strip().startswith('email'):
				cell['cell_type'] = 'markdown'

	ep.preprocess(notebook, {'metadata': {'path': 'notebooks/'}})

	(body, resources) = mail_exporter.from_notebook_node(notebook)
	email = Parser().parse(io.StringIO(body))

	test_redmail(email)

def test_clickhouse_blobs():
	from clickhouse_driver import Client
	import pickle

	settings = {'allow_experimental_object_type': 1, 'allow_experimental_lightweight_delete': 1}
	client: Client = Client(host='localhost', settings=settings)

	client.execute("drop table if exists test5")

	client.execute("create table test5 (id VARCHAR, blob BLOB) Engine=MergeTree() PRIMARY KEY id");
	print(client.execute("select * from test5"))

	data = {"key1": "value1", "key2": "value2"}
	pickled = pickle.dumps(data)

	id = "row1"

	client.execute("insert into test5 (id, blob) values", [{'id': id, 'blob': pickled}])
	rows = client.execute("select id, blob from test5")
	for row in rows:
		id = row[0]
		data = pickle.loads(row[1])
		print(f"{id}: {data}")

	# Create a table from a dataframe

	# Create pandas DataFrame from List
	import pandas as pd
	technologies = [ ["Spark",20000, "30days"], 
					["Pandas",25000, "40days"], 
				]
	df=pd.DataFrame(technologies, columns=["Courses","Fee","Duration"])
	print(df)

	schema = pa.Schema.from_pandas(df)
	col_specs = {}
	for col in schema.names:
		f = schema.field(col)
		if pa.types.is_boolean(f.type):
			col_specs[col] = "bool"
		elif pa.types.is_integer(f.type):
			col_specs[col] = "Int64"
		elif pa.types.is_floating(f.type):
			col_specs[col] = "Float64"
		elif pa.types.is_string(f.type):
			col_specs[col] = "varchar"
		elif pa.types.is_date(f.type):
			col_specs[col] = "Date"
		elif pa.types.is_timestamp(f.type):
			col_specs[col] = "DateTime"
		else:
			raise RuntimeError(f"Unknown type for dataframe column {col}: ", f.type)

	sql = "create table test6 (" + \
		", ".join([f"{col} {ctype}" for col, ctype in col_specs.items()]) + \
			") Engine=MergeTree() PRIMARY KEY " + schema.names[0]

	print(sql)
	client.execute("drop table if exists test6")
	client.execute(sql)
	client.insert_dataframe('INSERT INTO test6 VALUES', df, settings={'use_numpy': True})


test_clickhouse_blobs()


