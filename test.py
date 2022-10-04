from tkinter import E
from venv import create
import duckdb
import glob
import pandas as pd
import os
import base64
import pyarrow as pa
import requests
import time
import yaml
import uuid
import logging
from pprint import pprint
from datetime import datetime

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

def test_redmail_image():
	email = EmailSender(
		host='smtp.gmail.com',
		port='465',
		username=os.environ['EMAIL_USERNAME'],
		password=os.environ['EMAIL_PASSWORD'],
		cls_smtp=smtplib.SMTP_SSL,
		use_starttls=False
	)

	email.send(
		subject='Another inline image testing',
		sender="scottp@berkeleyzone.net",
		receivers=['scottp@berkeleyzone.net'],
		html="""
			<h1>Hi,</h1>
			<p>have you seen this?</p>
			{{ myimg }}
    	""",
			body_images={
				"myimg": "costs.png",
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


def test_school():
	from bs4 import BeautifulSoup

	UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
	headers = {"user-agent": UA}
	login = "https://login.jupitered.com/login/index.php"
	params = {"studid1": "Drew Persinger", "text_password":"XpsPezBm78V5",
			"access1":"1", "school1":"11907","city1":"Berkeley", "doit":"checkparent",
			"from":"login","loginpage":"parent", "username1":""}

	s = requests.Session()
	r = s.post(login, data=params, headers=headers)
	print(r.status_code)
	print(r.url)
	time.sleep(0.5)

	soup = BeautifulSoup(r.text, 'html.parser')
	if soup.find(id='mainpage') is not None or True:
		print('Successfully logged in')
		todo = "https://login.jupitered.com/0/student.php"
		data = {"from":"inbox", "to":"todo", "stud":"5321673","school":"11907","year":"20222023"}
		r2 = s.post(todo, data=data, headers=headers)
		print(r.status_code)
		print(r.text)

	else:
		print('Authentication Error')

def test_aws_cost_api():
	logging.basicConfig()
	logging.getLogger().setLevel(logging.DEBUG)
	requests_log = logging.getLogger("requests.packages.urllib3")
	requests_log.setLevel(logging.DEBUG)
	requests_log.propagate = True

	import requests
	from requests_aws4auth import AWS4Auth
	endpoint = 'https://ce.us-east-1.amazonaws.com/'
	auth = AWS4Auth(os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY'], os.environ['AWS_DEFAULT_REGION'], 'ce')
	spec = yaml.safe_load(open("rest_specs/aws_costs_spec.yaml"))

	table = spec['tables'][0]

	headers = table.get('headers', {})

	body = table['post']
	body['Dimension'] = 'SERVICE'

	response = requests.post(endpoint, headers=headers, auth=auth, json=body)
	pprint(response.json())

import pstats
from pstats import SortKey

def profile():
	p = pstats.Stats('stats')
	p.sort_stats(SortKey.CUMULATIVE).print_stats(30)

def test_duck_sqlalchemy():
	from sqlalchemy import Column, Integer, Sequence, String, DateTime, create_engine
	from sqlalchemy.ext.declarative import declarative_base
	from sqlalchemy.orm.session import Session

	Base = declarative_base()


	seq = Sequence("fakemodel_id_sequence")
	class FakeModel(Base):  # type: ignore
		__tablename__ = "fake"

		id = Column(Integer, seq, server_default=seq.next_value(), primary_key=True)
		name = Column(String)
		date = Column(DateTime)


	eng = create_engine('duckdb:////tmp/tesst.db')
	Base.metadata.create_all(eng)
	session = Session(bind=eng)

	m = FakeModel(name="Frank")
	m.date = datetime.utcnow()
	session.add(m)
	session.commit()

	records = session.query(FakeModel).all()
	print([f.__dict__ for f in records])

def test_ch_tunnel():
	import sshtunnel as sshtunnel
	from clickhouse_driver import connect

	server = sshtunnel.SSHTunnelForwarder(
		('unifyserver16', 22),
		ssh_username="xx",
		ssh_password="xx",
		remote_bind_address=('localhost', 9000))
		
	server.start()

	local_port = server.local_bind_port
	print(local_port)

	pw = 'xx'
	#conn = connect(f'clickhouse://default:{pw}@localhost:{local_port}/default')
	conn = connect(host='localhost', port=local_port, database='default', user='default', password=pw)

	cursor = conn.cursor()
	cursor.execute('SHOW TABLES')
	print(cursor.fetchall())	

def test_clicklhouse_sqlalchemy():
	from sqlalchemy import Column, Integer, Sequence, String, DateTime, create_engine
	from sqlalchemy.ext.declarative import declarative_base
	from sqlalchemy.orm.session import Session
	
	from clickhouse_sqlalchemy import engines

	Base = declarative_base()


	seq = Sequence("fakemodel_id_sequence")
	def uniq_id():
		return str(uuid.uuid4())

	class FakeModel(Base):  # type: ignore
		__tablename__ = "fake"

		id = Column(String, default=uniq_id, primary_key=True)
		name = Column(String)
		date = Column(DateTime)

		__table_args__ = (
        	engines.MergeTree(primary_key='id'),
			{"schema": "unify_schema"}
    	)

	#uri = 'clickhouse://default:@localhost/default'
	uri = 'clickhouse://default:@localhost/default'

	eng = create_engine(uri)
	Base.metadata.create_all(eng)
	session = Session(bind=eng)

	m = FakeModel(name="Frank")
	m.date = datetime.utcnow()
	m2 = FakeModel(name="Jane")
	m.date = datetime.utcnow()
	session.add(m)
	session.add(m2)
	session.commit()

	records = session.query(FakeModel).all()
	print([f.__dict__ for f in records])

def test_schemata():
	from unify.db_wrapper import get_sqla_engine, Schemata, UNIFY_META_SCHEMA, Base
	from sqlalchemy.orm.session import Session

	engine = get_sqla_engine()
	engine = engine.execution_options(
    	schema_translate_map={UNIFY_META_SCHEMA: "tenant_scottp", None: "tenant_scottp"}
	)
	Base.metadata.create_all(engine)

	session = Session(bind=engine)

	schema1 = Schemata(name="jira")
	session.add(schema1)
	session.commit()

def test_schemata2():
	from unify import dbmgr
	from unify.db_wrapper import Schemata
	from sqlalchemy.orm.session import Session

	with dbmgr() as db:
		session = Session(bind=db.engine)
		schema = Schemata(name="hubspot")
		session.add(schema)
		session.commit()

def dump_duck():
	def dump_tables(duck):
		df = duck.execute("select * from information_schema.tables").df()
		print(df)

	duck = duckdb.connect('./unify/data/duckdata', read_only=False)
	dump_tables(duck)

	#duck = duckdb.connect('/tmp/duckmeta', read_only=False)
	#dump_tables(duck)

#test_aws_cost_api()
#test_redmail_image()
#profile()
#test_ch_tunnel()
#test_clicklhouse_sqlalchemy()

#test_schemata2()
dump_duck()





