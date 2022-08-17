import duckdb
import glob
import pandas as pd
import os
import base64

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

	html = re.sub(r'<img src="cid:(.*)"', replace_tag, html)

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

	mail_exporter = MailExporter(template_name="mail")
	notebook = nbformat.reads(open("notebooks/Incident Stats.ipynb").read(), as_version=4)
	
	(body, resources) = mail_exporter.from_notebook_node(notebook)
	email = Parser().parse(io.StringIO(body))

	test_redmail(email)

convert_nb()
