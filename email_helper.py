import io
import os
import re
import smtplib
from email.parser import Parser

import pandas as pd
from redmail import EmailSender
from nb2mail import MailExporter
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class EmailHelper:
    def __init__(self):
        if 'EMAIL_USERNAME' in os.environ:
            self.username = os.environ['EMAIL_USERNAME']
        else:
            raise RuntimeError("Cannot send email without setting EMAIL_USERNAME")
        if 'EMAIL_PASSWORD' in os.environ:
            self.password = os.environ['EMAIL_PASSWORD']
        else:
            raise RuntimeError("Cannot send email without setting EMAIL_PASSWORD")
        if 'EMAIL_HOST' in os.environ:
            self.host = os.environ['EMAIL_HOST']
        else:
            raise RuntimeError("Cannot send email without setting EMAIL_HOST")
        if 'EMAIL_PORT' in os.environ:
            self.port = os.environ['EMAIL_PORT']
        else:
            raise RuntimeError("Cannot send email without setting EMAIL_PORT")
        if 'EMAIL_SENDER' in os.environ:
            self.from_address = os.environ['EMAIL_SENDER']
        else:
            self.from_address = 'unify@example.com'

        self.sender: EmailSender = EmailSender(
            host=os.environ['EMAIL_HOST'],
            port=os.environ['EMAIL_PORT'],
            username=os.environ['EMAIL_USERNAME'],
            password=os.environ['EMAIL_PASSWORD'],
            cls_smtp=smtplib.SMTP_SSL,
            use_starttls=False
        )
            
    def send_notebook(self, notebook_path, recipients: list, subject: str=None):
        mail_exporter = MailExporter(template_name="mail")
        notebook = nbformat.reads(open(notebook_path).read(), as_version=4)
        # Now execute the notebook to generate up to date output results (run live queries, etc...)
        ep = ExecutePreprocessor(timeout=600, kernel_name='unify_kernel')

        # Before executing, strip any recursive calls to email the notebook
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                src = cell['source']
                if src and src.strip().startswith('email'):
                    cell['cell_type'] = 'markdown'

        ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})

        (body, resources) = mail_exporter.from_notebook_node(notebook)
        nb_email = Parser().parse(io.StringIO(body))

        html = nb_email.get_payload()[0].get_payload()
        image_tags = []

        def replace_tag(match):
            cid = match.group(1).replace("-", "")
            cid = "image_" + str(len(image_tags))
            image_tags.append(cid)
            return "{{ " + cid + " }}"

        html = re.sub(r'<img src="cid:(.*)"/>', replace_tag, html)
        breakpoint()

        images = []
        # We are assuming the html references will match the payloads
        for payloads in nb_email.get_payload()[1:]:
            images.append(nb_email.get_payload()[1].get_payload(decode=True))

        self.sender.send(
            subject=subject,
            sender=self.from_address,
            receivers=recipients,
            html=html,
                body_images=dict(zip(image_tags, images))
        )

    def send_table(self, data_frame: pd.DataFrame, file_name:str, recipients: list, subject: str=None):
        if data_frame.shape[0] < 100:           
            self.sender.send(
                subject=subject,
                sender=self.from_address,
                receivers=recipients,
                html="<b>Please find your Unify data below:</b>{{ mytable }}",
                body_tables= {"mytable": data_frame},
                attachments={
                    file_name: data_frame
                }
            )
        else:
            self.sender.send(
                subject=subject,
                sender=self.from_address,
                receivers=recipients,
                html="<b>Please find your Unify data attached</b>",
                attachments={
                    file_name: data_frame
                }
            )
