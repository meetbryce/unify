# Executes notebooks according to the saved schedules
import logging
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from unify import CommandInterpreter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('/tmp/scheduler_unify.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def run_notebook(nb_contents: str, nb_path: str):
    try:
        notebook = nbformat.reads(nb_contents, as_version=4)
        # Now execute the notebook to generate up to date output results (run live queries, etc...)
        ep = ExecutePreprocessor(timeout=600, kernel_name='unify_kernel')
        logger.info("Executing notebook: {}".format(nb_path))
        ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(nb_path)}})
    except Exception as e:
        logger.error("Error executing notebook: {}".format(nb_path))
        logger.error(e)

def run_schedules():
    interpreter = CommandInterpreter(debug=True, silence_errors=True)
    for row in interpreter._list_schedules():
        logger.info("Executing notebook schedule id: {}".format(row["id"]))
        if 'contents' not in row['schedule']:
            logger.error("No notebook contents found for schedule: {}".format(str(row)))
            continue
        nb_contents = row['schedule']['contents']
        nb_path = row['schedule']['notebook']

        run_notebook(nb_contents, nb_path)

if __name__ == '__main__':
    run_schedules()
