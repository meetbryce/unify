# Executes notebooks according to the saved schedules
import logging
import os
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import schedule
import pandas as pd

from .unify import CommandInterpreter

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

notebook_contents = {}

def run_notebook(nb_path: str):
    global notebook_contents
    try:
        nb_contents=notebook_contents[nb_path]
        notebook = nbformat.reads(nb_contents, as_version=4)
        # Now execute the notebook to generate up to date output results (run live queries, etc...)
        ep = ExecutePreprocessor(timeout=600, kernel_name='unify_kernel')
        logger.info("Executing notebook: {}".format(nb_path))
        ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(nb_path)}})
        logger.info("Notebook done")
    except Exception as e:
        logger.error("Error executing notebook: {}".format(nb_path))
        logger.error(e)

def find_notebook(notebook: str):
    if os.path.exists(notebook):
        return notebook
    else:
        nbpath = os.path.join(os.path.dirname(__file__), "../notebooks", notebook)
        if os.path.exists(nbpath):
            return nbpath
        else:
            raise RuntimeError(f"Cannot find notebook '{notebook}'")
    
def run_schedules(notebook_list = []):
    global notebook_contents

    if notebook_list:
        for nb in notebook_list:
            nb_path = find_notebook(nb)
            notebook_contents[nb_path] = open(nb_path).read()
            run_notebook(nb_path)
        return
        
    interpreter = CommandInterpreter(silence_errors=True)
    for row in interpreter._list_schedules():
        logger.info("Executing notebook schedule id: {}".format(row["id"]))
        sched = row['schedule']
        if 'contents' not in sched:
            logger.error("No notebook contents found for schedule: {}".format(str(row)))
            continue
        notebook_contents[sched['notebook']] = sched['contents']
        
        run_at_time = pd.to_datetime(sched['run_at'])
        if sched['repeater'] == 'day':
            schedule.every().day.at(str(run_at_time.time())).do(
                run_notebook, 
                nb_path=sched['notebook']
            )
        elif sched['repeater'] == 'week':
            getattr(schedule.every(), run_at_time.day_name().lower()).at(str(run_at_time.time())).do(
                run_notebook, 
                nb_path=sched['notebook']
            )
        elif sched['repeat'] == 'month':
            # See if today is same day of the month as starting date, and if so then
            # run the job today. Assumes we re-schedule all jobs each day
            pass

        print(schedule.get_jobs())

    while True:
        logger.info("Waking...")
        schedule.run_pending()
        time.sleep(30)

if __name__ == '__main__':
    run_schedules(sys.argv[1:])
