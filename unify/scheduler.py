# Executes notebooks according to the saved schedules
import logging
import os
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import schedule
import pandas as pd
import fasteners

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

class OurProcessor(ExecutePreprocessor):
    def preprocess_cell(self, cell, resources, index):
        super().preprocess_cell(cell, resources, index)
        if 'source' in cell:
            logger.info("Executing cell: {}".format(str(cell['source'])))

def run_notebook(notebook_id: str):
    global notebook_contents
    try:
        nb_contents=notebook_contents[notebook_id]
        notebook = nbformat.reads(nb_contents, as_version=4)
        # Now execute the notebook to generate up to date output results (run live queries, etc...)
        ep = OurProcessor(timeout=600, kernel_name='unify_kernel')
        logger.info("Executing notebook: {}".format(notebook_id))
        ep.preprocess(notebook, {'metadata': {'path': "."}})
        logger.info("Notebook done")
    except Exception as e:
        logger.error("Error executing notebook: {}".format(notebook_id))
        logger.error(e)

def reload_schedules():
    rel_sentinel = "/tmp/reload_unify_schedule"
    logger.info(f"checking for schedule reload file at {rel_sentinel}")
    if os.path.exists(rel_sentinel):
        logger.critical("Found reload request")
        os.remove(rel_sentinel)
        schedule.clear()
        run_schedules()

def run_immediately(notebook_list: list):
    global notebook_contents

    interpreter = CommandInterpreter(silence_errors=True)
    for row in interpreter._list_schedules():
        if row['id'] in notebook_list:
            sched = row['schedule']
            notebook = sched['notebook']
            notebook_contents[notebook] = sched['contents']
            run_notebook(notebook)   

def run_schedules(notebook_list = []):
    global notebook_contents

    interpreter = CommandInterpreter(silence_errors=True)
    for row in interpreter._list_schedules():
        logger.info("Schedule run schedule id: {}".format(row["id"]))
        sched = row['schedule']
        if 'contents' not in sched:
            logger.error("No notebook contents found for schedule: {}".format(str(row)))
            continue

        notebook = sched['notebook']
        notebook_contents[notebook] = sched['contents']
        
        run_at_time = pd.to_datetime(sched['run_at'])
        if sched['repeater'] == 'day':
            schedule.every().day.at(str(run_at_time.time())).do(
                run_notebook, 
                notebook_id=notebook
            )
        elif sched['repeater'] == 'week':
            getattr(schedule.every(), run_at_time.day_name().lower()).at(str(run_at_time.time())).do(
                run_notebook, 
                notebook_id=notebook
            )
        elif sched['repeater'] == 'month':
            # See if today is same day of the month as starting date, and if so then
            # run the job today. Assumes we re-schedule all jobs each day
            pass

    schedule.every(2).minutes.do(reload_schedules)
    print("\n".join([str(j) for j in schedule.get_jobs()]))

def schedule_loop():
    try:
        while True:
            logger.info("Waking...")
            schedule.run_pending()
            time.sleep(30)
    except:
        logger.critical("Unhandled exception", exc_info=True)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_immediately(sys.argv[1:])
    else:
        lock = fasteners.InterProcessLock('/tmp/scheduler.lock')
        if lock.acquire(timeout=2):
            run_schedules(sys.argv[1:])
            schedule_loop()
            lock.release()
        else:
            print("Cannot acquire the process lock")
