#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
cd /home/scottp/src/unify
export $(cat .env)
conda activate venv
python -m unify.scheduler
