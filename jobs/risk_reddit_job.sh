#!/bin/bash

srun -K     --job-name risk_reddit_job     --output=logs/risk_reddit_job_%j.out     --error=logs/risk_reddit_job_%j.err     --partition=batch     --ntasks=1     --cpus-per-task=10     --gpus-per-task=1     --mem-per-cpu=6G     --mail-type=ALL \
    --mail-user=yaay01@dfki.de \
    --container-image=/enroot/python+3.10.4-buster.sqsh     --container-mounts=`pwd`:`pwd`     --container-workdir=`pwd`     scripts/install.sh python3 risk.py --data reddit --data_in data/reddit/reddit.jsonl --explainer greedy --explainer_in models/tri_pipelines/tab/20251119_162236 --save_to_jsonl data/reddit/reddit_risk_greedy.jsonl --sort_by scores
