#!/bin/bash

#SBATCH --array=0-3%3
#SBATCH --job-name=risk_jobs
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mem-per-cpu=6G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yaay01@dfki.de

job_names=(
  "greedy_reddit_0"
  "shap_reddit_1"
  "greedy_tab_2"
  "shap_tab_3"
)

job_cmds=(
  "python risk.py --data reddit --data_in data/reddit/reddit.jsonl --explainer greedy --explainer_in models/tri_pipelines/reddit/20251119_152245 --save_to_jsonl data/reddit/reddit_risk_greedy.jsonl --sort_by scores"
  "python risk.py --data reddit --data_in data/reddit/reddit.jsonl --explainer shap --explainer_in models/tri_pipelines/reddit/20251119_152245 --save_to_jsonl data/reddit/reddit_risk_shap.jsonl --sort_by scores"
  "python risk.py --data tab --data_in data/TAB/test.json --explainer greedy --explainer_in models/tri_pipelines/tab/20251119_162236 --save_to_jsonl data/TAB/test_risk_greedy.jsonl --sort_by scores"
  "python risk.py --data tab --data_in data/TAB/test.json --explainer shap --explainer_in models/tri_pipelines/tab/20251119_162236 --save_to_jsonl data/TAB/test_risk_shap.jsonl --sort_by scores"
)

JOB_NAME="${job_names[$SLURM_ARRAY_TASK_ID]}"
JOB_CMD="${job_cmds[$SLURM_ARRAY_TASK_ID]}"

echo "Running job $SLURM_ARRAY_TASK_ID: $JOB_NAME"
echo "Command: $JOB_CMD"

srun -K \
  --container-image=/enroot/python+3.10.4-buster.sqsh \
  --container-mounts=`pwd`:`pwd` \
  --container-workdir=`pwd` \
  scripts/install.sh $JOB_CMD 
