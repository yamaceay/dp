#!/bin/bash

#SBATCH --array=0-1%3
#SBATCH --job-name=tri_jobs
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
  "tri_reddit_0"
  "tri_tab_1"
)

job_cmds=(
  "python tri_by_bk.py --dataset reddit --data_path data/reddit/reddit.jsonl --attacker_extensions data/reddit/reddit_bk_sum.jsonl --use_pretraining --pretraining_epochs 3 --finetuning_epochs 7 --mode train --device cpu"
  "python tri_by_bk.py --dataset tab --data_path data/TAB/splitted/test.json --attacker_extensions data/TAB/test_bk_sum.jsonl --use_pretraining --pretraining_epochs 3 --finetuning_epochs 10 --mode train --device cpu"
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
