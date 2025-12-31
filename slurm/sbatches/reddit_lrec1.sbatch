#!/bin/bash

#SBATCH --array=0-21%10
#SBATCH --job-name=reddit_lrec1
#SBATCH --output=logs/%x/%a_%j.out
#SBATCH --error=logs/%x/%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1


# Map array task ID to job index and parallel task
MAX_TASKS=1
JOB_IDX=$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=$((SLURM_ARRAY_TASK_ID % MAX_TASKS))

job_names=(
  "reddit_lrec1/reddit_spacy_0"
  "reddit_lrec1/reddit_presidio_1"
  "reddit_lrec1/reddit_dpprompt_eps_010_2"
  "reddit_lrec1/reddit_dpprompt_eps_025_3"
  "reddit_lrec1/reddit_dpprompt_eps_100_4"
  "reddit_lrec1/reddit_dpprompt_eps_250_5"
  "reddit_lrec1/reddit_dpprompt_eps_500_6"
  "reddit_lrec1/reddit_dpparaphrase_eps_010_7"
  "reddit_lrec1/reddit_dpparaphrase_eps_025_8"
  "reddit_lrec1/reddit_dpparaphrase_eps_100_9"
  "reddit_lrec1/reddit_dpparaphrase_eps_250_10"
  "reddit_lrec1/reddit_dpparaphrase_eps_500_11"
  "reddit_lrec1/reddit_dpbart_eps_010_12"
  "reddit_lrec1/reddit_dpbart_eps_025_13"
  "reddit_lrec1/reddit_dpbart_eps_100_14"
  "reddit_lrec1/reddit_dpbart_eps_250_15"
  "reddit_lrec1/reddit_dpbart_eps_500_16"
  "reddit_lrec1/reddit_dpmlm_uniform_eps_010_17"
  "reddit_lrec1/reddit_dpmlm_uniform_eps_025_18"
  "reddit_lrec1/reddit_dpmlm_uniform_eps_100_19"
  "reddit_lrec1/reddit_dpmlm_uniform_eps_250_20"
  "reddit_lrec1/reddit_dpmlm_uniform_eps_500_21"
)

JOB_NAME="${job_names[$JOB_IDX]}"
STATE_FILE="slurm/states/${JOB_NAME}.state"

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Job: $JOB_NAME (index $JOB_IDX)"
echo "Task within job: $TASK_WITHIN_JOB"
echo "State file: $STATE_FILE"

export TASK_WITHIN_JOB JOB_NAME STATE_FILE SLURM_ARRAY_TASK_ID

# Execute the task using task.sh --incr (it will run the command directly)
srun -K \
  --container-image=/enroot/python+3.10.4-buster.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER \
  --container-workdir=`pwd` \
  scripts/install.sh scripts/task.sh --incr --state "$STATE_FILE"
