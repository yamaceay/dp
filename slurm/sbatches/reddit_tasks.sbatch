#!/bin/bash

#SBATCH --array=0-63%3
#SBATCH --job-name=reddit_tasks
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yaay01@dfki.de

# Map array task ID to job index and parallel task
MAX_TASKS=4
JOB_IDX=$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=$((SLURM_ARRAY_TASK_ID % MAX_TASKS))

job_names=(
  "reddit_baroud_0"
  "reddit_risk_1"
  "reddit_risk_2"
  "reddit_spacy_3"
  "reddit_presidio_4"
  "reddit_manual_5"
  "reddit_petre_6"
  "reddit_petre_7"
  "reddit_dpprompt_8"
  "reddit_dpparaphrase_9"
  "reddit_dpbart_10"
  "reddit_dpmlm_11"
  "reddit_dpmlm_12"
  "reddit_dpmlm_13"
  "reddit_dpmlm_14"
  "reddit_dpmlm_15"
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
  --container-mounts=`pwd`:`pwd` \
  --container-workdir=`pwd` \
  scripts/install.sh scripts/task.sh --incr --state "$STATE_FILE"
