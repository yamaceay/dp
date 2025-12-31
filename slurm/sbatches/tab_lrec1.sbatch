#!/bin/bash

#SBATCH --array=0-23%10
#SBATCH --job-name=tab_lrec1
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
  "tab_lrec1/tab_baroud_0"
  "tab_lrec1/tab_spacy_1"
  "tab_lrec1/tab_presidio_2"
  "tab_lrec1/tab_manual_3"
  "tab_lrec1/tab_dpprompt_eps_010_4"
  "tab_lrec1/tab_dpprompt_eps_025_5"
  "tab_lrec1/tab_dpprompt_eps_100_6"
  "tab_lrec1/tab_dpprompt_eps_250_7"
  "tab_lrec1/tab_dpprompt_eps_500_8"
  "tab_lrec1/tab_dpparaphrase_eps_010_9"
  "tab_lrec1/tab_dpparaphrase_eps_025_10"
  "tab_lrec1/tab_dpparaphrase_eps_100_11"
  "tab_lrec1/tab_dpparaphrase_eps_250_12"
  "tab_lrec1/tab_dpparaphrase_eps_500_13"
  "tab_lrec1/tab_dpbart_eps_010_14"
  "tab_lrec1/tab_dpbart_eps_025_15"
  "tab_lrec1/tab_dpbart_eps_100_16"
  "tab_lrec1/tab_dpbart_eps_250_17"
  "tab_lrec1/tab_dpbart_eps_500_18"
  "tab_lrec1/tab_dpmlm_uniform_eps_010_19"
  "tab_lrec1/tab_dpmlm_uniform_eps_025_20"
  "tab_lrec1/tab_dpmlm_uniform_eps_100_21"
  "tab_lrec1/tab_dpmlm_uniform_eps_250_22"
  "tab_lrec1/tab_dpmlm_uniform_eps_500_23"
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
