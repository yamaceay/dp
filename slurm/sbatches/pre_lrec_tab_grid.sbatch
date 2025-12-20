#!/bin/bash

#SBATCH --array=0-1%2
#SBATCH --job-name=pre_lrec_tab_grid
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
  "pre_lrec_tab_grid/tab_shap_0"
  "pre_lrec_tab_grid/tab_greedy_1"
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
