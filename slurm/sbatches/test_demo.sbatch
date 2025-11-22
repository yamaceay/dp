#!/bin/bash

#SBATCH --array=0-5%3
#SBATCH --job-name=test_demo
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G


# Map array task ID to job index and parallel task
MAX_TASKS=3
JOB_IDX=$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=$((SLURM_ARRAY_TASK_ID % MAX_TASKS))

job_names=(
  "test_job1"
  "test_job2"
)

JOB_NAME="${job_names[$JOB_IDX]}"
STATE_FILE="scripts/${JOB_NAME}.state"

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Job: $JOB_NAME (index $JOB_IDX)"
echo "Task within job: $TASK_WITHIN_JOB"
echo "State file: $STATE_FILE"

# Get the specific command for this task using task.sh --incr
JOB_CMD=$(scripts/task.sh --incr --state "$STATE_FILE")
if [[ $? -ne 0 ]]; then
    echo "Failed to get task command from state file"
    exit 1
fi

echo "Command: $JOB_CMD"

srun -K \
  --container-image=/enroot/python+3.10.4-buster.sqsh \
  --container-mounts=`pwd`:`pwd` \
  --container-workdir=`pwd` \
  scripts/install.sh $JOB_CMD 
