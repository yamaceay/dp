#!/bin/bash

#SBATCH --array=0-1%2
#SBATCH --job-name=tri_reddit_es_grid
#SBATCH --output=logs/%x/%a_%j.out
#SBATCH --error=logs/%x/%a_%j.err
#SBATCH --partition=RTXA6000
#SBATCH --gpus=1
#SBATCH --mem=40GB
#SBATCH --time=10
#SBATCH --ntasks=1


# Map array task ID to job index and parallel task
MAX_TASKS=1
JOB_IDX=$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=$((SLURM_ARRAY_TASK_ID % MAX_TASKS))

job_names=(
  "tri_reddit_es_grid/tri_reddit_es_lr5e-5_0"
  "tri_reddit_es_grid/tri_reddit_deid_es_lr5e-5_1"
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
    --container-mounts="`pwd`:`pwd`,/netscratch/$USER:/netscratch/$USER" \
    --container-workdir="`pwd`" \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_24.01-py3.sqsh \
    --task-prolog="`pwd`/scripts/install.sh" scripts/task.sh --incr --state "$STATE_FILE"
