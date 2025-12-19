#!/bin/bash

#SBATCH --array=0-35%8
#SBATCH --job-name=tri_reddit_grid
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
  "tri_reddit_grid/tri_reddit_lr1e-4_bs16_ft5_pre0_0"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs16_ft5_pre1_1"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs32_ft5_pre0_2"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs32_ft5_pre1_3"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs16_ft10_pre0_4"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs16_ft10_pre1_5"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs32_ft10_pre0_6"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs32_ft10_pre1_7"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs16_ft15_pre0_8"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs16_ft15_pre1_9"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs32_ft15_pre0_10"
  "tri_reddit_grid/tri_reddit_lr1e-4_bs32_ft15_pre1_11"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs16_ft5_pre0_12"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs16_ft5_pre1_13"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs32_ft5_pre0_14"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs32_ft5_pre1_15"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs16_ft10_pre0_16"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs16_ft10_pre1_17"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs32_ft10_pre0_18"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs32_ft10_pre1_19"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs16_ft15_pre0_20"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs16_ft15_pre1_21"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs32_ft15_pre0_22"
  "tri_reddit_grid/tri_reddit_lr2e-5_bs32_ft15_pre1_23"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs16_ft5_pre0_24"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs16_ft5_pre1_25"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs32_ft5_pre0_26"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs32_ft5_pre1_27"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs16_ft10_pre0_28"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs16_ft10_pre1_29"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs32_ft10_pre0_30"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs32_ft10_pre1_31"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs16_ft15_pre0_32"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs16_ft15_pre1_33"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs32_ft15_pre0_34"
  "tri_reddit_grid/tri_reddit_lr5e-5_bs32_ft15_pre1_35"
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
