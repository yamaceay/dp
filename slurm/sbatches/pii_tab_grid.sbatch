#!/bin/bash

#SBATCH --array=0-17%8
#SBATCH --job-name=pii_tab_grid
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
  "pii_tab_grid/pii_tab_lr1e-5_bs8_m_partial_recall_0"
  "pii_tab_grid/pii_tab_lr1e-5_bs8_m_strict_recall_1"
  "pii_tab_grid/pii_tab_lr1e-5_bs8_m_exact_recall_2"
  "pii_tab_grid/pii_tab_lr1e-5_bs16_m_partial_recall_3"
  "pii_tab_grid/pii_tab_lr1e-5_bs16_m_strict_recall_4"
  "pii_tab_grid/pii_tab_lr1e-5_bs16_m_exact_recall_5"
  "pii_tab_grid/pii_tab_lr2e-5_bs8_m_partial_recall_6"
  "pii_tab_grid/pii_tab_lr2e-5_bs8_m_strict_recall_7"
  "pii_tab_grid/pii_tab_lr2e-5_bs8_m_exact_recall_8"
  "pii_tab_grid/pii_tab_lr2e-5_bs16_m_partial_recall_9"
  "pii_tab_grid/pii_tab_lr2e-5_bs16_m_strict_recall_10"
  "pii_tab_grid/pii_tab_lr2e-5_bs16_m_exact_recall_11"
  "pii_tab_grid/pii_tab_lr3e-5_bs8_m_partial_recall_12"
  "pii_tab_grid/pii_tab_lr3e-5_bs8_m_strict_recall_13"
  "pii_tab_grid/pii_tab_lr3e-5_bs8_m_exact_recall_14"
  "pii_tab_grid/pii_tab_lr3e-5_bs16_m_partial_recall_15"
  "pii_tab_grid/pii_tab_lr3e-5_bs16_m_strict_recall_16"
  "pii_tab_grid/pii_tab_lr3e-5_bs16_m_exact_recall_17"
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
