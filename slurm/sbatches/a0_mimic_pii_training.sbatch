#!/bin/bash

#SBATCH --array=0-17%1
#SBATCH --job-name=a0_mimic_pii_training
#SBATCH --output=logs/%x/%j_%a.out
#SBATCH --error=logs/%x/%j_%a.err
#SBATCH --partition=RTXA6000
#SBATCH --gpus=1
#SBATCH --mem=40GB
#SBATCH --time=900
#SBATCH --ntasks=1


# Map array task ID to job index and parallel task
MAX_TASKS=1
JOB_IDX=$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=$((SLURM_ARRAY_TASK_ID % MAX_TASKS))

INSTALL_FILE="scripts/install.sh"

job_names=(
  "a0_mimic_pii_training/pii_mimic_lr1e-5_bs8_m_partial_recall_0"
  "a0_mimic_pii_training/pii_mimic_lr1e-5_bs8_m_strict_recall_1"
  "a0_mimic_pii_training/pii_mimic_lr1e-5_bs8_m_exact_recall_2"
  "a0_mimic_pii_training/pii_mimic_lr1e-5_bs16_m_partial_recall_3"
  "a0_mimic_pii_training/pii_mimic_lr1e-5_bs16_m_strict_recall_4"
  "a0_mimic_pii_training/pii_mimic_lr1e-5_bs16_m_exact_recall_5"
  "a0_mimic_pii_training/pii_mimic_lr2e-5_bs8_m_partial_recall_6"
  "a0_mimic_pii_training/pii_mimic_lr2e-5_bs8_m_strict_recall_7"
  "a0_mimic_pii_training/pii_mimic_lr2e-5_bs8_m_exact_recall_8"
  "a0_mimic_pii_training/pii_mimic_lr2e-5_bs16_m_partial_recall_9"
  "a0_mimic_pii_training/pii_mimic_lr2e-5_bs16_m_strict_recall_10"
  "a0_mimic_pii_training/pii_mimic_lr2e-5_bs16_m_exact_recall_11"
  "a0_mimic_pii_training/pii_mimic_lr3e-5_bs8_m_partial_recall_12"
  "a0_mimic_pii_training/pii_mimic_lr3e-5_bs8_m_strict_recall_13"
  "a0_mimic_pii_training/pii_mimic_lr3e-5_bs8_m_exact_recall_14"
  "a0_mimic_pii_training/pii_mimic_lr3e-5_bs16_m_partial_recall_15"
  "a0_mimic_pii_training/pii_mimic_lr3e-5_bs16_m_strict_recall_16"
  "a0_mimic_pii_training/pii_mimic_lr3e-5_bs16_m_exact_recall_17"
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
    --task-prolog="`pwd`/${INSTALL_FILE}" scripts/task.sh --incr --state "$STATE_FILE"
