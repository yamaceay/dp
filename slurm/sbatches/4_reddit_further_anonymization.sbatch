#!/bin/bash

#SBATCH --array=0-26%10
#SBATCH --job-name=4_reddit_further_anonymization
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
  "4_reddit_further_anonymization/reddit_risk_shap_0"
  "4_reddit_further_anonymization/reddit_petre_shap_1"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_eps_010_2"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_eps_025_3"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_eps_100_4"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_eps_250_5"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_eps_500_6"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_k_eps_010_7"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_k_eps_025_8"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_k_eps_100_9"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_k_eps_250_10"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_k_eps_500_11"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_masked_eps_010_12"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_masked_eps_025_13"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_masked_eps_100_14"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_masked_eps_250_15"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_masked_eps_500_16"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_risk_eps_010_17"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_risk_eps_025_18"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_risk_eps_100_19"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_risk_eps_250_20"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_risk_eps_500_21"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_unsorted_k_eps_010_22"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_unsorted_k_eps_025_23"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_unsorted_k_eps_100_24"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_unsorted_k_eps_250_25"
  "4_reddit_further_anonymization/reddit_dpmlm_shap_unsorted_k_eps_500_26"
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
