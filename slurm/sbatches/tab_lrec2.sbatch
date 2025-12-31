#!/bin/bash

#SBATCH --array=0-31%10
#SBATCH --job-name=tab_lrec2
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
  "tab_lrec2/tab_risk_shap_0"
  "tab_lrec2/tab_petre_shap_1"
  "tab_lrec2/tab_dpmlm_shap_eps_010_2"
  "tab_lrec2/tab_dpmlm_shap_eps_025_3"
  "tab_lrec2/tab_dpmlm_shap_eps_100_4"
  "tab_lrec2/tab_dpmlm_shap_eps_250_5"
  "tab_lrec2/tab_dpmlm_shap_eps_500_6"
  "tab_lrec2/tab_dpmlm_shap_k_eps_010_7"
  "tab_lrec2/tab_dpmlm_shap_k_eps_025_8"
  "tab_lrec2/tab_dpmlm_shap_k_eps_100_9"
  "tab_lrec2/tab_dpmlm_shap_k_eps_250_10"
  "tab_lrec2/tab_dpmlm_shap_k_eps_500_11"
  "tab_lrec2/tab_dpmlm_shap_masked_eps_010_12"
  "tab_lrec2/tab_dpmlm_shap_masked_eps_025_13"
  "tab_lrec2/tab_dpmlm_shap_masked_eps_100_14"
  "tab_lrec2/tab_dpmlm_shap_masked_eps_250_15"
  "tab_lrec2/tab_dpmlm_shap_masked_eps_500_16"
  "tab_lrec2/tab_dpmlm_shap_pii_eps_010_17"
  "tab_lrec2/tab_dpmlm_shap_pii_eps_025_18"
  "tab_lrec2/tab_dpmlm_shap_pii_eps_100_19"
  "tab_lrec2/tab_dpmlm_shap_pii_eps_250_20"
  "tab_lrec2/tab_dpmlm_shap_pii_eps_500_21"
  "tab_lrec2/tab_dpmlm_shap_risk_eps_010_22"
  "tab_lrec2/tab_dpmlm_shap_risk_eps_025_23"
  "tab_lrec2/tab_dpmlm_shap_risk_eps_100_24"
  "tab_lrec2/tab_dpmlm_shap_risk_eps_250_25"
  "tab_lrec2/tab_dpmlm_shap_risk_eps_500_26"
  "tab_lrec2/tab_dpmlm_shap_unsorted_k_eps_010_27"
  "tab_lrec2/tab_dpmlm_shap_unsorted_k_eps_025_28"
  "tab_lrec2/tab_dpmlm_shap_unsorted_k_eps_100_29"
  "tab_lrec2/tab_dpmlm_shap_unsorted_k_eps_250_30"
  "tab_lrec2/tab_dpmlm_shap_unsorted_k_eps_500_31"
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
