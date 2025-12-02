#!/bin/bash

#SBATCH --array=0-17%3
#SBATCH --job-name=tab_tasks
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yaay01@dfki.de

# Map array task ID to job index and parallel task
MAX_TASKS=1
JOB_IDX=$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=$((SLURM_ARRAY_TASK_ID % MAX_TASKS))

job_names=(
  "tab_baroud_baroud_0"
  "tab_risk_greedy_1"
  "tab_risk_shap_2"
  "tab_spacy_spacy_3"
  "tab_presidio_presidio_4"
  "tab_manual_manual_5"
  "tab_petre_greedy_6"
  "tab_petre_shap_7"
  "tab_dpprompt_dpprompt_8"
  "tab_dpparaphrase_dpparaphrase_9"
  "tab_dpbart_dpbart_10"
  "tab_dpmlm_greedy_pii_11"
  "tab_dpmlm_greedy_plus_12"
  "tab_dpmlm_greedy_risk_13"
  "tab_dpmlm_greedy_14"
  "tab_dpmlm_shap_15"
  "tab_dpmlm_uniform_plus_16"
  "tab_dpmlm_uniform_17"
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
