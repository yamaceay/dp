#!/bin/bash

#SBATCH --array=0-46%10
#SBATCH --job-name=lrec_tab_grid2
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
  "lrec_tab_grid2/tab_petre_greedy_0"
  "lrec_tab_grid2/tab_petre_shap_1"
  "lrec_tab_grid2/tab_dpmlm_uniform_eps_010_2"
  "lrec_tab_grid2/tab_dpmlm_uniform_eps_025_3"
  "lrec_tab_grid2/tab_dpmlm_uniform_eps_100_4"
  "lrec_tab_grid2/tab_dpmlm_uniform_eps_250_5"
  "lrec_tab_grid2/tab_dpmlm_uniform_eps_500_6"
  "lrec_tab_grid2/tab_dpmlm_greedy_k_eps_010_7"
  "lrec_tab_grid2/tab_dpmlm_greedy_k_eps_025_8"
  "lrec_tab_grid2/tab_dpmlm_greedy_k_eps_100_9"
  "lrec_tab_grid2/tab_dpmlm_greedy_k_eps_250_10"
  "lrec_tab_grid2/tab_dpmlm_greedy_k_eps_500_11"
  "lrec_tab_grid2/tab_dpmlm_greedy_unsorted_k_eps_010_12"
  "lrec_tab_grid2/tab_dpmlm_greedy_unsorted_k_eps_025_13"
  "lrec_tab_grid2/tab_dpmlm_greedy_unsorted_k_eps_100_14"
  "lrec_tab_grid2/tab_dpmlm_greedy_unsorted_k_eps_250_15"
  "lrec_tab_grid2/tab_dpmlm_greedy_unsorted_k_eps_500_16"
  "lrec_tab_grid2/tab_dpmlm_greedy_masked_eps_010_17"
  "lrec_tab_grid2/tab_dpmlm_greedy_masked_eps_025_18"
  "lrec_tab_grid2/tab_dpmlm_greedy_masked_eps_100_19"
  "lrec_tab_grid2/tab_dpmlm_greedy_masked_eps_250_20"
  "lrec_tab_grid2/tab_dpmlm_greedy_masked_eps_500_21"
  "lrec_tab_grid2/tab_dpmlm_greedy_eps_010_22"
  "lrec_tab_grid2/tab_dpmlm_greedy_eps_025_23"
  "lrec_tab_grid2/tab_dpmlm_greedy_eps_100_24"
  "lrec_tab_grid2/tab_dpmlm_greedy_eps_250_25"
  "lrec_tab_grid2/tab_dpmlm_greedy_eps_500_26"
  "lrec_tab_grid2/tab_dpmlm_shap_k_eps_010_27"
  "lrec_tab_grid2/tab_dpmlm_shap_k_eps_025_28"
  "lrec_tab_grid2/tab_dpmlm_shap_k_eps_100_29"
  "lrec_tab_grid2/tab_dpmlm_shap_k_eps_250_30"
  "lrec_tab_grid2/tab_dpmlm_shap_k_eps_500_31"
  "lrec_tab_grid2/tab_dpmlm_shap_unsorted_k_eps_010_32"
  "lrec_tab_grid2/tab_dpmlm_shap_unsorted_k_eps_025_33"
  "lrec_tab_grid2/tab_dpmlm_shap_unsorted_k_eps_100_34"
  "lrec_tab_grid2/tab_dpmlm_shap_unsorted_k_eps_250_35"
  "lrec_tab_grid2/tab_dpmlm_shap_unsorted_k_eps_500_36"
  "lrec_tab_grid2/tab_dpmlm_shap_masked_eps_010_37"
  "lrec_tab_grid2/tab_dpmlm_shap_masked_eps_025_38"
  "lrec_tab_grid2/tab_dpmlm_shap_masked_eps_100_39"
  "lrec_tab_grid2/tab_dpmlm_shap_masked_eps_250_40"
  "lrec_tab_grid2/tab_dpmlm_shap_masked_eps_500_41"
  "lrec_tab_grid2/tab_dpmlm_shap_eps_010_42"
  "lrec_tab_grid2/tab_dpmlm_shap_eps_025_43"
  "lrec_tab_grid2/tab_dpmlm_shap_eps_100_44"
  "lrec_tab_grid2/tab_dpmlm_shap_eps_250_45"
  "lrec_tab_grid2/tab_dpmlm_shap_eps_500_46"
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
