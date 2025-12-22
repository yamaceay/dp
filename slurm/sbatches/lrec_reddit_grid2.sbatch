#!/bin/bash

#SBATCH --array=0-36%10
#SBATCH --job-name=lrec_reddit_grid2
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
  "lrec_reddit_grid2/reddit_petre_greedy_0"
  "lrec_reddit_grid2/reddit_petre_shap_1"
  "lrec_reddit_grid2/reddit_dpmlm_uniform_eps_010_2"
  "lrec_reddit_grid2/reddit_dpmlm_uniform_eps_025_3"
  "lrec_reddit_grid2/reddit_dpmlm_uniform_eps_100_4"
  "lrec_reddit_grid2/reddit_dpmlm_uniform_eps_250_5"
  "lrec_reddit_grid2/reddit_dpmlm_uniform_eps_500_6"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_k_eps_010_7"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_k_eps_025_8"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_k_eps_100_9"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_k_eps_250_10"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_k_eps_500_11"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_k_eps_010_12"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_k_eps_025_13"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_k_eps_100_14"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_k_eps_250_15"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_k_eps_500_16"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_eps_010_17"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_eps_025_18"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_eps_100_19"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_eps_250_20"
  "lrec_reddit_grid2/reddit_dpmlm_greedy_unsorted_eps_500_21"
  "lrec_reddit_grid2/reddit_dpmlm_shap_k_eps_010_22"
  "lrec_reddit_grid2/reddit_dpmlm_shap_k_eps_025_23"
  "lrec_reddit_grid2/reddit_dpmlm_shap_k_eps_100_24"
  "lrec_reddit_grid2/reddit_dpmlm_shap_k_eps_250_25"
  "lrec_reddit_grid2/reddit_dpmlm_shap_k_eps_500_26"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_k_eps_010_27"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_k_eps_025_28"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_k_eps_100_29"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_k_eps_250_30"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_k_eps_500_31"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_eps_010_32"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_eps_025_33"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_eps_100_34"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_eps_250_35"
  "lrec_reddit_grid2/reddit_dpmlm_shap_unsorted_eps_500_36"
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
