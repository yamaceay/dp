#!/bin/bash

#SBATCH --array=0-21%5
#SBATCH --job-name=0_reddit_simple_anonymization
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

job_names=(
  "0_reddit_simple_anonymization/reddit_spacy_0"
  "0_reddit_simple_anonymization/reddit_presidio_1"
  "0_reddit_simple_anonymization/reddit_dpprompt_eps_010_2"
  "0_reddit_simple_anonymization/reddit_dpprompt_eps_025_3"
  "0_reddit_simple_anonymization/reddit_dpprompt_eps_100_4"
  "0_reddit_simple_anonymization/reddit_dpprompt_eps_250_5"
  "0_reddit_simple_anonymization/reddit_dpprompt_eps_500_6"
  "0_reddit_simple_anonymization/reddit_dpparaphrase_eps_010_7"
  "0_reddit_simple_anonymization/reddit_dpparaphrase_eps_025_8"
  "0_reddit_simple_anonymization/reddit_dpparaphrase_eps_100_9"
  "0_reddit_simple_anonymization/reddit_dpparaphrase_eps_250_10"
  "0_reddit_simple_anonymization/reddit_dpparaphrase_eps_500_11"
  "0_reddit_simple_anonymization/reddit_dpbart_eps_010_12"
  "0_reddit_simple_anonymization/reddit_dpbart_eps_025_13"
  "0_reddit_simple_anonymization/reddit_dpbart_eps_100_14"
  "0_reddit_simple_anonymization/reddit_dpbart_eps_250_15"
  "0_reddit_simple_anonymization/reddit_dpbart_eps_500_16"
  "0_reddit_simple_anonymization/reddit_dpmlm_uniform_eps_010_17"
  "0_reddit_simple_anonymization/reddit_dpmlm_uniform_eps_025_18"
  "0_reddit_simple_anonymization/reddit_dpmlm_uniform_eps_100_19"
  "0_reddit_simple_anonymization/reddit_dpmlm_uniform_eps_250_20"
  "0_reddit_simple_anonymization/reddit_dpmlm_uniform_eps_500_21"
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
