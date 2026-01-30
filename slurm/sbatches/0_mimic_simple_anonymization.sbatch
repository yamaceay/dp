#!/bin/bash

#SBATCH --array=0-23%10
#SBATCH --job-name=0_mimic_simple_anonymization
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
  "0_mimic_simple_anonymization/mimic_baroud_0"
  "0_mimic_simple_anonymization/mimic_spacy_1"
  "0_mimic_simple_anonymization/mimic_presidio_2"
  "0_mimic_simple_anonymization/mimic_manual_3"
  "0_mimic_simple_anonymization/mimic_dpprompt_eps_010_4"
  "0_mimic_simple_anonymization/mimic_dpprompt_eps_025_5"
  "0_mimic_simple_anonymization/mimic_dpprompt_eps_100_6"
  "0_mimic_simple_anonymization/mimic_dpprompt_eps_250_7"
  "0_mimic_simple_anonymization/mimic_dpprompt_eps_500_8"
  "0_mimic_simple_anonymization/mimic_dpparaphrase_eps_010_9"
  "0_mimic_simple_anonymization/mimic_dpparaphrase_eps_025_10"
  "0_mimic_simple_anonymization/mimic_dpparaphrase_eps_100_11"
  "0_mimic_simple_anonymization/mimic_dpparaphrase_eps_250_12"
  "0_mimic_simple_anonymization/mimic_dpparaphrase_eps_500_13"
  "0_mimic_simple_anonymization/mimic_dpbart_eps_010_14"
  "0_mimic_simple_anonymization/mimic_dpbart_eps_025_15"
  "0_mimic_simple_anonymization/mimic_dpbart_eps_100_16"
  "0_mimic_simple_anonymization/mimic_dpbart_eps_250_17"
  "0_mimic_simple_anonymization/mimic_dpbart_eps_500_18"
  "0_mimic_simple_anonymization/mimic_dpmlm_uniform_eps_010_19"
  "0_mimic_simple_anonymization/mimic_dpmlm_uniform_eps_025_20"
  "0_mimic_simple_anonymization/mimic_dpmlm_uniform_eps_100_21"
  "0_mimic_simple_anonymization/mimic_dpmlm_uniform_eps_250_22"
  "0_mimic_simple_anonymization/mimic_dpmlm_uniform_eps_500_23"
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
