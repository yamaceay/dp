#!/bin/bash

#SBATCH --array=0-15%3
#SBATCH --job-name=anon_reddit_jobs
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mem-per-cpu=6G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yaay01@dfki.de
#SBATCH --container-image=/enroot/python+3.10.4-buster.sqsh
#SBATCH --container-mounts=`pwd`:`pwd`
#SBATCH --container-workdir=`pwd`

job_names=(
  "reddit_baroud_0"
  "reddit_risk_1"
  "reddit_risk_2"
  "reddit_spacy_3"
  "reddit_presidio_4"
  "reddit_manual_5"
  "reddit_petre_6"
  "reddit_petre_7"
  "reddit_dpprompt_8"
  "reddit_dpparaphrase_9"
  "reddit_dpbart_10"
  "reddit_dpmlm_11"
  "reddit_dpmlm_12"
  "reddit_dpmlm_13"
  "reddit_dpmlm_14"
  "reddit_dpmlm_15"
)

job_cmds=(
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model baroud --model_in configs/model/baroud.yaml --runtime_in configs/runtime/pii_confidence/lambda_*.yaml --output jsonl --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model risk --model_in configs/model/risk/reddit/greedy.yaml --runtime_in configs/runtime/risk_tolerance/rho_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model risk --model_in configs/model/risk/reddit/shap.yaml --runtime_in configs/runtime/risk_tolerance/rho_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model spacy --model_in configs/model/spacy.yaml --runtime_in configs/runtime/pii_confidence/lambda_*.yaml --output jsonl --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model presidio --model_in configs/model/presidio.yaml --output jsonl --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model manual --model_in configs/model/manual.yaml --output jsonl --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model petre --model_in configs/model/petre/reddit/greedy.yaml --runtime_in configs/runtime/k_anon/k_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model petre --model_in configs/model/petre/reddit/shap.yaml --runtime_in configs/runtime/k_anon/k_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpprompt --model_in configs/model/dpprompt.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpparaphrase --model_in configs/model/dpparaphrase.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpbart --model_in configs/model/dpbart.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpmlm --model_in configs/model/dpmlm/reddit/greedy_plus.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpmlm --model_in configs/model/dpmlm/reddit/greedy.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpmlm --model_in configs/model/dpmlm/reddit/shap.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
  "python3 model.py --data reddit --data_in data/reddit/reddit.jsonl --model dpmlm --model_in configs/model/dpmlm/uniform.yaml --runtime_in configs/runtime/dp/eps_*.yaml --output jsonl --stream --start_idx 0"
)

JOB_NAME="${job_names[$SLURM_ARRAY_TASK_ID]}"
JOB_CMD="${job_cmds[$SLURM_ARRAY_TASK_ID]}"

echo "Running job $SLURM_ARRAY_TASK_ID: $JOB_NAME"
echo "Command: $JOB_CMD"

srun -K scripts/install.sh $JOB_CMD
