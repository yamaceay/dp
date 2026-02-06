#!/bin/bash
set -euo pipefail

DONEFILE="/tmp/install_done_${SLURM_JOBID:-local}"
CACHE_DIR="/netscratch/$USER/hf-cache"

if [[ "${SLURM_LOCALID:-0}" == "0" ]]; then
  python -m pip install --upgrade pip
  pip install --upgrade "transformers>=4.39,<5" "accelerate>=0.30"
  mkdir -p "${CACHE_DIR}"
  touch "${DONEFILE}"
else
  while [[ ! -f "${DONEFILE}" ]]; do
    sleep 1
  done
fi
