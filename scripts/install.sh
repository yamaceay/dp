#!/bin/bash
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  apt update
  apt install -y [...]
  apt clean
  python -m pip install --upgrade pip
  if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
  fi
  if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
  else
    echo "requirements.txt not found, skipping pip install."
    exit 1
  fi
  touch "${DONEFILE}"
else
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi
"$@"
