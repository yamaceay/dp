#!/bin/bash
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
VENV_DIR="/netscratch/${USER}/dp/.venv"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ $SLURM_LOCALID == 0 ]]; then
  log "Starting install on rank 0."

  log "Ensuring venv exists at ${VENV_DIR}."
  if [[ ! -d "${VENV_DIR}" ]]; then
    mkdir -p "$(dirname "${VENV_DIR}")"
    python -m venv "${VENV_DIR}"
  fi

  log "Activating venv at ${VENV_DIR}."
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  log "Ensuring pip and uv are installed."
  python -m ensurepip --upgrade >/dev/null 2>&1 || true
  python -m pip install --upgrade --force-reinstall pip setuptools wheel
  if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
  fi

  if [ -f requirements.txt ]; then
    if command -v uv >/dev/null 2>&1; then
      log "Installing requirements.txt via uv."
      uv pip install -r requirements.txt
    else
      log "Installing requirements.txt via pip."
      python -m pip install -r requirements.txt
    fi
  else
    log "requirements.txt not found, skipping pip install."
    exit 1
  fi

  touch "${DONEFILE}"
  log "Install completed on rank 0."
else
  log "Waiting for rank 0 to finish install."
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
  log "Install done signal detected."
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  log "Activating venv for run: ${VENV_DIR}."
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
else
  log "Venv not found at ${VENV_DIR}; continuing without activation."
fi

log "Running command: $*"
"$@"
