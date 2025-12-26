#!/bin/bash
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
VENV_DIR=".venv"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ $SLURM_LOCALID == 0 ]]; then
  log "Starting install on rank 0."
  apt update
  apt install -y [...]
  apt clean

  log "Ensuring venv exists at ${VENV_DIR}."
  if [[ ! -d "${VENV_DIR}" ]]; then
    python -m venv "${VENV_DIR}"
  fi

  log "Activating venv at ${VENV_DIR}."
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  log "Repairing pip if needed and ensuring uv is installed."
  python -m ensurepip --upgrade >/dev/null 2>&1 || true
  if ! python -m pip --version >/dev/null 2>&1; then
    log "pip missing in venv; forcing reinstall."
  fi
  python -m pip install --upgrade --force-reinstall pip setuptools wheel
  if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
  fi

  if [ -f requirements.txt ]; then
    log "Installing requirements.txt via uv."
    uv pip install --active -r requirements.txt
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
