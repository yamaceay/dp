#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/post.sh [options]

Runs data transformation (optional) and visualization scripts.

Options:
  --meta            Enable meta mode for scripts that support it.
  --skip-transform  Skip parse/merge/transform and run visualizations only.
  -h, --help        Show this help message.
EOF
}

meta_mode=false
skip_transform=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --meta)
      meta_mode=true
      shift
      ;;
    --skip-transform)
      skip_transform=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

run_step() {
  echo "==> $*"
  "$@"
}

if [[ "$skip_transform" == false ]]; then
  run_step "$PYTHON_BIN" parse_runtime.py
  run_step "$PYTHON_BIN" merge_logs.py
  run_step "$PYTHON_BIN" transform_logs.py
else
  echo "==> Skipping data transformation (--skip-transform)"
fi

run_step "$PYTHON_BIN" plot_shap_tokens.py
