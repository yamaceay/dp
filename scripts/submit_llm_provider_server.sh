#!/bin/bash
set -euo pipefail

RUN_NAME="llm_provider_server"
PORT=18082
TABLE_FILE="slurm/tables/a8_llm_provider_server.table"
SERVER_INFO_FILE="logs/llm_provider_server.json"
REGISTRY_FILE="logs/llm_provider_endpoints.json"
MAX_ENDPOINTS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --table-file=*) TABLE_FILE="${1#*=}"; shift ;;
    --server-info-file=*) SERVER_INFO_FILE="${1#*=}"; shift ;;
    --registry-file=*) REGISTRY_FILE="${1#*=}"; shift ;;
    --max-endpoints=*) MAX_ENDPOINTS="${1#*=}"; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/submit_llm_provider_server.sh [options]
  --run-name=NAME
  --port=PORT
  --table-file=PATH
  --server-info-file=PATH
  --registry-file=PATH
  --max-endpoints=N
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$TABLE_FILE")" "$(dirname "$SERVER_INFO_FILE")" "$(dirname "$REGISTRY_FILE")"

inner_cmd="export HF_HOME=/netscratch/\$USER/hf-cache; "
inner_cmd+="export HUGGINGFACE_HUB_CACHE=/netscratch/\$USER/hf-cache; "
inner_cmd+="python scripts/llm_provider_server.py "
inner_cmd+="--host 0.0.0.0 "
inner_cmd+="--port $(printf '%q' "$PORT") "
inner_cmd+="--server-info-file $(printf '%q' "$SERVER_INFO_FILE") "
inner_cmd+="--registry-file $(printf '%q' "$REGISTRY_FILE") "
inner_cmd+="--max-endpoints $(printf '%q' "$MAX_ENDPOINTS") "
inner_cmd="${inner_cmd%" "}"

printf "%s|bash -lc %q\n" "$RUN_NAME" "$inner_cmd" > "$TABLE_FILE"
echo "Wrote table: $TABLE_FILE"

scripts/run.sh \
  --max-concurrent=1 \
  --max-tasks=1 \
  --install-file=scripts/install_llm_provider_server.sh \
  -y \
  "$TABLE_FILE"
