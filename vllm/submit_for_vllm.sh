#!/bin/bash
set -euo pipefail

RUN_NAME="vllm_server"
PORT=18082
TABLE_FILE="slurm/tables/a8_vllm_server.table"
SERVER_INFO_FILE="logs/vllm_server.json"
REGISTRY_FILE="logs/vllm_endpoints.json"
MAX_ENDPOINTS=4
SKIP_RUN=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --table-file=*) TABLE_FILE="${1#*=}"; shift ;;
    --server-info-file=*) SERVER_INFO_FILE="${1#*=}"; shift ;;
    --registry-file=*) REGISTRY_FILE="${1#*=}"; shift ;;
    --max-endpoints=*) MAX_ENDPOINTS="${1#*=}"; shift ;;
    -y|--yes)
      SKIP_RUN=false
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/submit_for_vllm.sh [options]
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
inner_cmd+="python vllm/server.py "
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
  --install-file=vllm/install_for_vllm.sh \
  $([[ "$SKIP_RUN" = false ]] && echo "-y") \
  "$TABLE_FILE"
