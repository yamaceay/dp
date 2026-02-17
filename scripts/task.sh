#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --cmd \"<base-cmd>\" [--task-id N]" >&2
    exit 2
}

is_number() { [[ ${1:-} =~ ^[0-9]+$ ]]; }

BASE_CMD=""
TASK_ID="${TASK_WITHIN_JOB:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cmd)
            shift
            BASE_CMD="${1:-}"
            ;;
        --task-id)
            shift
            TASK_ID="${1:-}"
            ;;
        --run-timestamp)
            shift
            RUN_TIMESTAMP="${1:-}"
            ;;
        *)
            usage
            ;;
    esac
    shift
done

if [[ -z "$BASE_CMD" ]]; then
    usage
fi
if ! is_number "$TASK_ID"; then
    echo "Invalid --task-id: $TASK_ID" >&2
    exit 1
fi

export TASK_ID
export TASK_WITHIN_JOB="$TASK_ID"
export RUN_TIMESTAMP

FULL_CMD="${BASE_CMD//\{task_id\}/$TASK_ID}"
FULL_CMD="${FULL_CMD//\{run_timestamp\}/$RUN_TIMESTAMP}"
echo "Executing (task_id=${TASK_ID}): $FULL_CMD" >&2
eval "$FULL_CMD"
