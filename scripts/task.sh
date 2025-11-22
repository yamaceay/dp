#!/usr/bin/env bash

# Usage:
#   task.sh "<base-cmd>" --init N    # run base-cmd for task ids 0..N-1 (use {} in base-cmd to place id)
#   task.sh "<base-cmd>" --incr      # run base-cmd for next id and increment counter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
STATE_FILE="$SCRIPT_DIR/task.state"

usage() {
    echo "Usage:"
    echo "  $0 \"<base-cmd>\" --init N"
    echo "  $0 \"<base-cmd>\" --incr"
    exit 2
}

is_number() { [[ $1 =~ ^[0-9]+$ ]]; }

extend_cmd() {
    local cmd="$1"
    local ntasks="$2"
    local idx="$3"
}

if [[ $# -lt 2 ]]; then usage; fi

BASE_CMD="$1"
shift

case "$1" in
    --init)
        if [[ $# -ne 2 ]] || ! is_number "$2"; then usage; fi
        NTASKS="$2"
        echo "$NTASKS,0" > "$STATE_FILE"
        wait
        ;;
    --incr)
        if [[ $# -ne 1 ]]; then usage; fi
        if [[ ! -f "$STATE_FILE" ]]; then
            echo "State file not found. Initialize first with --init N." >&2
            exit 1
        fi
        IFS=',' read -r NTASKS CURRENT < "$STATE_FILE"
        if ! is_number "$CURRENT" || ! is_number "$NTASKS"; then
            echo "Invalid state file contents." >&2
            exit 1
        fi
        NEXT_ID="$CURRENT"
        echo "$NTASKS,$((CURRENT + 1))" > "$STATE_FILE"
        if [[ "$BASE_CMD" == *'{}'* ]]; then
            CMD="${BASE_CMD//\{\}/$NEXT_ID}"
            bash -c "$CMD" &
        else
            bash -c "$BASE_CMD $NEXT_ID" &
        fi
        wait
        ;;
    *)
        usage
        ;;
esac