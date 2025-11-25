#!/usr/bin/env bash
set -euo pipefail

DEFAULT_STATE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/task.state"

usage() {
    echo "Usage:" >&2
    echo "  $0 --init N --cmd \"<base-cmd>\" [--state path]" >&2
    echo "  $0 --incr [--state path]" >&2
    exit 2
}

is_number() { [[ ${1:-} =~ ^[0-9]+$ ]]; }

extend_cmd() {
    local cmd="$1" ntasks="$2" idx="$3"
    if [[ "$ntasks" -eq 1 ]]; then
        echo "${cmd}"
    else
        echo "${cmd} --start ${idx} --step ${ntasks}"
    fi
}

STATE_FILE="$DEFAULT_STATE_FILE"
MODE=""
NTASKS=""
BASE_CMD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --state)
            shift
            STATE_FILE="${1:-}"
            [[ -z "$STATE_FILE" ]] && usage
            ;;
        --init)
            MODE="init"
            shift
            NTASKS="${1:-}"
            ;;
        --incr)
            MODE="incr"
            ;;
        --cmd)
            shift
            BASE_CMD="${1:-}"
            ;;
        *)
            usage
            ;;
    esac
    shift
done

[[ -z "$MODE" ]] && usage

if [[ "$MODE" == "init" ]]; then
    if ! is_number "$NTASKS" || [[ -z "$BASE_CMD" ]]; then usage; fi
    printf '%s\n%s\n%s\n' "$NTASKS" 0 "$BASE_CMD" > "$STATE_FILE"
    exit 0
fi

if [[ "$MODE" == "incr" ]]; then
    if [[ ! -f "$STATE_FILE" ]]; then
        echo "State file not found. Initialize first." >&2
        exit 1
    fi
    # Portable read (macOS bash 3.x lacks mapfile)
    {
        IFS= read -r NTASKS || true
        IFS= read -r CURRENT || true
        IFS= read -r BASE_CMD || true
    } < "$STATE_FILE"
    if [[ -z "$NTASKS" || -z "$CURRENT" || -z "$BASE_CMD" ]]; then
        echo "Corrupt state file." >&2
        exit 1
    fi
    if ! is_number "$NTASKS" || ! is_number "$CURRENT"; then
        echo "Invalid numeric values in state." >&2
        exit 1
    fi
    if (( CURRENT >= NTASKS )); then
        echo "All tasks have been assigned." >&2
        exit 1
    fi
    NEXT_ID="$CURRENT"
    NEW_CURRENT=$((CURRENT + 1))
    printf '%s\n%s\n%s\n' "$NTASKS" "$NEW_CURRENT" "$BASE_CMD" > "$STATE_FILE"
    FULL_CMD=$(extend_cmd "$BASE_CMD" "$NTASKS" "$NEXT_ID")
    echo "Executing: $FULL_CMD" >&2
    eval "$FULL_CMD"
    exit $?
fi

usage