#!/usr/bin/env bash
set -euo pipefail

dataset_to_max() {
    local dataset="$1"
    case "$dataset" in
        db-bio) echo "2420" ;;
        tab) echo "1268" ;;
        reddit) echo "525" ;;
        *) return 1 ;;
    esac
}

run_count_mode() {
    local dataset=""
    local max_rows=""
    local file_input=""

    echo "Enter the dataset name (db-bio, tab, reddit):"
    read -r dataset

    if ! max_rows="$(dataset_to_max "$dataset")"; then
        echo "Unknown dataset: $dataset"
        exit 1
    fi

    echo "Enter one or more file paths/globs (leave blank to use stdin):"
    read -r file_input

    if [[ -n "$file_input" ]]; then
        # shellcheck disable=SC2086
        scripts/count.sh --max "$max_rows" $file_input
        return
    fi

    if [[ -t 0 ]]; then
        echo "No stdin detected. Example:"
        echo "  wc -l /path/to/files/*.jsonl | scripts/count.sh --max $max_rows"
        exit 1
    fi

    scripts/count.sh --max "$max_rows"
}

run_watch_mode() {
    watch -n 1 -x squeue -u yay
}

echo "Select an action:"
echo "1) Count row progress for a dataset"
echo "2) Watch squeue"
echo "Enter your choice [default: 2]:"
read -r choice
choice="${choice:-2}"

case "$choice" in
    1) run_count_mode ;;
    2) run_watch_mode ;;
    *)
        echo "Invalid choice: $choice"
        exit 1
        ;;
esac