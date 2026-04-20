#!/bin/bash

FILE_PATH=""
PRINT_SPANS=false
LINE_NUMS=()

if [ $# -eq 0 ]; then
    echo "Usage: $0 -f <file_path> [-s] [line_num]..."
    exit 1
fi

while getopts "f:s" opt; do
    case $opt in
        f) FILE_PATH="$OPTARG" ;;
        s) PRINT_SPANS=true ;;
        *) echo "Invalid option"; exit 1 ;;
    esac
done

shift $((OPTIND - 1))
if [ $# -gt 0 ]; then
    for arg in "$@"; do
        if [[ "$arg" =~ ^[0-9]+$ ]]; then
            LINE_NUMS+=("$arg")
        elif [[ "$arg" =~ ^[0-9]+-[0-9]+$ ]]; then
            start=$(echo "$arg" | cut -d- -f1)
            end=$(echo "$arg" | cut -d- -f2)
            for ((i = start; i <= end; i++)); do
                LINE_NUMS+=("$i")
            done
        else
            echo "Error: invalid line number format '$arg'. Use single numbers or ranges (e.g., 5-10)."
            exit 1
        fi
    done
fi

if [ ${#LINE_NUMS[@]} -eq 0 ]; then
    echo "Error: no valid line numbers provided."
    exit 1
fi

separator="-----------------------------"
echo "Processing files in ${FILE_PATH} with line numbers: ${LINE_NUMS[*]}"
for file in $(ls ${FILE_PATH}); do
    echo "${separator}"
    for line_num in "${LINE_NUMS[@]}"; do
        line_num=$((line_num + 1)) # Adjust for 1-based indexing
        sed -nE "${line_num}s/^.*\"idx\":[[:space:]]*[0-9]+,[[:space:]]*\"text\":[[:space:]]*\"(([^\"\\\\]|\\\\.)*?)\".*$/\1/p" "${file}"
    done
done