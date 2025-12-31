#!/bin/bash

declare -A DATASET_LENGTHS=( ["reddit"]=525 ["tab"]=127 )

show_menu() {
    echo "Select an action:"
    echo "1) Watch a dataset's outputs lengths"
    echo "2) Watch the logs of a file"
    echo "3) Watch all squeues"
    echo "Enter the number of your choice:"
}

show_menu
read -r CHOICE

case "$CHOICE" in
    1)
        echo "Available datasets:"
        i=1
        for ds in "${!DATASET_LENGTHS[@]}"; do
            echo "$i) $ds"
            DATASETS[$i]=$ds
            ((i++))
        done
        echo "Enter the number of the dataset:"
        read -r DS_CHOICE
        DATASET="${DATASETS[$DS_CHOICE]}"
        LEN="${DATASET_LENGTHS[$DATASET]}"
        if [[ -z "$DATASET" || -z "$LEN" ]]; then
            echo "Unknown dataset selection"
            exit 1
        fi
        watch -n 10 "wc -l outputs/$DATASET/*/*.jsonl | awk '\$1 != $LEN {print}'"
        ;;
    2)
        mapfile -t task_groups < <(ls slurm/tables | sed 's/\.table$//')
        if [[ ${#task_groups[@]} -eq 0 ]]; then
            echo "No task groups found"
            exit 1
        fi
        echo "Available task groups:"
        for i in "${!task_groups[@]}"; do
            idx=$((i+1))
            TASK_GROUP_IS_IN_LOGS=false
            if [[ -d "logs/${task_groups[$i]}" ]]; then
                TASK_GROUP_IS_IN_LOGS=true
            fi
            echo "$([[ "$TASK_GROUP_IS_IN_LOGS" == false ]] && echo "-)" || echo "$idx)") ${task_groups[$i]}"
        done
        echo "Enter the number of the task group to select:"
        read -r TASK_GROUP_IDX
        if ! [[ "$TASK_GROUP_IDX" =~ ^[0-9]+$ ]] || ((TASK_GROUP_IDX < 1 || TASK_GROUP_IDX > ${#task_groups[@]})); then
            echo "Invalid task group selection"
            exit 1
        fi
        TASK_GROUP="${task_groups[$((TASK_GROUP_IDX-1))]}"
        TABLE_FILE="slurm/tables/$TASK_GROUP.table"
        mapfile -t jobs < <(awk -F'|' '{print $1}' "$TABLE_FILE")
        if [[ ${#jobs[@]} -eq 0 ]]; then
            echo "No jobs found in $TABLE_FILE"
            exit 1
        fi
        echo "Available jobs in $TASK_GROUP:"
        for i in "${!jobs[@]}"; do
            idx=$((i+1))
            echo "$idx) ${jobs[$i]}"
        done
        echo "Enter the number of the job to select:"
        read -r JOB_IDX
        if ! [[ "$JOB_IDX" =~ ^[0-9]+$ ]] || ((JOB_IDX < 1 || JOB_IDX > ${#jobs[@]})); then
            echo "Invalid job selection"
            exit 1
        fi
        JOB_NUM="${jobs[$((JOB_IDX-1))]}"
        LOG_DIR="logs/$TASK_GROUP"
        mapfile -t log_files < <(find "$LOG_DIR" -maxdepth 1 -type f -regextype posix-extended -regex "$LOG_DIR/$((JOB_IDX-1))_[^/]+\\.(out|err)" | sort)
        if [[ ${#log_files[@]} -eq 0 ]]; then
            echo "No matching log files found in $LOG_DIR"
            exit 1
        fi
        echo "Streaming all log files for job $JOB_NUM in $LOG_DIR:"
        watch -n 1 tail -n 40 "${log_files[@]}"
        ;;
    3)
        watch -n 1 -x squeue -u yay
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac