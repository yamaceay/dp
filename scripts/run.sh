#!/bin/bash

MAILTO=""
TABLE_FILE=""
MAX_CONCURRENT=1
MAX_TASKS=1
PARTITION="RTXA6000"
YES=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --mail-to=*) MAILTO="${1#*=}"; shift ;;
        --max-concurrent=*) MAX_CONCURRENT="${1#*=}"; shift ;;
        --max-tasks=*) MAX_TASKS="${1#*=}"; shift ;;
        --partition=*) PARTITION="${1#*=}"; shift ;;
        -y) YES=1; shift ;;
        -h)
            echo "Usage: $0 [--mail-to=email] [--max-concurrent=3] [--max-tasks=1] table_file"
            exit 0
            ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$TABLE_FILE" ]]; then
                TABLE_FILE="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$TABLE_FILE" ]]; then
    echo "Error: table file name is required" >&2
    exit 1
fi

FILE_NAME="$(basename "${TABLE_FILE%.*}")"

TARGET_FILE="slurm/sbatches/${FILE_NAME}.sbatch"

[[ -z "$TABLE_FILE" ]] && TABLE_FILE="jobs.table"

mkdir -p logs slurm/sbatches slurm/states slurm/states/${FILE_NAME} logs/${FILE_NAME}

if [[ ! -f "$TABLE_FILE" ]]; then
    echo "Table file not found: $TABLE_FILE" >&2
    exit 1
fi

declare -a job_names
declare -a job_cmds
idx=0

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line//[[:space:]]/}" || "${line:0:1}" == "#" ]] && continue

    IFS='|' read -r -a parts <<< "$line"
    NAME="$(echo "${parts[0]}" | xargs)"
    if [[ -z "$NAME" ]]; then
        echo "Skipping line with empty job name: $line" >&2
        continue
    fi

    if [[ ${#parts[@]} -gt 1 ]]; then
        CMD="$(echo "${parts[1]}" | xargs)"
        if [[ -n "$CMD" ]]; then
            STATE_FILE_NAME="${FILE_NAME}/${NAME}_${idx}"
            job_names[$idx]="$STATE_FILE_NAME"
            job_cmds[$idx]="$CMD"
            scripts/task.sh --init "$MAX_TASKS" --cmd "$CMD" --state "slurm/states/${STATE_FILE_NAME}.state"
            echo "Initialized state for ${STATE_FILE_NAME} with ${MAX_TASKS} parallel tasks"
            ((idx++))
        fi
    fi
done < "$TABLE_FILE"

NUM_JOBS=${#job_names[@]}
if [[ $NUM_JOBS -eq 0 ]]; then
    echo "No valid jobs found in $TABLE_FILE" >&2
    exit 1
fi

# Total array size is NUM_JOBS
TOTAL_TASKS=$((NUM_JOBS * MAX_TASKS))
MAX_IDX=$((TOTAL_TASKS - 1))
echo "Found $NUM_JOBS jobs with ${MAX_TASKS} parallel tasks each, creating array job 0-${MAX_IDX}%${MAX_CONCURRENT}" >&2

MAIL_LINES=""
if [[ -n "$MAILTO" ]]; then
    MAIL_LINES="#SBATCH --mail-type=ALL
#SBATCH --mail-user=${MAILTO}"
fi

TASK_LINES=""
if [[ -n "$MAX_TASKS" ]]; then
    TASK_LINES="#SBATCH --ntasks=${MAX_TASKS}
"
fi

EXTRA_LINES=""
if [[ -n "$TASK_LINES" && -n "$MAIL_LINES" ]]; then
    EXTRA_LINES="${TASK_LINES}${MAIL_LINES}"
elif [[ -n "$TASK_LINES" ]]; then
    EXTRA_LINES="${TASK_LINES}"
elif [[ -n "$MAIL_LINES" ]]; then
    EXTRA_LINES="${MAIL_LINES}"
fi

cat > "$TARGET_FILE" <<EOF
#!/bin/bash

#SBATCH --array=0-${MAX_IDX}%${MAX_CONCURRENT}
#SBATCH --job-name=${FILE_NAME}
#SBATCH --output=logs/%x/%a_%j.out
#SBATCH --error=logs/%x/%a_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=1
#SBATCH --mem=40GB
#SBATCH --time=10
${EXTRA_LINES}

# Map array task ID to job index and parallel task
MAX_TASKS=${MAX_TASKS}
JOB_IDX=\$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=\$((SLURM_ARRAY_TASK_ID % MAX_TASKS))

job_names=(
EOF

for name in "${job_names[@]}"; do
    echo "  \"$name\"" >> "$TARGET_FILE"
done

cat >> "$TARGET_FILE" <<'EOF'
)

JOB_NAME="${job_names[$JOB_IDX]}"
STATE_FILE="slurm/states/${JOB_NAME}.state"

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Job: $JOB_NAME (index $JOB_IDX)"
echo "Task within job: $TASK_WITHIN_JOB"
echo "State file: $STATE_FILE"

export TASK_WITHIN_JOB JOB_NAME STATE_FILE SLURM_ARRAY_TASK_ID

# Execute the task using task.sh --incr (it will run the command directly)
srun -K \
    --container-mounts="`pwd`:`pwd`,/netscratch/$USER:/netscratch/$USER" \
    --container-workdir="`pwd`" \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_24.01-py3.sqsh \
    --task-prolog="`pwd`/scripts/install.sh" scripts/task.sh --incr --state "$STATE_FILE"
EOF

if [[ $YES -eq 0 ]]; then
    echo "Wrote $TARGET_FILE with $NUM_JOBS jobs"
    echo "Submit with: sbatch $TARGET_FILE"
    echo ""
    read -p "Do you want to submit the job now? (Press Enter to continue, Ctrl+C to cancel): "
fi

sbatch "$TARGET_FILE"
exit $?
