#!/bin/bash

MAILTO=""
FILE_NAME=""
TABLE_FILE=""
MAX_CONCURRENT=3
MAX_TASKS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --mail-to=*) MAILTO="${1#*=}"; shift ;;
        --max-concurrent=*) MAX_CONCURRENT="${1#*=}"; shift ;;
        --max-tasks=*) MAX_TASKS="${1#*=}"; shift ;;
        -h)
            echo "Usage: $0 [--mail-to=email] [--max-concurrent=3] [--max-tasks=4] job_file_name table_file"
            exit 0
            ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$FILE_NAME" ]]; then
                FILE_NAME="$1"
            elif [[ -z "$TABLE_FILE" ]]; then
                TABLE_FILE="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$FILE_NAME" ]]; then
    echo "Error: job file name is required" >&2
    exit 1
fi

TARGET_FILE="slurm/sbatches/${FILE_NAME}.sbatch"

[[ -z "$TABLE_FILE" ]] && TABLE_FILE="jobs.table"

mkdir -p jobs logs slurm/states

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
            job_names[$idx]="${NAME}_${idx}"
            job_cmds[$idx]="$CMD"
            scripts/task.sh --init "$MAX_TASKS" --cmd "$CMD" --state "slurm/states/${NAME}_${idx}.state"
            echo "Initialized state for ${NAME}_${idx} with ${MAX_TASKS} parallel tasks"
            ((idx++))
        fi
    fi
done < "$TABLE_FILE"

NUM_JOBS=${#job_names[@]}
if [[ $NUM_JOBS -eq 0 ]]; then
    echo "No valid jobs found in $TABLE_FILE" >&2
    exit 1
fi

# Total array size is NUM_JOBS * MAX_TASKS
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
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
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
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
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
  --container-image=/enroot/python+3.10.4-buster.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER \
  --container-workdir=`pwd` \
  scripts/install.sh scripts/task.sh --incr --state "$STATE_FILE"
EOF

echo "Wrote $TARGET_FILE with $NUM_JOBS jobs × ${MAX_TASKS} parallel tasks = ${TOTAL_TASKS} total tasks"
echo "Submit with: sbatch $TARGET_FILE"