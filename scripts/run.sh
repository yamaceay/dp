#!/bin/bash

MAILTO=""
TABLE_FILE=""
MAX_CONCURRENT=1
MAX_TASKS=1
INSTALL_FILE="scripts/install.sh"
PARTITION="RTXA6000"
YES=0

trim_whitespace() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mail-to=*) MAILTO="${1#*=}"; shift ;;
        --max-concurrent=*) MAX_CONCURRENT="${1#*=}"; shift ;;
        --max-tasks=*) MAX_TASKS="${1#*=}"; shift ;;
        --partition=*) PARTITION="${1#*=}"; shift ;;
        --install-file=*) INSTALL_FILE="${1#*=}"; shift ;;
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
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs slurm/sbatches logs/${FILE_NAME}

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
    NAME="$(trim_whitespace "${parts[0]}")"
    if [[ -z "$NAME" ]]; then
        echo "Skipping line with empty job name: $line" >&2
        continue
    fi

    if [[ ${#parts[@]} -gt 1 ]]; then
        CMD="$(trim_whitespace "${parts[1]}")"
        if [[ -n "$CMD" ]]; then
            job_names[$idx]="$NAME"
            job_cmds[$idx]="$CMD"
            ((idx++))
        fi
    fi
done < "$TABLE_FILE"

NUM_JOBS=${#job_names[@]}
if [[ $NUM_JOBS -eq 0 ]]; then
    echo "No valid jobs found in $TABLE_FILE" >&2
    exit 1
fi

TOTAL_TASKS=$((NUM_JOBS * MAX_TASKS))
MAX_IDX=$((TOTAL_TASKS - 1))
echo "Found $NUM_JOBS jobs with ${MAX_TASKS} parallel tasks each, creating array job 0-${MAX_IDX}%${MAX_CONCURRENT}" >&2

MAIL_LINES=""
if [[ -n "$MAILTO" ]]; then
    MAIL_LINES="#SBATCH --mail-type=ALL
#SBATCH --mail-user=${MAILTO}"
fi

EXTRA_LINES=""
if [[ -n "$MAIL_LINES" ]]; then
    EXTRA_LINES="${MAIL_LINES}"
fi

cat > "$TARGET_FILE" <<EOF
#!/bin/bash

#SBATCH --array=0-${MAX_IDX}%${MAX_CONCURRENT}
#SBATCH --job-name=${FILE_NAME}
#SBATCH --output=logs/%x/%j_%a.out
#SBATCH --error=logs/%x/%j_%a.err
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=1
#SBATCH --mem=40GB
#SBATCH --time=1440
${EXTRA_LINES}

MAX_TASKS=${MAX_TASKS}
JOB_IDX=\$((SLURM_ARRAY_TASK_ID / MAX_TASKS))
TASK_WITHIN_JOB=\$((SLURM_ARRAY_TASK_ID % MAX_TASKS))
RUN_TIMESTAMP="${RUN_TIMESTAMP}"

INSTALL_FILE="${INSTALL_FILE}"

job_names=(
EOF

for name in "${job_names[@]}"; do
    echo "  \"$name\"" >> "$TARGET_FILE"
done

cat >> "$TARGET_FILE" <<'EOF'
)

job_cmds=(
EOF

for cmd in "${job_cmds[@]}"; do
    escaped_cmd="${cmd//\\/\\\\}"
    escaped_cmd="${escaped_cmd//\"/\\\"}"
    echo "  \"$escaped_cmd\"" >> "$TARGET_FILE"
done

cat >> "$TARGET_FILE" <<'EOF'
)

JOB_NAME="${job_names[$JOB_IDX]}"
CMD="${job_cmds[$JOB_IDX]}"

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Job: $JOB_NAME (index $JOB_IDX)"
echo "Task within job: $TASK_WITHIN_JOB"
echo "Base command: $CMD"

export TASK_WITHIN_JOB JOB_NAME SLURM_ARRAY_TASK_ID RUN_TIMESTAMP

srun -K \
    --ntasks=1 \
    --container-mounts="`pwd`:`pwd`,/netscratch/$USER:/netscratch/$USER" \
    --container-workdir="`pwd`" \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_24.01-py3.sqsh \
    --task-prolog="`pwd`/${INSTALL_FILE}" scripts/task.sh --cmd "$CMD" --task-id "$TASK_WITHIN_JOB"
EOF

if [[ $YES -eq 0 ]]; then
    echo "Wrote $TARGET_FILE with $NUM_JOBS jobs"
    echo "Submit with: sbatch $TARGET_FILE"
    echo ""
    read -p "Do you want to submit the job now? (Press Enter to continue, Ctrl+C to cancel): "
fi

sbatch "$TARGET_FILE"
exit $?
