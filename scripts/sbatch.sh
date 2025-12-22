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
            echo "Usage: $0 [--mail-to=email] [--max-concurrent=3] job_file_name table_file"
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

[[ -z "$TABLE_FILE" ]] && TABLE_FILE="jobs.table"

mkdir -p logs

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
            ((idx++))
        fi
    fi
done < "$TABLE_FILE"

NUM_JOBS=${#job_names[@]}
if [[ $NUM_JOBS -eq 0 ]]; then
    echo "No valid jobs found in $TABLE_FILE" >&2
    exit 1
fi

MAX_IDX=$((NUM_JOBS - 1))
echo "Found $NUM_JOBS jobs, creating array job 0-${MAX_IDX}%${MAX_CONCURRENT}" >&2

TASK_LINES=""
if [[ $MAX_TASKS -gt 1 ]]; then
    TASK_LINES="#SBATCH --ntasks=${MAX_TASKS}
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none"
fi

MAIL_LINES=""
if [[ -n "$MAILTO" ]]; then
    MAIL_LINES="#SBATCH --mail-type=ALL
#SBATCH --mail-user=${MAILTO}"
fi

EXTRA_LINES=""
declare -a _extra
[[ -n "$MAIL_LINES" ]] && _extra+=("$MAIL_LINES")
[[ -n "$TASK_LINES" ]] && _extra+=("$TASK_LINES")

if [[ ${#_extra[@]} -gt 0 ]]; then
    EXTRA_LINES="$(printf '%s\n' "${_extra[@]}")"
    EXTRA_LINES="${EXTRA_LINES%$'\n'}"
fi

cat > "jobs/${FILE_NAME}.sbatch" <<EOF
#!/bin/bash

#SBATCH --array=0-${MAX_IDX}%${MAX_CONCURRENT}
#SBATCH --job-name=${FILE_NAME}
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
${EXTRA_LINES}

job_names=(
EOF

for name in "${job_names[@]}"; do
    echo "  \"$name\"" >> "jobs/${FILE_NAME}.sbatch"
done

cat >> "jobs/${FILE_NAME}.sbatch" <<EOF
)

job_cmds=(
EOF

for cmd in "${job_cmds[@]}"; do
    echo "  \"$cmd\"" >> "jobs/${FILE_NAME}.sbatch"
done

cat >> "jobs/${FILE_NAME}.sbatch" <<'EOF'
)

JOB_NAME="${job_names[$SLURM_ARRAY_TASK_ID]}"
JOB_CMD="${job_cmds[$SLURM_ARRAY_TASK_ID]}"

echo "Running job $SLURM_ARRAY_TASK_ID: $JOB_NAME"
echo "Command: $JOB_CMD"

srun -K \
  --container-image=/enroot/python+3.10.4-buster.sqsh \
  --container-mounts=`pwd`:`pwd` \
  --container-workdir=`pwd` \
  scripts/install.sh $JOB_CMD 
EOF

echo "Wrote jobs/${FILE_NAME}.sbatch with $NUM_JOBS tasks"
echo "Submit with: sbatch jobs/${FILE_NAME}.sbatch"
