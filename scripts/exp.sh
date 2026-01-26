TYPE=""
DATASET=""
METRIC=""
LABEL=""
SKIP_RUN=false

while [ $# -gt 0 ]; do
    case "$1" in
        -t) TYPE="$2"; shift 2 ;;
        -d) DATASET="$2"; shift 2 ;;
        -m) METRIC="$2"; shift 2 ;;
        -l) LABEL="$2"; shift 2 ;;
        --skip_run) SKIP_RUN=true; shift ;;
        *) echo "Usage: $0 -t <type> -d <dataset> [-m <metric>] [-l <label>] [--skip_run]" >&2
           exit 1 ;;
    esac
done

if [ -z "$TYPE" ]; then
    echo "Error: TYPE (-t) is required. Must be one of utility, privacy, divergence, runtime."
    exit 1
fi
if [ -z "$DATASET" ]; then
    echo "Error: DATASET (-d) is required. Must be one of reddit, tab."
    exit 1
fi

if [ "$TYPE" != "utility" ] && [ "$TYPE" != "privacy" ] && [ "$TYPE" != "divergence" ] && [ "$TYPE" != "runtime" ]; then
    echo "Invalid TYPE: $TYPE. Must be one of utility, privacy, divergence, runtime."
    exit 1
fi
if [ "$DATASET" != "reddit" ] && [ "$DATASET" != "tab" ] && [ "$DATASET" != "db_bio" ]; then
    echo "Invalid DATASET: $DATASET. Must be one of reddit, tab, db_bio."
    exit 1
fi

result_set="${TYPE}_${DATASET}"
metric="${TYPE}_${METRIC}"

cmds=()
if [ "$TYPE" = "utility" ]; then
    if [ "$SKIP_RUN" = false ]; then
        cmds+=("python3 run.py ${TYPE} --config configs/experiments/${TYPE}_${DATASET}_${LABEL}.yaml")
    fi
    result_set+="_${LABEL}"
    metric+="_${LABEL}"
elif [ "$TYPE" = "privacy" ]; then 
    if [ "$SKIP_RUN" = false ]; then
        cmds+=("python3 run.py ${TYPE} --config configs/experiments/${TYPE}_${DATASET}_tri.yaml")
    fi
elif [ "$TYPE" = "divergence" ]; then
    if [ "$SKIP_RUN" = false ]; then
        cmds+=("python3 run.py ${TYPE} --config configs/experiments/${TYPE}_${DATASET}_${METRIC}.yaml")
    fi
    result_set+="_${METRIC}"
fi

cmds+=("python3 visualize/${TYPE}.py")
cmds+=("python3 visualize/summary_plots.py --results-set ${result_set} --methods-set lrec_paper --dataset ${DATASET} --experiment ${TYPE} --metric ${metric} --debug")

for cmd in "${cmds[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done