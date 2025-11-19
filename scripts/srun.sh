#!/bin/bash

MAILTO=""
NAME=""
CMD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mail-to=*) MAILTO="${1#*=}"; shift ;;
        -h)
            echo "Usage: $0 [--mail-to=email] jobname \"command\""
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$NAME" ]]; then
                NAME="$1"
            elif [[ -z "$CMD" ]]; then
                CMD="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$NAME" ]] || [[ -z "$CMD" ]]; then
    echo "Error: jobname and command are required" >&2
    exit 1
fi

mkdir -p jobs logs

MAIL_LINES=""
if [[ -n "$MAILTO" ]]; then
    MAIL_LINES="--mail-type=ALL \\
    --mail-user=${MAILTO} \\"
fi

FILE=$(cat << EOF
#!/bin/bash

srun -K \
    --job-name ${NAME} \
    --output=logs/${NAME}_%j.out \
    --error=logs/${NAME}_%j.err \
    --partition=batch \
    --ntasks=1 \
    --cpus-per-task=10 \
    --gpus-per-task=1 \
    --mem-per-cpu=6G \
    ${MAIL_LINES}
    --container-image=/enroot/python+3.10.4-buster.sqsh \
    --container-mounts=\`pwd\`:\`pwd\` \
    --container-workdir=\`pwd\` \
    scripts/install.sh $CMD
EOF
)
echo "$FILE" > jobs/${NAME}.sh

chmod +x jobs/${NAME}.sh
echo "Created jobs/${NAME}.sh"