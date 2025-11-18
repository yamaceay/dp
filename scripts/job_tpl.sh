NAME=$1
CMD=$2
DIR=/home/yay/dp
EMAIL=yaay01@dfki.de

TPL_CONTENT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}-%j
#SBATCH --output=${DIR}/logs/${NAME}-%j.out
#SBATCH --error=${DIR}/logs/${NAME}-%j.err
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00
#SBATCH --mem=1G

cd ${DIR}
. .venv/bin/activate
${CMD}
EOF
)

echo "${TPL_CONTENT}" > ${DIR}/jobs/${NAME}.sbatch