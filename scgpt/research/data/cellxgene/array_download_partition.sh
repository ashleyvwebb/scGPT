#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4591
#SBATCH --time=6:00:00
#SBATCH --array=1-4
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80,ARRAY_TASKS # Events to send email on, remove if you don't want this
#SBATCH --mail-user=ashley.webb@warwick.ac.uk
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

SCRIPT_DIR="/springbrook/share/bioing/csuxfw/GPT/scgpt/research/data/cellxgene"

INDEX_PATH="${SCRIPT_DIR}/index"
QUERY_PATH="${SCRIPT_DIR}/query_list.txt"
DATA_PATH="${SCRIPT_DIR}/h5ad"

mkdir -p $OUTPUT_PATH

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

echo "downloading ${query_name}"

./download_partition.sh ${query_name} ${INDEX_PATH} ${DATA_PATH}