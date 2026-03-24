#!/bin/bash
#SBATCH --job-name=array_build_scb
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

QUERY_PATH="path/to/query.txt"

query_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $QUERY_PATH)

DATA_PATH="${SCRIPT_DIR}/h5ad"
OUTPUT_PATH="${SCRIPT_DIR}/scb"
VOCAB_PATH="${SCRIPT_DIR}/../../tokenizer/default_census_vocab.json"

echo "processing ${query_name}"
N=25000


mkdir -p $OUTPUT_PATH

echo "downloading to ${OUTPUT_PATH}"

python build_large_scale_data.py \
    --input-dir ${DATA_PATH} \
    --output-dir ${OUTPUT_PATH} \
    --vocab-file ${VOCAB_PATH} \
    --N ${N}
