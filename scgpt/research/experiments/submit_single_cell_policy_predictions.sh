#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5960
#SBATCH --time=00:20:00


set -euo pipefail

PROJECT_ROOT="/springbrook/share/bioinf/csuxfw/scGPT/scgpt"

H5AD_ROOT="${PROJECT_ROOT}/research/data/cellxgene/h5ad"
MODEL_DIR="${PROJECT_ROOT}/research/pretrained_models/whole_human"
CANCER_GENE_PATH="${PROJECT_ROOT}/research/data/cancer_genes/cancer_gene_list.txt"
OUTPUT_DIR="${PROJECT_ROOT}/research/results/single_cell_policy_predictions/lung_cancer"

QUERY="lung-cancer"
MAX_FILES=1
SUBSET_N_CELLS=1
MASK_RATIO=0.15
MASK_TOKEN_VALUE=-1
PAD_VALUE=-2

mkdir -p "${OUTPUT_DIR}"

python -m scgpt.research.experiments.run_all_single_cell_policy_predictions \
  --h5ad-root "${H5AD_ROOT}" \
  --query "${QUERY}" \
  --model-dir "${MODEL_DIR}" \
  --cancer-gene-path "${CANCER_GENE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-files "${MAX_FILES}" \
  --subset-n-cells "${SUBSET_N_CELLS}" \
  --mask-ratio "${MASK_RATIO}" \
  --mask-token-value "${MASK_TOKEN_VALUE}" \
  --pad-value "${PAD_VALUE}" \
  --cell-index 0 \
  --device cpu