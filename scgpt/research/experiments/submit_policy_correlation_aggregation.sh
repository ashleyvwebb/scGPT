#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4591
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="/springbrook/share/bioinf/csuxfw/scGPT/scgpt"

RESULTS_DIR="${PROJECT_ROOT}/research/results/single_cell_policy_predictions/lung_cancer"
OUTPUT_DIR="${PROJECT_ROOT}/research/results/single_cell_policy_predictions/lung_cancer_summary"

mkdir -p "${OUTPUT_DIR}"

python -m research.experiments.run_policy_correlation_aggregation \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}"