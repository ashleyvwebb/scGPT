#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4591
#SBATCH --time=00:15:00
#SBATCH --array=0-199

set -euo pipefail

BATCH_SIZE=100
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE))

python -m scgpt.research.experiments.run_single_cell_policy_predictions \
  --start-idx "${START}" \
  --end-idx "${END}"