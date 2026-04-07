#!/bin/bash
set -euo pipefail

QUERIES=("blood" "lung" "blood-cancer" "lung-cancer")

MODELS=(
    "pretrained_human"
    "pretrained_pancancer"
    "human_uniform"
    "human_cancer"
    "pancancer_uniform"
    "pancancer_cancer"
)

JOB_IDS=()

echo "Submitting jobs..."
for QUERY in "${QUERIES[@]}"; do 
    for MODEL in "${MODELS[@]}"; do 

        JOB_ID=$(sbatch submit_single_cell_policy_predictions.sh "$QUERY" "$MODEL" | awk '{print $4}')
        echo "Submitted ${QUERY} / ${MODEL} -> JobID=${JOB_ID}"

        JOB_IDS += ("${JOB_ID}")

    done
done

echo "All jobs submitted."
echo "Waiting for completion..."

while true; do 
    sleep 300

    RUNNING=$(squeue -h 0j $(IFS=,; echo "${JOB_IDS[*]}") | wc -l)

    if [ "$RUNNING" -eq 0 ]; then
        echo "All jobs completed."
        break
    else
        echo "$RUNNING jobs still running..."
    fi
done

echo "Starting aggregation..."

python run_full_aggregation.py

echo "Done."