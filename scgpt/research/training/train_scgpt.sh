#!/bin/bash
#SBATCH --job-name=scgpt_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=5960
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ashley.webb@warwick.ac.uk


echo "Starting job $SLURM_JOB_ID"
echo "Running on $(hostname)"

# Move to project dir
cd /springbrook/share/bioinf/csuxfw/scGPT/

# Run training
python -m scgpt.research.training.train_scgpt_two_stage

echo "Finished job"