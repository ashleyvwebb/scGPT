#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4591
#SBATCH --time=00:00:05

python -m scgpt.research.training.build_train_test_dataset