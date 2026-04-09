# Experiments Module

This directory contains all scripts required to **run, manage, and analyse experiments** for evaluating masking strategies in the scGPT framework.

The experiments pipeline operates in three stages:

1. **Inference** (single-cell masked prediction)
2. **Aggregation** (combine batch outputs)
3. **Analysis & visualisation** (plots + metrics)

---

## Overview

Experiments are structured around evaluating different:

- **Models** (e.g. pretrained, fine-tuned variants)
- **Queries** (tissue / cancer datasets)
- **Masking policies** (uniform, HVG, cancer genes)

Each experiment produces:
- Per-cell predictions (batched)
- Aggregated predictions
- Evaluation plots and correlation metrics

---

## Core Components

### run_single_cell_policy_predictions.py

Runs **masked prediction inference** for a batch of cells.

For each cell:
- Preprocesses gene expression data
- Tokenises input into scGPT format
- Applies masking policy
- Runs model forward pass
- Stores:
  - Target values (true expression)
  - Predicted values
  - Gene names

Key characteristics:
- Operates on **fixed-size batches of cells**
- Supports multiple masking policies:
  - Uniform masking
  - HVG-only masking
  - Cancer gene masking
- Outputs results as JSON per batch

This is the **core experiment execution script**

---

### submit_single_cell_policy_predictions.sh

SLURM job script for parallel execution.

- Splits dataset into batches of size 100
- Each array job processes one batch
- Passes:
  - Query (dataset)
  - Model name

Enables scalable execution across compute nodes

---

### run_full_experiment.sh

Top-level orchestration script.

- Iterates over:
  - All queries
  - All models
- Submits SLURM jobs for each combination
- Tracks job IDs and waits for completion
- Automatically triggers aggregation after all jobs finish

This script runs the **entire experimental pipeline end-to-end** 

---

## Aggregation & Analysis

### run_full_aggregation.py

Aggregates batch-level outputs into experiment-level results.

For each:
- model / query / policy

It:
- Loads all batch JSON files
- Concatenates predictions and targets
- Computes:
  - Pearson correlation
  - Spearman correlation

Generates:
- 2D histogram plots (predicted vs target)
- Split plots:
  - Low expression region
  - High expression region
- Target distribution histograms

Key details:
- Applies jitter to targets to reduce binning artefacts
- Uses log-scaled density plots for visibility
- Saves all outputs per policy directory

Outputs a global summary file:
- `summary.json`

---

### combine_results.py

Converts aggregated JSON results into CSV format.

- Input: `summary.json`
- Output: `summary.csv`

Facilitates:
- Filtering by model / query / policy
- Analysis in tools such as Excel


---

## Output Structure

Results are written to:
results/<expr_set_name>/<model>/<query>/<policy>/
    batch_*.json
    aggregated.png
    aggregated_low.png
    aggregated_high.png
    target_hist.png


Global outputs:
results/<expr_set_name>/
    summary.json
    summary.csv

---

## Typical Workflow

### 1. Run full experiment

```bash
bash run_full_experiment.sh
```

This will:
- submit all jobs
- Wait for completion
- Run aggregation automatically

### 2. (Optinal) Run aggregation manually

```bash
python -m scgpt.research.experiments.run_full_aggregation
```

### 3. Convert results to CSV
```bash
python combine_results.py
```

## Notes

- Experiments are fully batch-parallelised via SLURM arrays
- Each batch operates independently → fault-tolerant
- Aggregation assumes all batch outputs are present
- Masking is applied only to valid (non-padding) tokens
- Evaluation is based on masked positions only (MLM objective)


## Key Design Decisions 
- Batch-based execution: enables scaling to large datasets
- Policy abstraction: allows direct comparison of masking strategies
- Post-hoc aggregation: avoids memory bottlenecks during inference
- Histogram-based evaluation: robust to high-density prediction regions

## Summary
This module provides a complete pipeline for:
- Running large-scale masked prediction experiments
- Comparing masking strategies across datasets and models
- Producing quantitative and visual evaluation outputs
It is designed for scalability, reproducibility, and systematic analysis.