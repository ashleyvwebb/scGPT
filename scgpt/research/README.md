# Research Module

This directory contains the complete experimental pipeline for investigating masking strategies in scGPT for gene expression modelling.

It integrates data processing, masking strategy design, model training, and experimental evaluation into a unified workflow.

---

## Overview

The research pipeline consists of five main components:

- Data → dataset construction and preprocessing  
- Masking → definition of masking strategies and schedules  
- Training → model fine-tuning using masking objectives  
- Experiments → large-scale evaluation across models and datasets  
- Results → storage and analysis of outputs  

These components are designed to work together but remain modular, allowing independent experimentation and iteration.

---

## Pipeline Structure

The typical workflow follows:

1. Build datasets from CellxGene data  
2. Define masking strategies and schedules  
3. Train or fine-tune models  
4. Run experiments to generate predictions  
5. Aggregate and analyse results  

---

## Data Module

Handles all dataset construction and preprocessing.

Responsibilities:
- Download and structure CellxGene data  
- Build train/test splits  
- Ensure compatibility with scGPT preprocessing  
- Provide inspection and validation tools  

Outputs:
- Preprocessed `.h5ad` datasets  
- Gene sets (e.g. HVGs, cancer genes)  

---

## Masking Module

Defines how masking is performed during training and evaluation.

Includes:
- Masking policies (uniform, weighted, deterministic)  
- Gene set utilities  
- Dynamic masking schedules  

Enables:
- Standard MLM training  
- Targeted masking of biologically relevant gene sets  
- Multi-stage training strategies  

---

## Training Module

Implements model fine-tuning.

Responsibilities:
- Load pre-trained scGPT models  
- Apply preprocessing and tokenisation  
- Train using masked reconstruction objective  
- Dynamically switch masking strategies using schedules  
- Save checkpoints and final models  

Supports:
- Human-wide training  
- Cancer-focused training  

---

## Experiments Module

Runs large-scale evaluation of trained models.

Responsibilities:
- Apply masking policies to test data  
- Generate masked predictions per cell  
- Store batch-level outputs  
- Aggregate predictions across datasets  

Designed for:
- Parallel execution using SLURM  
- Systematic comparison across:
  - models  
  - datasets  
  - masking strategies  

---

## Results Module

Stores and processes outputs from experiments.

Includes:
- Batch-level prediction files  
- Aggregated prediction distributions  
- Evaluation plots  
- Summary statistics  

Outputs:
- Correlation metrics (Pearson, Spearman)  
- Visualisations (histograms, density plots)  
- CSV summaries for analysis  

---

## Key Design Principles

### Modular Structure
Each component is independent but interoperable, allowing flexible experimentation.

### Reproducibility
- Fixed preprocessing pipeline  
- Controlled masking strategies  
- Deterministic dataset splits  

### Scalability
- Batch-based processing  
- SLURM-based parallelisation  
- Efficient aggregation of results  

### Experimental Control
- Masking policies isolate specific gene subsets  
- Dynamic schedules enable staged training  
- Evaluation focuses on masked reconstruction behaviour  

---

## Typical Workflow

### 1. Prepare data
- Build datasets using the data module  
- Validate gene alignment and preprocessing  

### 2. Train models
- Run training using predefined masking schedules  
- Save checkpoints and final models  

### 3. Run experiments
- Generate predictions across datasets and models  
- Use different masking policies for evaluation  

### 4. Aggregate results
- Combine batch outputs  
- Compute correlations and generate plots  

### 5. Analyse outputs
- Compare performance across masking strategies  
- Interpret model behaviour  

---

## Notes

- All components assume compatibility with the scGPT vocabulary  
- Gene alignment is critical; genes not in the vocabulary are excluded  
- Masking is applied only to valid (non-padding) tokens  
- Evaluation focuses on masked positions only  
- Results are sensitive to preprocessing and binning choices  

---

## Summary

This directory provides a complete research framework for:

- Studying the effect of masking strategies on gene expression modelling  
- Comparing training regimes and evaluation behaviours  
- Producing reproducible and scalable experimental results  

It enables systematic exploration of how masking influences representation learning in scGPT.