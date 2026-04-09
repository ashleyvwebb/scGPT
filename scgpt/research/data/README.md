# Data Module

This directory contains all scripts and utilities for data acquisition, inspection, preprocessing, and dataset construction used in the scGPT experiments.

The pipeline builds structured datasets from the CellxGene Census and prepares them for training and evaluation.

---

## Overview

The data workflow consists of four stages:

1. Data acquisition (CellxGene Census → `.h5ad`)
2. Inspection and validation
3. Dataset construction (train/test splits)
4. Gene set analysis (cancer genes and HVGs)

The `cellxgene/` subdirectory contains the full data download and preprocessing pipeline.

---

## Core Scripts

### build_train_test_dataset.py

Constructs training and testing datasets from downloaded `.h5ad` partitions.

- Loads data per query (e.g. blood, lung, cancer variants)
- Splits into train/test subsets using a fixed seed
- Saves:
  - Per-query datasets
  - Aggregated datasets across all queries

---

### cxg_loader.py

Handles loading and merging of `.h5ad` partitions.

- Locates partition files for a query
- Loads `.h5ad` files into AnnData
- Optionally subsets cells
- Merges partitions into a single dataset

Ensures consistent gene naming and tracks source partitions.

---

### alignment.py

Provides utilities for gene-level alignment and overlap analysis.

- Extracts gene names from datasets
- Computes overlap with:
  - Model vocabulary
  - Cancer gene sets

Used to validate dataset compatibility and biological relevance.

---

### inspect_downloaded_data.py

Performs exploratory inspection of downloaded datasets.

Outputs:
- Dataset shape (cells × genes)
- Metadata structure (`obs`, `var`)
- Vocabulary overlap
- Cancer gene overlap

Supports subsampling and optional JSON output.

---

### inspect_downloaded_data.sh

SLURM script for running dataset inspection on a compute cluster.

---

## Cancer Gene Utilities

### cancer_gene/extract_cancer_genes.py

Extracts a curated cancer gene list from a source CSV.

- Filters genes (e.g. Tier 1)
- Outputs a deduplicated gene list

Used for cancer-focused masking and evaluation.

---

## Highly Variable Genes (HVGs)

### HVGs/high_variance_genes.py

Identifies highly variable genes using Scanpy.

- Uses `seurat_v3`
- Selects top-N genes
- Saves gene lists
- Produces diagnostic plots

---

### HVGs/hvg_cancer_overlap.py

Computes overlap between HVGs and cancer gene sets.

- Supports multiple HVG sets
- Outputs summary statistics to CSV

---

## cellxgene Submodule

Contains the full pipeline for building datasets from the CellxGene Census, including:

- Query configuration
- Index construction
- Partitioned downloads
- Data transformation

Refer to `cellxgene/README.md` for full details.

---

## Typical Workflow

1. Download data using scripts in `cellxgene/`
2. Inspect datasets with `inspect_downloaded_data.py`
3. Build datasets using `build_train_test_dataset.py`
4. Generate gene sets (cancer genes and HVGs)
5. Analyse overlaps between gene sets

---

## Notes

- All pipelines assume compatibility with the scGPT vocabulary
- Gene alignment is critical; genes not in the vocabulary are excluded downstream
- Dataset consistency is enforced during loading and merging