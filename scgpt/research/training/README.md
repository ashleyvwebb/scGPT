# Training Module

This directory contains the implementation of the model fine-tuning pipeline, along with the scripts required to execute training on a compute cluster.

The training process extends a pre-trained scGPT model using a masking-based objective, with masking behaviour controlled dynamically through scheduling.

---

## Overview

Training follows a structured pipeline:

1. Load a pre-trained model  
2. Load and concatenate datasets  
3. Apply preprocessing (normalisation, log transform, binning)  
4. Tokenise gene expression data  
5. Train using masked reconstruction  
6. Save checkpoints and final model  

The training objective is masked mean squared error (MSE), computed only over masked positions.

---

## Core Training Script

### train_two_stage.py

This script performs model fine-tuning using a dynamic masking schedule.

Key components:

- Model loading: Loads a pre-trained Transformer model and vocabulary  
- Data pipeline: Concatenates datasets and applies preprocessing  
- Tokenisation: Converts gene expression into fixed-length sequences  
- Training loop: Applies masking, runs forward pass, computes loss, and updates weights  
- Evaluation: Computes validation loss using the same masking behaviour  
- Checkpointing: Saves best-performing and final models  

---

## Masking Strategy

Training uses a DynamicSchedule to control masking behaviour over epochs.

Two-stage configuration:

- Stage 1: Uniform masking  
- Stage 2: Cancer-weighted masking  

The schedule is defined as:

policies = [UniformMaskingPolicy(), CancerWeightedMaskingPolicy(...)]  
mask_ratios = [MASK_RATIO, MASK_RATIO]  
switch_epochs = [EPOCHS_STAGE1]  

This results in:

epoch < switch_epoch → uniform masking  
epoch ≥ switch_epoch → cancer-weighted masking  

Unlike a hard two-stage implementation, this approach uses a single continuous training loop with dynamic policy selection.

---

## Training Loop

At each epoch:

1. Retrieve current policy and mask ratio from the schedule  
2. For each batch:
   - Sample mask positions using the policy  
   - Replace masked values with a mask token  
   - Run model forward pass  
   - Compute masked MSE loss  
   - Backpropagate and update weights  

Validation uses the same masking configuration as the current epoch.

---

## Dataset Handling

Datasets are loaded from preprocessed `.h5ad` files and concatenated.

For the human model:
- blood  
- lung  
- blood-cancer  
- lung-cancer  

For the pancancer model:
- blood-cancer  
- lung-cancer  

Data is split into training and validation sets using a random permutation.

---

## Preprocessing

Each dataset undergoes:

- Library size normalisation (sum to 10,000)  
- Log1p transformation  
- Discretisation into 51 bins  

The processed values are stored in `adata.layers["X_binned"]`.

---

## Tokenisation

Tokenisation converts gene expression into model-compatible sequences:

- Genes not in the vocabulary are removed  
- Sequences are padded to a fixed length  
- Padding tokens are excluded during masking and loss computation  

---

## Model Output

The model predicts expression values for masked genes.

Loss is computed only over masked positions:

masked_mse_loss(predictions, targets, mask)

---

## Checkpointing

Models are saved in the `models/` directory.

Each checkpoint includes:
- best_model.pt (model weights)  
- vocab.json (tokeniser vocabulary)  
- args.json (model configuration)  

Best models are selected based on validation loss.

---

## Training Script (Cluster)

### train_scgpt.sh

This script submits a training job using SLURM.

Usage:

bash train_scgpt.sh <base_model>

Where `<base_model>` is one of:
- pretrained_human  
- pretrained_pancancer  

The script:
- Allocates GPU resources  
- Runs the training module  
- Logs output to the logs directory  

:contentReference[oaicite:0]{index=0}

---

## Models Directory

The `models/` directory stores all trained models and checkpoints.

Naming convention:

<model_prefix>_best → best validation checkpoint  
<model_prefix> → final trained model  

Each model directory contains all information required for inference and reuse.

---

## Notes

- Training is performed on GPU if available  
- Masking is applied only to valid (non-padding) tokens  
- Masking behaviour changes dynamically across epochs  
- Validation uses the same masking configuration as training  
- Optimisation is continuous across stages (no reset between stages)  

---

## Summary

This module provides a complete training pipeline for:

- Fine-tuning scGPT models  
- Experimenting with masking strategies  
- Producing reusable trained models  

It is designed for scalability, reproducibility, and controlled experimentation.