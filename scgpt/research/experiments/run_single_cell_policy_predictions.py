from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import issparse

# Data loading + alignment utilities
from scgpt.research.data.cxg_loader import load_h5ad
from scgpt.research.data.alignment import get_gene_names

# Masking policies
from scgpt.research.masking.cancer_gene_sets import load_gene_set
from scgpt.research.masking.policies import (
    UniformMaskingPolicy,
    DeterministicMaskingPolicy,
    apply_mask_to_values
)

# Model + tokenisation
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch


# =======
# CONFIG
# =======

# Root of project (used to construct all paths)
PROJECT_ROOT = "/springbrook/share/bioinf/csuxfw/scGPT/scgpt"

# External resources
CANCER_GENE_PATH = f"{PROJECT_ROOT}/research/data/cancer_genes/cancer_gene_list.txt"
HVG_GENE_PATH    = f"{PROJECT_ROOT}/research/data/HVGs/hvg_genes_70.txt"

# Output directory for predictions
OUTPUT_DIR = f"{PROJECT_ROOT}/research/results/cancer_predictions/"

# Masking behaviour
MASK_RATIO = 0.15            # fraction of valid tokens to mask
MASK_TOKEN_VALUE = -1        # value used to replace masked entries
PAD_VALUE = -2               # padding value (must match model config)

DEVICE = "cpu"               # inference device


def load_hvg_genes(path):
    """Load HVG gene list into a set for O(1) membership checks."""
    with open(path, "r") as f:
        return set(line.strip() for line in f)


def build_policies(cancer_gene_set: set[str], hvg_gene_set: set[str]):
    """
    Build masking policies used during evaluation.

    - Uniform: random masking across all genes
    - Deterministic (HVG): only mask HVGs
    - Deterministic (COSMIC): only mask cancer genes
    """
    return [
        UniformMaskingPolicy(),
        DeterministicMaskingPolicy("HVG", hvg_gene_set),
        DeterministicMaskingPolicy("COSMIC", cancer_gene_set)
    ]


def load_model(model_dir: str, device: str):
    """
    Load trained Transformer model + vocabulary.

    Important:
    - Ensures required special tokens exist
    - Model config must match training config
    """
    model_dir = Path(model_dir)

    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # Load vocab and ensure required tokens exist
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Load architecture config
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead   = model_configs["nheads"]
    d_hid   = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    # Instantiate model
    model = TransformerModel(
        len(vocab),
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=PAD_VALUE,
        n_input_bins=51,
    )

    # Load weights
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)

    model.to(device)
    model.eval()  # ensure inference mode

    return model, vocab


def preprocess_adata(adata, n_bins: int = 51):
    """
    Apply scGPT-style preprocessing:
    - library size normalisation (1e4)
    - log1p transform
    - discretisation into bins

    Output stored in adata.layers["X_binned"].
    """
    adata = adata.copy()

    # Ensure gene names exist
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = get_gene_names(adata)

    # Matches preprocessing used during pretraining
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=n_bins,
        result_binned_key="X_binned",
    )

    preprocessor(adata)
    return adata


def prepare_single_cell_tokenized_input(
    adata,
    cell_index: int,
    vocab,
    max_seq_len: int = 1200,
    pad_token: str = "<pad>",
    pad_value: int = -2,
    append_cls: bool = False,
    include_zero_gene: bool = False,
):
    """
    Convert a single cell into tokenised model input.

    Key steps:
    - extract one cell
    - filter genes not in vocab (critical)
    - tokenize + pad to fixed length
    """

    # Sanity checks
    if "X_binned" not in adata.layers:
        raise ValueError("Run preprocessing first.")
    if "gene_name" not in adata.var.columns:
        raise ValueError("Missing gene names.")

    # Extract counts (handle sparse matrices)
    all_counts = (
        adata.layers["X_binned"].toarray()
        if issparse(adata.layers["X_binned"])
        else adata.layers["X_binned"]
    )

    genes = adata.var["gene_name"].tolist()

    # Preserve 2D shape (1, G) for tokenizer
    one_cell = np.asarray(all_counts[cell_index : cell_index + 1])

    # Filter to genes present in vocab (critical for alignment)
    in_vocab = np.array([g in vocab for g in genes], dtype=bool)
    one_cell = one_cell[:, in_vocab]
    genes = [g for g, keep in zip(genes, in_vocab) if keep]

    gene_ids = np.array(vocab(genes), dtype=int)

    # Tokenise + pad
    tokenized = tokenize_and_pad_batch(
        one_cell,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
    )

    # Extract 1D arrays (batch size = 1)
    tokenized_gene_ids = tokenized["genes"][0].cpu().numpy()
    tokenized_values   = tokenized["values"][0].cpu().numpy()

    # Build reverse vocab mapping (id → gene name)
    stoi = vocab.get_stoi()
    pad_token_id = stoi[pad_token]
    id_to_token = {v: k for k, v in stoi.items()}

    tokenized_gene_names = [
        id_to_token.get(int(gid), "<unk>")
        for gid in tokenized_gene_ids
    ]

    # Mask of valid (non-pad) positions
    valid_mask = tokenized_values != pad_value

    return {
        "tokenized_gene_ids": tokenized_gene_ids,
        "tokenized_values": tokenized_values,
        "tokenized_gene_names": tokenized_gene_names,
        "valid_mask": valid_mask,
        "pad_token_id": pad_token_id,
    }


def run_model_forward(
    model,
    gene_ids: np.ndarray,
    masked_values: np.ndarray,
    device: str,
    pad_token_id: int,
):
    """
    Forward pass for a single cell.

    Inputs must already be:
    - tokenised
    - padded
    - 1D (sequence length)
    """

    # Convert to tensors and add batch dimension
    input_gene_ids = torch.as_tensor(gene_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_values   = torch.as_tensor(masked_values, dtype=torch.float32, device=device).unsqueeze(0)

    # Identify padding positions
    src_key_padding_mask = input_gene_ids.eq(pad_token_id)

    with torch.no_grad():
        output_dict = model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            MVC=False,
            ECS=False,
        )
        pred = output_dict["mlm_output"]

    # Remove batch dimension
    pred = pred.squeeze(0)

    # Handle shape (L,1) → (L,)
    if pred.ndim == 2 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    return pred.cpu().numpy()


def run_one_batch(start_idx, end_idx, h5ad_path, model_dir, model_name, query):
    """
    Run evaluation over a slice of cells.

    Outputs JSON per masking policy containing:
    - targets (true values)
    - preds (model predictions)
    - genes (gene names)
    """

    # Create output directory structure
    run_output_dir = Path(OUTPUT_DIR) / model_name / query
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Load data + model
    adata = load_h5ad(h5ad_path)
    model, vocab = load_model(model_dir, DEVICE)

    # Load gene sets
    cancer_gene_set = load_gene_set(CANCER_GENE_PATH)
    hvg_gene_set    = load_hvg_genes(HVG_GENE_PATH)

    policies = build_policies(cancer_gene_set, hvg_gene_set)

    # Preprocess dataset once
    adata = preprocess_adata(adata)

    # Initialise storage
    batch_results = {
        p.name: {"targets": [], "preds": [], "genes": []}
        for p in policies
    }

    # Iterate over cells
    for cell_index in range(start_idx, min(end_idx, adata.n_obs)):

        prepared = prepare_single_cell_tokenized_input(
            adata=adata,
            cell_index=cell_index,
            vocab=vocab,
        )

        gene_ids     = prepared["tokenized_gene_ids"]
        values       = prepared["tokenized_values"]
        names        = prepared["tokenized_gene_names"]
        valid_mask   = prepared["valid_mask"]
        pad_token_id = prepared["pad_token_id"]

        for policy in policies:

            # Sample mask positions
            masking = policy.sample_mask(
                gene_names=names,
                values=values,
                mask_ratio=MASK_RATIO,
                valid_mask=valid_mask,
            )

            # Apply mask
            masked_values = apply_mask_to_values(
                values=values,
                mask=masking.mask,
                mask_token_value=MASK_TOKEN_VALUE,
            )

            # Run model
            pred = run_model_forward(
                model=model,
                gene_ids=gene_ids,
                masked_values=masked_values,
                device=DEVICE,
                pad_token_id=pad_token_id,
            )

            # Store only masked positions (MLM objective)
            batch_results[policy.name]["targets"].extend(values[masking.masked_indices])
            batch_results[policy.name]["preds"].extend(pred[masking.masked_indices])
            batch_results[policy.name]["genes"].extend(names[masking.masked_indices])

    # Save results per policy
    for policy, results in batch_results.items():
        out = run_output_dir / policy / f"batch_{start_idx}.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(results, f, indent=2)


def parse_args():
    """CLI arguments for batch processing."""
    p = argparse.ArgumentParser()
    p.add_argument("--start-idx", type=int, required=True)
    p.add_argument("--end-idx", type=int, required=True)
    p.add_argument("--query", type=str, required=True,
                   choices=["blood", "lung", "blood-cancer", "lung-cancer"])
    p.add_argument("--model-name", type=str, required=True,
                   choices=["pretrained_human", "pretrained_pancancer",
                            "human_uniform", "human_cancer",
                            "pancancer_uniform", "pancancer_cancer"])
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    h5ad_path = f"{PROJECT_ROOT}/research/data/dataset/{args.query}/test.h5ad"
    model_dir = f"{PROJECT_ROOT}/research/training/models/{args.model_name}"

    run_one_batch(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        h5ad_path=h5ad_path,
        model_dir=model_dir,
        model_name=args.model_name,
        query=args.query
    )


if __name__ == "__main__":
    main()