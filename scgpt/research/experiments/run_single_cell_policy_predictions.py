from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import issparse

from scgpt.research.data.cxg_loader import load_h5ad
from scgpt.research.data.alignment import get_gene_names
from scgpt.research.masking.cancer_gene_sets import load_gene_set
from scgpt.research.masking.policies import (
    UniformMaskingPolicy,
    CancerWeightedMaskingPolicy,
    HVGMaskingPolicy,
)
from scgpt.research.experiments.common_prediction import apply_mask_to_values

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch

# =======
# CONFIG
# =======
PROJECT_ROOT = "/springbrook/share/bioinf/csuxfw/scGPT/scgpt"

H5AD_PATH=f"{PROJECT_ROOT}/research/data/dataset/train.h5ad"
MODEL_DIR=f"{PROJECT_ROOT}/research/training/models/hvg"
CANCER_GENE_PATH=f"{PROJECT_ROOT}/research/data/cancer_genes/cancer_gene_list.txt"
HVG_GENE_PATH=f"{PROJECT_ROOT}/research/data/HVGs/hvg_genes.txt"
OUTPUT_DIR=f"{PROJECT_ROOT}/research/results/batched_predictions/"

MASK_RATIO = 0.15
MASK_TOKEN_VALUE = -1
PAD_VALUE = -2

DEVICE = "cpu"
EXPR_NAME = "hvg_model_no_zero"

def load_hvg_genes(path):
    with open(path, "r") as f:
        return set(line.strip() for line in f)


def build_policies(cancer_gene_set: set[str], hvg_gene_set: set[str]):
    return [
        UniformMaskingPolicy(),
        CancerWeightedMaskingPolicy(
            cancer_gene_set,
            cancer_weight=5.0,
            non_cancer_weight=1.0,
        ),
        HVGMaskingPolicy(hvg_gene_set)
    ]


def load_model(model_dir: str, device: str):
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=-2,
        n_input_bins=51,
    )
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    model.to(device)
    model.eval()
    return model, vocab


def preprocess_adata(
    adata,
    n_bins: int = 51,
):
    """
    Minimal preprocessing to produce X_binned in the same style as scGPT examples.
    """
    adata = adata.copy()

    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = get_gene_names(adata)

    # Use raw X as input, no HVG selection here.
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
    Preprocess and tokenize one cell so the model sees shape (1, L), not raw (G,).
    """
    if "X_binned" not in adata.layers:
        raise ValueError("adata.layers['X_binned'] not found. Run preprocessing first.")

    if "gene_name" not in adata.var.columns:
        raise ValueError("adata.var['gene_name'] not found.")

    all_counts = (
        adata.layers["X_binned"].toarray()
        if issparse(adata.layers["X_binned"])
        else adata.layers["X_binned"]
    )
    genes = adata.var["gene_name"].tolist()

    # Keep one cell but preserve 2D shape: (1, G)
    one_cell = np.asarray(all_counts[cell_index : cell_index + 1])

    # Keep only genes that exist in vocab
    in_vocab = np.array([g in vocab for g in genes], dtype=bool)
    one_cell = one_cell[:, in_vocab]
    genes = [g for g, keep in zip(genes, in_vocab) if keep]

    gene_ids = np.array(vocab(genes), dtype=int)

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

    # tokenized["genes"] and tokenized["values"] are shape (1, L)
    tokenized_gene_ids = tokenized["genes"][0].detach().cpu().numpy()
    tokenized_values = tokenized["values"][0].detach().cpu().numpy()
    
    stoi = vocab.get_stoi()
    pad_token_id = stoi[pad_token]

    id_to_token = {v: k for k, v in stoi.items()}
    tokenized_gene_names = [id_to_token.get(int(gid), "<unk>") for gid in tokenized_gene_ids]

    # Valid positions are real genes only; exclude <pad>
    valid_mask = tokenized_values != pad_value

    return {
        "tokenized_gene_ids": tokenized_gene_ids,
        "tokenized_values": tokenized_values,
        "tokenized_gene_names": tokenized_gene_names,
        "valid_mask": valid_mask,
        "pad_token_id": pad_token_id,
        "gene_ids_unpadded": gene_ids,
        "gene_names_unpadded": genes,
    }


def run_model_forward(
    model,
    gene_ids: np.ndarray,
    masked_values: np.ndarray,
    device: str,
    pad_token_id: int,
):
    """
    Run a single-cell forward pass. Inputs must already be tokenized/padded and 1D.
    """
    model.eval()

    input_gene_ids = torch.as_tensor(
        gene_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # (1, L)

    input_values = torch.as_tensor(
        masked_values,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # (1, L)

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

    pred = pred.squeeze(0)
    if pred.ndim == 2 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    return pred.detach().cpu().numpy()


def run_one_batch(
    start_idx,
    end_idx
):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_h5ad(H5AD_PATH)

    model, vocab = load_model(MODEL_DIR, DEVICE)
    cancer_gene_set = load_gene_set(CANCER_GENE_PATH)
    hvg_gene_set = load_hvg_genes(HVG_GENE_PATH)

    policies = build_policies(cancer_gene_set, hvg_gene_set)

    adata = preprocess_adata(adata)

    batch_results = {p.name: {"targets": [], "preds": []}
                     for p in policies}

    for cell_index in range(start_idx, min(end_idx, adata.n_obs)):

        prepared = prepare_single_cell_tokenized_input(
            adata=adata,
            cell_index=cell_index,
            vocab=vocab,
            include_zero_gene=False,
        )

        gene_ids = prepared["tokenized_gene_ids"]
        values = prepared["tokenized_values"]
        names = prepared["tokenized_gene_names"]
        valid_mask = prepared["valid_mask"]
        pad_token_id = prepared["pad_token_id"]

        for policy in policies:

            masking = policy.sample_mask(
                gene_names=names,
                values=values,
                mask_ratio=MASK_RATIO,
                valid_mask=valid_mask,
            )

            masked_values = apply_mask_to_values(
                values=values,
                mask=masking.mask,
                mask_token_value=MASK_TOKEN_VALUE,
            )

            pred = run_model_forward(
                model=model,
                gene_ids=gene_ids,
                masked_values=masked_values,
                device=DEVICE,
                pad_token_id=pad_token_id,
            )

            batch_results[policy.name]["targets"].extend(
                values[masking.masked_indices].tolist()
            )
            batch_results[policy.name]["preds"].extend(
                pred[masking.masked_indices].tolist()
            )

    for policy, results in batch_results.items():
        out = output_dir / EXPR_NAME / policy / f"batch_{start_idx}.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(results, f)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-idx", type=int, required=True)
    p.add_argument("--end-idx", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    run_one_batch(
        start_idx = args.start_idx,
        end_idx = args.end_idx
    )


if __name__ == "__main__":
    main()