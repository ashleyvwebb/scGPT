from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import issparse

from scgpt.research.data.cxg_loader import load_query_as_adata
from scgpt.research.data.alignment import get_gene_names
from scgpt.research.masking.cancer_gene_sets import load_gene_set
from scgpt.research.masking.policies import (
    UniformMaskingPolicy,
    CancerWeightedMaskingPolicy,
    ValueAwareCancerWeightedMaskingPolicy,
)
from scgpt.research.experiments.common_prediction import (
    SingleCellPredictionResult,
    apply_mask_to_values,
)
from scgpt.research.experiments.plot_prediction_results import plot_single_cell_predictions

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch


def build_policies(cancer_gene_set: set[str]):
    return [
        UniformMaskingPolicy(),
        CancerWeightedMaskingPolicy(
            cancer_gene_set,
            cancer_weight=5.0,
            non_cancer_weight=1.0,
        ),
        ValueAwareCancerWeightedMaskingPolicy(
            cancer_gene_set,
            cancer_weight=5.0,
            non_cancer_weight=1.0,
            value_power=0.5,
        ),
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
        log1p=False,
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

    print(tokenized["genes"])

    if True:
        return

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


def run_one_cell(
    h5ad_root,
    query,
    model_dir,
    cancer_gene_path,
    output_dir,
    cell_index,
    max_files,
    subset_n_cells,
    mask_ratio,
    mask_token_value,
    pad_value,
    device,
    expr_name
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata, _ = load_query_as_adata(
        h5ad_root=h5ad_root,
        query_name=query,
        max_files=max_files,
        subset_n_cells=subset_n_cells,
        seed=0,
    )

    model, vocab = load_model(str(model_dir), device)
    cancer_gene_set = load_gene_set(cancer_gene_path)

    adata = preprocess_adata(adata, n_bins=51)

    prepared = prepare_single_cell_tokenized_input(
        adata=adata,
        cell_index=cell_index,
        vocab=vocab,
        max_seq_len=1200,
        pad_token="<pad>",
        pad_value=pad_value,
        append_cls=False,
        include_zero_gene=False,
    )

    tokenized_gene_ids = prepared["tokenized_gene_ids"]
    tokenized_values = prepared["tokenized_values"]
    tokenized_gene_names = prepared["tokenized_gene_names"]
    valid_mask = prepared["valid_mask"]
    pad_token_id = prepared["pad_token_id"]

    policies = build_policies(cancer_gene_set)

    for policy in policies:
        rng = np.random.default_rng(0)

        masking = policy.sample_mask(
            gene_names=tokenized_gene_names,
            values=tokenized_values,
            mask_ratio=mask_ratio,
            rng=rng,
            valid_mask=valid_mask,
        )

        masked_values = apply_mask_to_values(
            values=tokenized_values,
            mask=masking.mask,
            mask_token_value=mask_token_value,
        )

        pred_values = run_model_forward(
            model=model,
            gene_ids=tokenized_gene_ids,
            masked_values=masked_values,
            device=device,
            pad_token_id=pad_token_id,
        )

        print("target min/max/mean:", tokenized_values.min(), tokenized_values.max(), tokenized_values.mean())
        print("pred min/max/mean:", pred_values.min(), pred_values.max(), pred_values.mean())
        print("fraction target==0 on masked:", np.mean(tokenized_values[masking.masked_indices] == 0))

        result = SingleCellPredictionResult(
            gene_names=list(tokenized_gene_names),
            target_values=np.asarray(tokenized_values, dtype=float),
            predicted_values=np.asarray(pred_values, dtype=float),
            predicted_bins=None,
            masked_indices=masking.masked_indices,
            policy_name=policy.name,
            cell_id=str(cell_index),
        )

        base = output_dir / expr_name / policy.name /f"cell_{cell_index}"

        plot_single_cell_predictions(
            result=result,
            output_path=base.with_suffix(".png"),
            discrete=False,
        )

        with base.with_suffix(".json").open("w") as f:
            json.dump(
                {
                    "policy": policy.name,
                    "cell_index": cell_index,
                    "masked_indices": result.masked_indices.tolist(),
                    "gene_names": [result.gene_names[i] for i in result.masked_indices],
                    "target_values": result.target_values[result.masked_indices].tolist(),
                    "predicted_values": result.predicted_values[result.masked_indices].tolist(),
                },
                f,
                indent=2,
            )

    print(f"Saved outputs to {output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad-root", type=Path, required=True)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cancer-gene-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cell-index", type=int, default=0)
    p.add_argument("--max-files", type=int, default=1)
    p.add_argument("--subset-n-cells", type=int, default=None)
    p.add_argument("--mask-ratio", type=float, default=0.15)
    p.add_argument("--mask-token-value", type=float, default=-1)
    p.add_argument("--pad-value", type=float, default=-2)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--expr-name",
        type=str,
        default="expr1",
        help="Name of this experiment, used for organizing output subdirectories.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_one_cell(
        h5ad_root=args.h5ad_root,
        query=args.query,
        model_dir=args.model_dir,
        cancer_gene_path=args.cancer_gene_path,
        output_dir=args.output_dir,
        cell_index=args.cell_index,
        max_files=args.max_files,
        subset_n_cells=args.subset_n_cells,
        mask_ratio=args.mask_ratio,
        mask_token_value=args.mask_token_value,
        pad_value=args.pad_value,
        device=args.device,
        expr_name=args.expr_name
    )


if __name__ == "__main__":
    main()