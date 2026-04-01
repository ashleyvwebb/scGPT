from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.data.cxg_loader import load_query_as_adata
from research.data.alignment import get_gene_names
from research.masking.cancer_gene_sets import load_gene_set
from research.masking.policies import (
    UniformMaskingPolicy,
    CancerWeightedMaskingPolicy,
    ValueAwareCancerWeightedMaskingPolicy,
)
from research.experiments.common_prediction import (
    SingleCellPredictionResult,
    apply_mask_to_values,
    build_valid_mask,
)
from research.experiments.plot_prediction_results import plot_single_cell_predictions


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
    p.add_argument("--pad-value", type=float, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_policies(cancer_gene_set: set[str]):
    return [
        UniformMaskingPolicy(),
        CancerWeightedMaskingPolicy(cancer_gene_set, cancer_weight=5.0, non_cancer_weight=1.0),
        ValueAwareCancerWeightedMaskingPolicy(
            cancer_gene_set,
            cancer_weight=5.0,
            non_cancer_weight=1.0,
            value_power=0.5,
        ),
    ]


def load_model(model_dir: str, device):
    from pathlib import Path
    import json
    import torch
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.model import TransformerModel
    from scgpt.utils import load_pretrained

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
        pad_value=0,
        n_input_bins=51,
    )
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    model.to(device)
    model.eval()
    return model, vocab


def run_model_forward_stub(
    model,
    gene_ids: np.ndarray,
    masked_values: np.ndarray,
    device: str,
    pad_token_id: int,
):
    model.eval()

    input_gene_ids = torch.as_tensor(gene_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_values = torch.as_tensor(masked_values, device=device).unsqueeze(0)

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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata, _ = load_query_as_adata(
        h5ad_root=args.h5ad_root,
        query_name=args.query,
        max_files=args.max_files,
        subset_n_cells=args.subset_n_cells,
        seed=0,
    )

    model, vocab = load_model(str(args.model_dir), args.device)
    cancer_gene_set = load_gene_set(args.cancer_gene_path)

    gene_names = get_gene_names(adata)

    # one cell
    cell_values = np.asarray(adata.X[args.cell_index]).reshape(-1)
    print("Cell values:", cell_values)
    print(cell_values.toarray())
    print()

    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in gene_names],
        dtype=int,
    )

    valid_mask = build_valid_mask(cell_values, pad_value=args.pad_value)
    policies = build_policies(cancer_gene_set)

    for policy in policies:
        rng = np.random.default_rng(0)
        masking = policy.sample_mask(
            gene_names=gene_names,
            values=cell_values,
            mask_ratio=args.mask_ratio,
            rng=rng,
            valid_mask=valid_mask,
        )

        masked_values = apply_mask_to_values(
            values=cell_values,
            mask=masking.mask,
            mask_token_value=args.mask_token_value,
        )

        pred_values = run_model_forward_stub(
            model=model,
            gene_ids=gene_ids,
            masked_values=masked_values,
            device=args.device,
            pad_token_id=vocab["<pad>"],
        )

        result = SingleCellPredictionResult(
            gene_names=list(gene_names),
            target_values=np.asarray(cell_values, dtype=float),
            predicted_values=np.asarray(pred_values, dtype=float),
            predicted_bins=None,
            masked_indices=masking.masked_indices,
            policy_name=policy.name,
            cell_id=str(args.cell_index),
        )

        base = args.output_dir / f"cell_{args.cell_index}_{policy.name}"

        plot_single_cell_predictions(
            result=result,
            output_path=base.with_suffix(".png"),
            discrete=False,
        )

        with base.with_suffix(".json").open("w") as f:
            json.dump(
                {
                    "policy": policy.name,
                    "cell_index": args.cell_index,
                    "masked_indices": result.masked_indices.tolist(),
                    "target_values": result.target_values[result.masked_indices].tolist(),
                    "predicted_values": result.predicted_values[result.masked_indices].tolist(),
                },
                f,
                indent=2,
            )

    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()