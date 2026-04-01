# load data for one query
# use the processed all_counts data
# do 1 train epoch
# save metrics/checkpoint / print loss 
# no masking changes yet
# this validates .scb -> training -> loss

# GOALS: 
# - processed data can be loaded 
# - scGPT can see it 
# - one training loop works 
# - loss is finite 
# - checkpointing works

# 1. parse config / arguments
# 2. choose one query
# 3. locate query's h5ad files
# 4. load the processed data
# 5. build the baseline scGPT model
# 6. run one short training loop
# 7. print loss each step / epoch
# 8. save a checkpoint
from __future__ import annotations

from pathlib import Path
import argparse
import json
import random

import anndata as ad
import numpy as np
import torch
import scanpy as sc
from torch.utils.data import DataLoader, Dataset

from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch, random_mask_value
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained

import sys
sys.path.append("../")

from data.cxg_loader import (
    get_query_partition_files,
    load_query_as_adata,
    subset_cells,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline scGPT pretraining run")
    parser.add_argument("--h5ad-root", type=Path, required=True)
    parser.add_argument("--query", type=str, default="lung-cancer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--subset-n-cells", type=int, default=2000)
    parser.add_argument("--combined-name", type=str, default="combined_input.h5ad")
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TokenisedCellDataset(Dataset):
    def __init__(self, tokenized):
        self.genes = tokenized["genes"]
        self.values = tokenized["values"]
        self.target_values = tokenized["target_values"]

    def __len__(self):
        return self.genes.shape[0]
    
    def __getitem__(self, idx):
        return {
            "genes": self.genes[idx],
            "values": self.values[idx],
            "target_values": self.target_values[idx]
        }

def collate_batch(batch):
    gene_ids = torch.stack([item["gene_ids"] for item in batch], dim=0)
    values = torch.stack([item["values"] for item in batch], dim=0)
    target_values = torch.stack([item["target_values"] for item in batch], dim=0)
    return {
        "genes": gene_ids,
        "values": values,
        "target_values": target_values
    }

def prepare_tokenized_data(
        adata,
        vocab,
        max_seq_len: int = 1200,
        mask_ration: float = 0.15,
        mask_value: int = -1
):
    adata.layers["counts"] = adata.X.copy()

    prepocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        binning=51,
        result_binned_key="X_binned"
    )

    prepocessor(adata)

    if "feature_name" in adata.var.columns:
        gene_names = adata.var["feature_name"].astype(str).tolist()
    else:
        gene_names = adata.var_names.astype(str).tolist()

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in gene_names], dtype=int)

    data = adata.layers["X_binned"] if "X_binned" in adata.layers else adata.X
    if not isinstance(data, np.ndarray):
        data = data.toarray()

    tokenized = tokenize_and_pad_batch(
        data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=False
    )

    target_values = tokenized["values"].clone()
    input_values = random_mask_value(
        tokenized["values"],
        mask_ratio=mask_ration,
        mask_value=mask_value,
        pad_value=-2
    )

    return {
        "genes": tokenized["genes"],
        "values": input_values,
        "target_values": target_values
    }    

def load_model(model_dir: str, device):
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
    return model

def train_one_epoch(model, loader, optimizer, device, mask_value=-1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        target_values = batch["target_values"].to(device)

        src_key_padding_mask = gene_ids.eq(0)
        masked_positions = values.eq(mask_value)

        output_dict = model(
            src=gene_ids,
            values=values,
            src_key_padding_mask=src_key_padding_mask,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
        )

        pred = output_dict["mlm_output"]

        if masked_positions.sum() == 0:
            continue

        loss = ((pred[masked_positions] - target_values[masked_positions]) ** 2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
    
    return total_loss / max(n_batches, 1)

def main():
    args = parse_args()
    seed_everything(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_files = get_query_partition_files(args.h5ad_root, args.query, args.max_files)
    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No .h5ad files found in {args.h5ad_root}")
    
    merged_adata, used_files = load_query_as_adata(
        h5ad_root=args.h5ad_root,
        query_name=args.query,
        max_files=args.max_files,
        subset_n_cells=args.subset_n_cells,
        seed=args.seed
    )

    combined_h5ad_path = args.output_dir / args.combined_name
    merged_adata.write_h5ad(combined_h5ad_path)

    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"Partition path: {args.h5ad_root / args.query}")
    print(f".h5ad files used: {len(h5ad_files)}")
    print(f"Combined shape: {merged_adata.nobs} cells x {merged_adata.n_vars} genes")
    print(f"Combined h5ad: {combined_h5ad_path}")
    print(f"Device: {args.device}")
    print("=" * 80)

    config_path = args.output_dir / "run_config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "query": args.query,
                "h5ad_root": str(args.h5ad_root),
                "partition_path": str(args.h5ad_root / args.query),
                "used_files": [str(p) for p in used_files],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": args.device,
                "max_files": args.max_files,
                "subset_n_cells": args.subset_n_cells,
                "combined_h5ad": str(combined_h5ad_path)
            },
            f,
            indent=2
        )

    #TODO: later at some point in time, look at example script to figure out order for this
        # ------------------------------------------------------------------
    # Baseline training from merged .h5ad
    # ------------------------------------------------------------------
    # vocab = GeneVocab.from_file("/path/to/default_census_vocab.json")

    # tokenized = prepare_tokenized_data(
    #     merged_adata,
    #     vocab=vocab,
    #     max_seq_len=1200,
    #     mask_ratio=0.15,
    #     mask_value=-1,
    # )

    # dataset = TokenisedCellDataset(tokenized)
    # loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     collate_fn=collate_batch,
    # )

    # model = build_model(
    #     vocab_size=len(vocab),
    #     pad_token_id=vocab["<pad>"],
    #     device=args.device,
    # )

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # losses = []
    # for epoch in range(args.epochs):
    #     epoch_loss = train_one_epoch(
    #         model=model,
    #         loader=loader,
    #         optimizer=optimizer,
    #         device=args.device,
    #         mask_value=-1,
    #     )
    #     losses.append(epoch_loss)
    #     print(f"Epoch {epoch + 1}/{args.epochs} | loss = {epoch_loss:.6f}")

    # ckpt_path = args.output_dir / "checkpoint.pt"
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "losses": losses,
    #         "query": args.query,
    #     },
    #     ckpt_path,
    # )

    # with (args.output_dir / "losses.json").open("w") as f:
    #     json.dump({"losses": losses}, f, indent=2)

    # print(f"Saved checkpoint to {ckpt_path}")
    
if __name__ == "__main__":
    main()