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
# 3. locate query's all_counts parquet directory
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

import sys
sys.path.append("../")

from data.cxg_loader import (
    get_query_partition_files,
    load_query_partitions,
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
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    seed_everything(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_counts_path = get_all_counts_path(args.scb_root, args.query)
    if not all_counts_path.exists():
        raise FileNotFoundError(f"Missing processed data: {all_counts_path}")

    parquet_files = sorted(all_counts_path.glob("*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {all_counts_path}")
    
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"All-counts path: {all_counts_path}")
    print(f"Parquet files: {len(parquet_files)}")
    print(f"Device: {args.device}")

    config_path = args.output_dir / "run_config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "query": args.query,
                "scb_root": str(args.scb_root),
                "all_counts_path": str(all_counts_path),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": args.device
            },
            f,
            indent=2
        )
    
if __name__ == "__main__":
    main()