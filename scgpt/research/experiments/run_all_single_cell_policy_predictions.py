from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from research.data.cxg_loader import load_query_as_adata
from research.experiments.run_single_cell_policy_predictions import run_one_cell


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad-root", type=Path, required=True)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cancer-gene-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--max-files", type=int, default=1)
    p.add_argument("--subset-n-cells", type=int, default=None)
    p.add_argument("--mask-ratio", type=float, default=0.15)
    p.add_argument("--mask-token-value", type=float, default=-1)
    p.add_argument("--pad-value", type=float, default=0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument(
        "--cell-index",
        type=int,
        default=None,
        help="If provided, run only this cell. Otherwise run all selected cells.",
    )
    p.add_argument(
        "--max-total-cells",
        type=int,
        default=None,
        help="Optional cap on total number of cells to evaluate.",
    )
    return p.parse_args()


def get_n_cells(h5ad_root: Path, query: str, max_files: int, subset_n_cells: int | None) -> int:
    adata, _ = load_query_as_adata(
        h5ad_root=h5ad_root,
        query_name=query,
        max_files=max_files,
        subset_n_cells=subset_n_cells,
        seed=0,
    )
    return int(adata.n_obs)


def main():
    args = parse_args()
    print("HERE")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("NOW HERE")

    # if args.cell_index is not None:
    #     run_one_cell(args.h5ad_root, args.query, args.model_dir, args.cancer_gene_path, args.output_dir, args.cell_index, args.max_files, args.subset_n_cells, args.mask_ratio, args.mask_token_value, args.pad_value, args.device)
    #     return

    # print("At line 61")
    # n_cells = get_n_cells(
    #     h5ad_root=args.h5ad_root,
    #     query=args.query,
    #     max_files=args.max_files,
    #     subset_n_cells=args.subset_n_cells,
    # )
    # print("At line 68")

    # if args.max_total_cells is not None:
    #     n_cells = min(n_cells, args.max_total_cells)
    # print("At line 72")

    # print(f"Running prediction jobs locally for {n_cells} cells")

    # for cell_index in range(n_cells):
    #     print(cell_index)
    #     run_one_cell(args.h5ad_root, args.query, args.model_dir, args.cancer_gene_path, args.output_dir, args.cell_index, args.max_files, args.subset_n_cells, args.mask_ratio, args.mask_token_value, args.pad_value, args.device)


if __name__ == "__main__":
    main()