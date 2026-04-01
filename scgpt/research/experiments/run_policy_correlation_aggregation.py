from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from research.data.cxg_loader import load_query_as_adata


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


def run_one_cell(args, cell_index: int):
    cmd = [
        sys.executable,
        "-m",
        "research.experiments.run_single_cell_policy_predictions",
        "--h5ad-root",
        str(args.h5ad_root),
        "--query",
        args.query,
        "--model-dir",
        str(args.model_dir),
        "--cancer-gene-path",
        str(args.cancer_gene_path),
        "--output-dir",
        str(args.output_dir / f"cell_{cell_index:06d}"),
        "--cell-index",
        str(cell_index),
        "--max-files",
        str(args.max_files),
        "--mask-ratio",
        str(args.mask_ratio),
        "--mask-token-value",
        str(args.mask_token_value),
        "--pad-value",
        str(args.pad_value),
        "--device",
        args.device,
    ]

    if args.subset_n_cells is not None:
        cmd.extend(["--subset-n-cells", str(args.subset_n_cells)])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.cell_index is not None:
        run_one_cell(args, args.cell_index)
        return

    n_cells = get_n_cells(
        h5ad_root=args.h5ad_root,
        query=args.query,
        max_files=args.max_files,
        subset_n_cells=args.subset_n_cells,
    )

    if args.max_total_cells is not None:
        n_cells = min(n_cells, args.max_total_cells)

    print(f"Running prediction jobs locally for {n_cells} cells")

    for cell_index in range(n_cells):
        run_one_cell(args, cell_index)


if __name__ == "__main__":
    main()