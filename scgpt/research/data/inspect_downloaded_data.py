from __future__ import annotations

import argparse
from pathlib import Path
import json

from scgpt.tokenizer import GeneVocab

from research.data.cxg_loader import (
    list_query_partitions,
    load_h5ad,
    subset_cells
)

from research.data.alignment import (
    get_gene_names,
    compute_vocab_overlap,
    compute_cancer_gene_overlap
)

DEFAULT_QUERIES = ["lung", "blood", "lung-cancer", "blood-cancer"]

def load_cancer_gene_set(path: str | Path) -> set[str]:
    path = Path(path)
    with path.open() as f:
        genes = {line.strip() for line in f if line.strip()}
    return genes

def summarise_adata(adata) -> dict:
    obs_cols = list(adata.obs.columns)
    var_cols = list(adata.var.columns)

    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": obs_cols,
        "var_columns": var_cols,
    }

def inspect_query(
        h5ad_root: Path,
        query_name: str,
        vocab,
        cancer_gene_set: set[str],
        max_files: int,
        subset_n_cells: int | None,
) -> dict:
    files = list_query_partitions(h5ad_root, query_name)

    result = {
        "query": query_name,
        "n_files_found": len(files),
        "files": [],
    }

    if len(files) == 0:
        result["error"] = "No .h5ad files found for this query."
        return result
    
    files = files[:max_files]

    for file_path in files:
        file_info: dict = {"file": str(file_path)}

        try:
            adata = load_h5ad(file_path)

            if subset_n_cells is not None:
                adata = subset_cells(adata, subset_n_cells, seed=0)

            basic = summarise_adata(adata)
            gene_names = get_gene_names(adata)

            vocab_overlap = compute_vocab_overlap(gene_names, vocab)
            cancer_overlap = compute_cancer_gene_overlap(gene_names, cancer_gene_set)

            file_info["summary"] = basic
            file_info["vocab_overlap"] = {
                "n_genes": vocab_overlap["n_genes"],
                "n_vocab": vocab_overlap["n_vocab"],
                "n_overlap": vocab_overlap["n_overlap"],
                "overlap_fraction": vocab_overlap["overlap_fraction"],
            }
            file_info["cancer_gene_overlap"] = {
                "n_cancer_genes": cancer_overlap["n_cancer_genes"],
                "n_present": cancer_overlap["n_present"],
                "present_genes_sample": sorted(list(cancer_overlap["present_genes"]))[:20],
            }

            file_info["obs_head_columns"] = {
                col: adata.obs[col].astype(str).head(5).tolist()
                for col in adata.obs.columns[: min(5, len(adata.obs.columns))]
            }
            file_info["var_head_columns"] = {
                col: adata.var[col].astype(str).head(5).tolist()
                for col in adata.var.columns[: min(5, len(adata.var.columns))]
            }
        except Exception as e:
            file_info["error"] = f"{type(e).__name__}: {e}"
        
        result["files"].append(file_info)
    return result

def print_result(result: dict) -> None:
    print("=" * 80)
    print(f"QUERY: {result["query"]}")
    print(f"FILES FOUND: {result["n_files_found"]}")

    if "error" in result:
        print(f"ERROR: {result["error"]}")
        return
    
    for file_info in result["files"]:
        print("-" * 80)
        print(f"FILE: {file_info["file"]}")

        if "error" in file_info:
            print(f"ERROR: {file_info["error"]}")
            continue

        summary = file_info["summary"]
        vocab_overlap = file_info["vocab_overlap"]
        cancer_overlap = file_info["cancer_gene_overlap"]

        print(f"Shape: ({summary["n_cells"]}, {summary["n_genes"]})")
        print(f"obs columns ({len(summary["obs_columns"])}): {summary["obs_columns"][:10]}")
        print(f"var columns ({len(summary["var_columns"])}): {summary["var_columns"][:10]}")
        print(
            "Vocab overlap: "
            f"{vocab_overlap["n_overlap"]}/{vocab_overlap["n_genes"]} "
            f"({vocab_overlap["overlap_fraction"]:.3f})"
        )
        print(
            "Cancer gene overlap: "
            f"{vocab_overlap["n_overlap"]}/{vocab_overlap["n_genes"]} "
            f"({vocab_overlap["overlap_fraction"]:.3f})"
        )
        print(
            "Cancer genes present sample: "
            f"{cancer_overlap["present_genes_sample"]}"
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect downloaded CellxGene .h5ad partitions for scGPT compatibility."
    )
    parser.add_argument(
        "--h5ad-root",
        type=Path,
        required=True,
        help="Root directory containing query subfolders of downloaded .h5ad files."
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        required=True,
        help="Path to scGPT vocab JSON, e.g. default_census_vocab.json"
    )
    parser.add_argument(
        "--cancer-gene-path",
        type=Path,
        required=True,
        help="Path to cancer gene list text file, one gene per line."
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Queries to inspect. Defaults to lung, blood, lung-cancer, blood-cancer"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="How many .h5ad files to inspect per query."
    )
    parser.add_argument(
        "--subset-n-cells",
        type=int,
        default=None,
        help="Optionally subset each loaded partition to this many cells before inspection."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the inspection results as JSON."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    vocab = GeneVocab.from_file(args.vocab_path)
    cancer_gene_set = load_cancer_gene_set(args.cancer_gene_path)

    all_results = []
    for query_name in args.queries:
        result = inspect_query(
            h5ad_root=args.h5ad_root,
            query_name=query_name,
            vocab=vocab,
            cancer_gene_set=cancer_gene_set,
            max_files=args.max_files,
            subset_n_cells=args.subset_n_cells
        )
        all_results.append(result)
        print_result(result)
    
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(all_results, f, indent=2)
        print("=" * 80)
        print(f"Saved results to {args.output_json}")

if __name__ == "__main__":
    main()