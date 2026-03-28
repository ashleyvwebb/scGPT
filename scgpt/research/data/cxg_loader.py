from pathlib import Path
import scanpy as sc
import numpy as np
import anndata as ad

def load_h5ad(path: str | Path) -> ad.AnnData:
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    return adata

def list_query_partitions(h5ad_root: str | Path, query_name: str):
    query_dir = Path(h5ad_root) / query_name
    return sorted(query_dir.glob("*.h5ad"))

def load_query_partitions(h5ad_root: str | Path, query_name: str, max_files=int | None):
    # looks inside {h5ad_root}/{query_name}/*.h5ad
    # loads one or more partititioons
    # currently each partition has 25000 cells, so can set a max number of files
    files = list_query_partitions(h5ad_root, query_name)
    if max_files is not None:
        files = files[:max_files]
    return files

def subset_cells(adata: ad.AnnData, n_cells: int, seed: int = 0) -> ad.AnnData:
    if adata.n_obs <= n_cells:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
    return adata[idx].copy()

def get_all_counts_path(scb_root: str | Path, query_name: str) -> Path:
    # returns {scb_root}/{query_name}/all_counts
    # these are the processed parquet locations created by build_large_scale_data.py
    return Path(scb_root) / query_name / "all_counts"