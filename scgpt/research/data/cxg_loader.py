from pathlib import Path
import scanpy as sc
import numpy as np
import anndata as ad

def load_h5ad(path: str | Path) -> ad.AnnData:
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    return adata

def get_query_partition_files(h5ad_root: str | Path, query_name: str, max_files: int | None):
    # looks inside {h5ad_root}/{query_name}/*.h5ad
    # loads one or more partititioons
    # currently each partition has 25000 cells, so can set a max number of files
    query_dir = Path(h5ad_root) / query_name
    files = sorted(query_dir.glob("*.h5ad"))
    if max_files is not None:
        files = files[:max_files]
    return files

def subset_cells(adata: ad.AnnData, n_cells: int, seed: int = 0) -> ad.AnnData:
    if adata.n_obs <= n_cells:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
    return adata[idx].copy()

def load_query_as_adata(h5ad_root: str | Path, query_name: str, max_files: int | None, subset_n_cells: int | None, seed: int = 0) -> tuple[ad.AnnData, list[Path]]:
    files = get_query_partition_files(h5ad_root, query_name, max_files)
    
    if len(files) == 0:
        raise FileNotFoundError(f"No .h5ad files found for query {query_name} in {Path(h5ad_root) / query_name}")
    
    adatas = []
    for i, file_path in enumerate(files):
        adata = load_h5ad(file_path)

        if subset_n_cells is not None:
            adata = subset_cells(adata, subset_n_cells, seed=seed + i)
        
        adata.obs["source_file"] = file_path.name
        adatas.append(adata)

    if len(adatas) == 1:
        merged = adatas[0]
    else:
        merged = ad.concat(
            adatas,
            axis=0,
            join="outer",
            label="partition",
            keys=[p.stem for p in files],
            merge="same",
            index_unique="-",
        )

    merged.var_names_make_unique()
    return merged, files

# TO BE USED IF DATA IS STORED AS .scb FILES
def get_all_counts_path(scb_root: str | Path, query_name: str) -> Path:
    # returns {scb_root}/{query_name}/all_counts
    # these are the processed parquet locations created by build_large_scale_data.py
    return Path(scb_root) / query_name / "all_counts"