# This should 
# 1. load a small .h5ad file 
# 2. subset the file 
# 3. return an AnnData object

import scanpy as sc
import numpy as np

def load_h5ad(path: str):
    return sc.read_h5ad(path)

def subset_cells(adata, n_cells: int, seed: int = 0):
    np.random.seed(seed)
    idx = np.random.choice(adata.n_obs, size=min(n_cells, adata.n_obs), replace=False)
    return adata[idx].copy()