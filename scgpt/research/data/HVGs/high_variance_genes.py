import scanpy as sc
import pandas as pd
import numpy as np

NUM_HVG = 50
DATASET_PATH = "scgpt/research/data/dataset/train.h5ad"

adata = sc.read_h5ad(DATASET_PATH)

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=NUM_HVG,
    batch_key="batch"
)

hvg_mask = adata.var["highly_variable"].values
hvg_genes = adata.var.loc[hvg_mask, "feature_name"].astype(str)

# Save gene names
hvg_genes.to_csv("scgpt/research/data/HVGs/train_hvg_genes.txt", index=False, header=False)

# Plot HVGs
sc.pl.highly_variable_genes(adata, save="scgpt/research/data/HVGs/train_hvg_plot.png")


def load_hvg_mask(adata, hvg_file):
    """
    Reconstruct HVG mask aligned with adata.X columns
    
    Parameters:
        adata: AnnData
        hvg_file: path to saved hvg_genes.txt

    Returns:
        hvg_mask: np.ndarray (bool)
        hvg_indicies: np.ndarray (int)
    """
    hvg_genes = pd.reas_csv(hvg_file, header=None)[0].astype(str).values
    
    gene_names = adata.var["feature_name"].astype(str).values

    hvg_set = set(hvg_genes)

    hvg_mask = np.array([g in hvg_set for g in gene_names])

    hvg_indicies = np.where(hvg_mask)[0]

    return hvg_mask, hvg_indicies