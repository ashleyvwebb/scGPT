import scanpy as sc
import numpy as np
import pandas as pd

NUM_HVG = 50
DATASET_PATH = "scgpt/research/data/dataset/test.h5ad"

adata = sc.read_h5ad(DATASET_PATH)

if "gene_name" not in adata.var.columns:
    raise ValueError("Expected 'gene_name' column in adata.var")

adata.var_names = adata.var["gene_name"].astype(str)

adata.var_names_make_unique()

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=NUM_HVG,
    batch_key="batch"
)

hvg_mask = adata.var["highly_variable"]
hvg_genes = adata.var_names[hvg_mask]
hvg_indicies = np.where(hvg_mask)[0]

print(f"Number of HVGs: {len(hvg_genes)}")

#TODO: finish this - need to complete the file path, as these may not be saved in the correct place
# Save gene names
pd.Series(hvg_genes).to_csv("scgpt/research/data/HVGs/hvg_genes.txt", index=False, header=False)

# Save indicies
np.save("scgpt/research/data/HVGs/hvg_indicies.npy", hvg_indicies)

# Save annotated AnnData
adata.write_h5ad(DATASET_PATH)

# Plot HVGs
# TODO - look into saving this
sc.pl.highly_variable_genes(adata, save="scgpt/research/data/HVGs/hvg_plot.png")
