import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUM_HVG = 70
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
hvg_genes.to_csv("scgpt/research/data/HVGs/hvg_genes_70.txt", index=False, header=False)

# Plot HVGs
means = adata.var["means"]
variances = adata.var["variances_norm"]

plt.figure(figsize=(6, 5))

plt.scatter(
    means[~hvg_mask],
    variances[~hvg_mask],
    s=5,
    c="lightgrey",
    label="other genes"
)

plt.scatter(
    means[hvg_mask],
    variances[hvg_mask],
    s=8,
    c="red",
    label="highly variable genes"
)

plt.xlabel("mean expressions of genes")
plt.ylabel("variances of genes (normalized)")
plt.legend()
plt.savefig("scgpt/research/data/HVGs/hvg_plot_70.png", dpi=300)
plt.close()