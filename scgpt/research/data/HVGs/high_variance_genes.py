import scanpy as sc

NUM_HVG = 50

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=NUM_HVG,
    batch_key=None
)