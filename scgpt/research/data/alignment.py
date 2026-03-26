from pathlib import Path

def get_gene_names(adata):
    if "feature_name" in adata.var.columns:
        return adata.var["feature_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()

def compute_vocab_overlap(gene_names, vocab):
    vocab_genes = set(vocab.get_stoi().keys()) if hasattr(vocab, "get_stoi") else set(vocab.keys())
    gene_set = set(gene_names)
    overlap = gene_set & vocab_genes
    return {
        "n_genes": len(gene_set),
        "n_vocab": len(vocab_genes),
        "n_overlap": len(overlap),
        "overlap_fraction": len(overlap) / max(1, len(gene_set)),
        "overlap_genes": overlap,
    }

def compute_cancer_gene_overlap(gene_names, cancer_gene):
    gene_set = set(gene_names)
    cancer_gene_set = set(cancer_gene)
    overlap = gene_set & cancer_gene_set
    return {
        "n_cancer_genes": len(cancer_gene_set),
        "n_present": len(overlap),
        "present_genes": overlap,
    }
