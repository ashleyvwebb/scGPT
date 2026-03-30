from pathlib import Path

def get_gene_names(adata):
    if "feature_name" in adata.var.columns:
        return adata.var["feature_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()

def compute_vocab_overlap(gene_names, vocab):
    vocab_genes = set(vocab.get_stoi().keys()) if hasattr(vocab, "get_stoi") else set(vocab.keys())
    gene_set = set(gene_names)
    overlap = gene_set & vocab_genes
    overlap_fraction = len(overlap) / len(vocab_genes) if len(vocab_genes) > 0 else 0.0
    return {
        "n_genes": len(gene_set),
        "n_vocab": len(vocab_genes),
        "n_overlap": len(overlap),
        "overlap_fraction": overlap_fraction,
        "overlap_genes": overlap,
    }

def compute_cancer_gene_overlap(gene_names, cancer_genes):
    gene_set = set(gene_names)
    cancer_gene_set = set(cancer_genes)
    overlap = gene_set & cancer_gene_set
    overlap_fraction = len(overlap) / len(cancer_gene_set) if len(cancer_gene_set) > 0 else 0.0
    return {
        "n_genes": len(gene_set),
        "n_cancer_genes": len(cancer_gene_set),
        "n_overlap": len(overlap),
        "overlap_fraction": overlap_fraction,
        "overlap_genes": overlap,
    }
