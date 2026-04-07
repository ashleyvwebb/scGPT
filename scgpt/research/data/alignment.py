from pathlib import Path

def get_gene_names(adata):
    if "feature_name" in adata.var.columns:
        return adata.var["feature_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()

def gene_overlap(list1, list1_label, list2, list2_label):
    list1_set = set(list1)
    list2_set = set(list2)
    overlap = list1_set & list2_set
    overlap_fraction = len(overlap) / len(list2_set) if len(list2_set) > 0 else 0.0
    return {
        f"n_{list1_label}": len(list1_set),
        f"n_{list2_label}": len(list2_set),
        "n_overlap": len(overlap),
        "overlap_fraction": overlap_fraction,
        "overlap_genes": overlap,
    }

def compute_vocab_overlap(gene_names, vocab):
    vocab_list = vocab.get_stoi().keys() if hasattr(vocab, "get_stoi") else vocab.keys()
    return gene_overlap(gene_names, "genes", vocab_list, "vocab")

def compute_cancer_gene_overlap(gene_names, cancer_genes):
    return gene_overlap(gene_names, "genes", cancer_genes, "cancer_genes")

