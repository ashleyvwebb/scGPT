from scgpt.research.data.alignment import gene_overlap

def read_gene_list(path):
    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]
    
if __name__ == "__main__":
    cancer_file = "scgpt/research/data/cancer_genes/cancer_gene_list.txt"
    hvg_files = [
        "scgpt/research/data/HVGs/hvg_genes_50.txt",
        "scgpt/research/data/HVGs/hvg_genes_70.txt",
        "scgpt/research/data/HVGs/hvg_genes_100.txt",
    ]

    cancer_genes = read_gene_list(cancer_file)

    results = []

    for hvg_file in hvg_files:
        hvgs = read_gene_list(hvg_file)

        stats = gene_overlap(cancer_genes, "cancer_genes", hvgs, "hvgs")

        results.append((hvg_file, stats))
    
    with open("scgpt/research/data/HVGs/hvg_overlap_summary.csv", "w") as f:
        f.write("file, n_hvgs, n_overlap, n_overlap_fraction\n")

        for fname, s in results:
            f.write(
                f"{fname}, {s['n_hvgs']}, {s['n_overlap']}, {s['overlap_fraction']:.2f}\n"
            )
