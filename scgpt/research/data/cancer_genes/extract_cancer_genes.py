import csv

input_path = "COSMIC_gene_census.csv"
output_path = "cancer_gene_list.txt"

genes = []

with open(input_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        tier = row["Tier"].strip()
        if tier != "1":
            continue
        
        gene = row["Gene Symbol"].strip()
        
        if gene:
            genes.append(gene)

genes = sorted(set(genes))

with open(output_path, "w", encoding="utf-8") as f:
    for gene in genes:
        f.write(gene + "\n")

print(f"Wrote {len(genes)} genes to {output_path}")