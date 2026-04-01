from __future__ import annotations

from pathlib import Path
from typing import Iterable


def load_gene_set(path: str | Path, uppercase: bool = True) -> set[str]:
    path = Path(path)
    genes: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            gene = line.strip()
            if not gene:
                continue
            genes.add(gene.upper() if uppercase else gene)

    return genes


def normalise_gene_name(gene: str, uppercase: bool = True) -> str:
    gene = gene.strip()
    return gene.upper() if uppercase else gene


def build_cancer_gene_indicator(
    gene_names: Iterable[str],
    cancer_gene_set: set[str],
    uppercase: bool = True,
) -> list[bool]:
    indicator = []
    for gene in gene_names:
        g = normalise_gene_name(gene, uppercase=uppercase)
        indicator.append(g in cancer_gene_set)
    return indicator


def get_present_cancer_genes(
    gene_names: Iterable[str],
    cancer_gene_set: set[str],
    uppercase: bool = True,
) -> list[str]:
    present = []
    for gene in gene_names:
        g = normalise_gene_name(gene, uppercase=uppercase)
        if g in cancer_gene_set:
            present.append(g)
    return present