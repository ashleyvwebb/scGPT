import numpy as np
import anndata as ad
from pathlib import Path

from scgpt.research.data.cxg_loader import load_query_as_adata

# -----------------------------
# CONFIG
# -----------------------------
H5AD_ROOT = "scgpt/research/data/cellxgene/h5ad"
OUTPUT_DIR = Path("scgpt/research/data/dataset")

QUERIES = ["blood", "lung", "blood-cancer", "lung-cancer"]

TRAIN_PER_QUERY = 5000
TEST_PER_QUERY = 1000

SEED = 0


def split_dataset(adata, train_n, test_n, seed):
    rng = np.random.default_rng(seed)

    idx = rng.permutation(adata.n_obs)

    train_idx = idx[:train_n]
    test_idx = idx[train_n:train_n + test_n]

    return adata[train_idx].copy(), adata[test_idx].copy()


def process_query(query):
    print(f"Processing {query}")

    # load up to 2 partitions (≈50k cells max)
    adata, files = load_query_as_adata(
        h5ad_root=H5AD_ROOT,
        query_name=query,
        max_files=1,
        subset_n_cells=None,
        seed=SEED,
    )

    print(f"Loaded {adata.n_obs} cells")

    train, test = split_dataset(
        adata,
        TRAIN_PER_QUERY,
        TEST_PER_QUERY,
        SEED,
    )

    query_dir = OUTPUT_DIR / query
    query_dir.mkdir(parents=True, exist_ok=True)

    train.write_h5ad(query_dir / "train.h5ad")
    test.write_h5ad(query_dir / "test.h5ad")

    print(f"{query}: train={train.n_obs}, test={test.n_obs}")

    return test, train


def main():
    test_adatas = []
    train_adatas = []

    for q in QUERIES:
        test, train = process_query(q)
        
        test_adatas.append(test)
        train_adatas.append(train)

    test_data = ad.concat(test_adatas, merge='same')
    train_data = ad.concat(train_adatas, merge='same')

    test_data.write_h5ad(OUTPUT_DIR / "test.h5ad")
    train_data.write_h5ad(OUTPUT_DIR / "train.h5ad")

    print("Done")


if __name__ == "__main__":
    main()