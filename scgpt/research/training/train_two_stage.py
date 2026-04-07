import numpy as np
import torch
import anndata as ad
import json
from torch.optim import AdamW
from pathlib import Path
import argparse

from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.loss import masked_mse_loss

from scgpt.research.data.alignment import get_gene_names
from scgpt.research.masking.cancer_gene_sets import load_gene_set

from scgpt.preprocess import Preprocessor
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained

from scgpt.research.masking.policies import (
    UniformMaskingPolicy,
    CancerWeightedMaskingPolicy,
    HVGMaskingPolicy
)


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS_STAGE1 = 6
EPOCHS_STAGE2 = 6
LR = 1e-4

MASK_RATIO = 0.4
MASK_VALUE = -1
PAD_VALUE = -2

MAX_SEQ_LEN = 1200

# paths
OUTPUT_DIR = Path("scgpt/research/training/models")


# -----------------------------
# SAVE UTILS
# -----------------------------
def save_checkpoint(model, vocab, stage_name):
    save_dir = OUTPUT_DIR / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # model weights
    torch.save(model.state_dict(), save_dir / "best_model.pt")

    # vocab
    vocab.save_json(save_dir / "vocab.json")

    # config (CRITICAL)
    config = {
        "embsize":512,
        "nheads":8,
        "d_hid":512,
        "nlayers":12,
    }

    with open(save_dir / "args.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved checkpoint: {save_dir}")


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model(model_dir):
    vocab = GeneVocab.from_file(model_dir / "vocab.json")

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=512,
        nhead=8,
        d_hid=512,
        nlayers=12,
        vocab=vocab,
        pad_value=PAD_VALUE,
        n_input_bins=51,
    )

    load_pretrained(model, torch.load(model_dir / "best_model.pt", map_location=DEVICE))
    model.to(DEVICE)

    return model, vocab


# -----------------------------
# LOAD PROCESSED DATASET
# -----------------------------
def load_and_concat_dataset(paths):
    adatas = []
    for p in paths:
        print(f"Loading dataset: {p}")
        adatas.append(ad.read_h5ad(p))

    print("Concatenating datasets...")
    return ad.concat(adatas, join="outer", merge="same")


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(adata):
    if "gene_name" not in adata.var:
        adata.var["gene_name"] = get_gene_names(adata)

    preprocessor = Preprocessor(
        use_key="X",
        normalize_total=1e4,
        log1p=True,
        binning=51,
    )

    preprocessor(adata)
    return adata


# -----------------------------
# TOKENISE
# -----------------------------
def tokenize_dataset(adata, vocab):
    X = adata.layers["X_binned"]
    if hasattr(X, "toarray"):
        X = X.toarray()

    genes = adata.var["gene_name"].tolist()

    in_vocab = np.array([g in vocab for g in genes])
    X = X[:, in_vocab]
    genes = [g for g, keep in zip(genes, in_vocab) if keep]

    gene_ids = np.array(vocab(genes))

    tokenized = tokenize_and_pad_batch(
        X,
        gene_ids,
        max_len=MAX_SEQ_LEN,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=PAD_VALUE,
        append_cls=False,
        include_zero_gene=False,
    )

    return (
        tokenized["genes"].cpu().numpy(),
        tokenized["values"].cpu().numpy(),
        genes,
    )


# -----------------------------
# TRAIN LOOP
# -----------------------------
def train(
    model,
    vocab,
    gene_ids,
    values,
    gene_names,
    policy,
    pad_token_id,
    epochs,
    optimizer,
    stage_name,
    model_prefix,
    val_data=None
):
    model.train()
    best_val = float("inf")
    N = gene_ids.shape[0]

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        total_loss = 0

        print("STARTING EPOCH", epoch)

        for i in range(0, N, BATCH_SIZE):
            print("STARTING BATCH", i)
            idx = perm[i:i+BATCH_SIZE]

            g = torch.tensor(gene_ids[idx]).to(DEVICE)
            v = torch.tensor(values[idx]).float().to(DEVICE)

            masked_batch = []
            masks = []

            for j in range(len(idx)):
                valid_mask = values[idx[j]] != PAD_VALUE

                masking = policy.sample_mask(
                    gene_names,
                    values[idx[j]],
                    mask_ratio=MASK_RATIO,
                    valid_mask=valid_mask,
                )

                m = values[idx[j]].copy()
                m[masking.mask] = MASK_VALUE

                masked_batch.append(m)
                masks.append(masking.mask)

            masked_v = torch.tensor(np.stack(masked_batch)).to(DEVICE)
            mask = torch.tensor(np.stack(masks)).to(DEVICE)

            src_key_padding_mask = g.eq(pad_token_id)

            out = model(
                g,
                masked_v,
                src_key_padding_mask=src_key_padding_mask,
                MVC=False,
                ECS=False,
            )

            pred = out["mlm_output"]

            loss = masked_mse_loss(pred, v, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{stage_name}] Epoch {epoch}: train_loss = {total_loss:.4f}")

        if val_data is not None:
            val_loss = evaluate(
                model,
                val_data[0],
                val_data[1],
                gene_names,
                policy,
                pad_token_id
            )
            print(f"[{stage_name}] Epoch {epoch}: val_loss = {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                print(f"[{stage_name}] New best model (val_loss={val_loss:.4f})")

                save_checkpoint(
                    model,
                    vocab,
                    f"{model_prefix}_{stage_name.lower()}_best"
                )

def evaluate(model, gene_ids, values, gene_names, policy, pad_token_id):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(gene_ids), BATCH_SIZE):
            g = torch.tensor(gene_ids[i:i+BATCH_SIZE]).to(DEVICE)
            v = torch.tensor(values[i:i+BATCH_SIZE]).float().to(DEVICE)

            masked_batch = []
            masks = []

            for j in range(len(g)):
                valid_mask = values[i+j] != PAD_VALUE

                masking = policy.sample_mask(
                    gene_names,
                    values[i+j],
                    mask_ratio=MASK_RATIO,
                    valid_mask=valid_mask,
                )

                m = values[i+j].copy()
                m[masking.mask] = MASK_VALUE

                masked_batch.append(m)
                masks.append(masking.mask)

            masked_v = torch.tensor(np.stack(masked_batch)).to(DEVICE)
            mask = torch.tensor(np.stack(masks)).to(DEVICE)

            src_key_padding_mask = g.eq(pad_token_id)

            out = model(
                g,
                masked_v,
                src_key_padding_mask=src_key_padding_mask,
                MVC=False,
                ECS=False,
            )

            pred = out["mlm_output"]

            loss = masked_mse_loss(pred, v, mask)

            total_loss += loss.item()
            num_batches += 1
    
    if was_training:
        model.train()

    return total_loss / num_batches



# -----------------------------
# MAIN
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, choices=["pretrained_human", "pretrained_pancancer"])
    return p.parse_args()

def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.base_model == "pretrained_human":
        model_dir = Path("scgpt/research/training/models/pretrained_human")

        dataset_paths = [
            Path("scgpt/research/data/dataset/blood/train.h5ad"),
            Path("scgpt/research/data/dataset/lung/train.h5ad"),
            Path("scgpt/research/data/dataset/blood-cancer/train.h5ad"),
            Path("scgpt/research/data/dataset/lung-cancer/train.h5ad")
        ]
    elif args.base_model == "pretrained_pancancer":
        model_dir = Path("scgpt/research/training/models/pretrained_pancancer")

        dataset_paths = [
            Path("scgpt/research/data/dataset/blood-cancer/train.h5ad"),
            Path("scgpt/research/data/dataset/lung-cancer/train.h5ad")
        ]
    else:
        raise ValueError("Invalid base model")
    
    model_prefix = args.base_model.replace("pretrained_", "")
    
    print("\n=== DATASETS USED ===")
    for p in dataset_paths:
        print("-", p)
    print("")

    print("LOADING MODEL")
    model, vocab = load_model(model_dir)
    optimizer = AdamW(model.parameters(), lr=LR)

    # load dataset
    print("LOADING DATASET")
    adata = load_and_concat_dataset(dataset_paths)
    print(f"Total cells after concat: {adata.n_obs}")
    print(f"Total genes: {adata.n_vars}")

    print("PREPROCESSING DATASET")
    adata = preprocess(adata)

    print("TOKENIZING DATASET")
    gene_ids, values, gene_names = tokenize_dataset(adata, vocab)
    pad_token_id = vocab["<pad>"]
    print(f"Tokenized shape: {gene_ids.shape}")

    N = gene_ids.shape[0]
    split = int(0.9 * N)
    perm = np.random.permutation(N)
    train_idx = perm[:split]
    val_idx = perm[split:]

    train_data = (
        gene_ids[train_idx],
        values[train_idx]
    )

    val_data = (
        gene_ids[val_idx],
        values[val_idx]
    )

    # -----------------------------
    # STAGE 1: UNIFORM
    # -----------------------------
    uniform_policy = UniformMaskingPolicy()

    print("TRAINING: UNIFORM")
    train(
        model,
        vocab,
        train_data[0],
        train_data[1],
        gene_names,
        uniform_policy,
        pad_token_id,
        EPOCHS_STAGE1,
        optimizer,
        "Uniform",
        model_prefix,
        val_data=val_data,
    )

    print("SAVING:", f"{model_prefix}_uniform")
    save_checkpoint(model, vocab, f"{model_prefix}_uniform")

    # -----------------------------
    # STAGE 2: CANCER
    # -----------------------------
    optimizer = AdamW(model.parameters(), lr=LR)
    
    cancer_genes = load_gene_set("scgpt/research/data/cancer_genes/cancer_gene_list.txt")

    cancer_policy = CancerWeightedMaskingPolicy(
        cancer_genes,
    )

    print("TRAINING: CANCER")
    train(
        model,
        vocab,
        train_data[0],
        train_data[1],
        gene_names,
        cancer_policy,
        pad_token_id,
        EPOCHS_STAGE2,
        optimizer,
        "Cancer",
        model_prefix,
        val_data=val_data,
    )

    print("SAVING:", f"{model_prefix}_cancer")
    save_checkpoint(model, vocab, f"{model_prefix}_cancer")


if __name__ == "__main__":
    main()