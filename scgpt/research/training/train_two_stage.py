import numpy as np
import torch
import anndata as ad
import json
from torch.optim import AdamW
from pathlib import Path

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
)


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS_STAGE1 = 3
EPOCHS_STAGE2 = 3
LR = 1e-4

MASK_RATIO = 0.3
MASK_VALUE = -1
PAD_VALUE = -2

MAX_SEQ_LEN = 1200

# paths
MODEL_DIR = Path("scgpt/research/pretrained_models/whole_human")
DATASET_ROOT = Path("scgpt/research/data/dataset")
OUTPUT_DIR = Path("scgpt/research/training/models")


# -----------------------------
# SAVE UTILS
# -----------------------------
def save_checkpoint(model, vocab, stage_name):
    save_dir = OUTPUT_DIR / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # model weights
    torch.save(model.state_dict(), save_dir / "model.pt")

    # vocab
    vocab.save_json(save_dir / "vocab.json")

    # config (CRITICAL)
    config = {
        "pad_value": PAD_VALUE,
        "mask_value": MASK_VALUE,
        "mask_ratio": MASK_RATIO,
        "max_seq_len": MAX_SEQ_LEN,
        "input_style": "binned",
        "normalize_total": 1e4,
        "log1p": True,
        "binning": 51,
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
def load_processed_dataset(root):

    path = root / "train.h5ad"
    print(f"Loading {path}")
    
    return ad.read_h5ad(path)


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
    gene_ids,
    values,
    gene_names,
    policy,
    pad_token_id,
    epochs,
    optimizer,
    stage_name,
):
    model.train()
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

        print(f"[{stage_name}] Epoch {epoch}: loss = {total_loss:.4f}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("LOADING MODEL")
    model, vocab = load_model(MODEL_DIR)
    optimizer = AdamW(model.parameters(), lr=LR)

    # load dataset
    print("LOADING DATASET")
    adata = load_processed_dataset(DATASET_ROOT)
    print("PREPROCESSING DATASET")
    adata = preprocess(adata)

    print("TOKENIZING DATASET")
    gene_ids, values, gene_names = tokenize_dataset(adata, vocab)
    pad_token_id = vocab["<pad>"]

    # -----------------------------
    # STAGE 1: UNIFORM
    # -----------------------------
    uniform_policy = UniformMaskingPolicy()

    print("STARTING TRAINING -- UNIFORM")
    train(
        model,
        gene_ids,
        values,
        gene_names,
        uniform_policy,
        pad_token_id,
        EPOCHS_STAGE1,
        optimizer,
        "Uniform",
    )

    print("SAVING TRAINING -- UNIFORM")
    save_checkpoint(model, vocab, "stage1_uniform")

    # -----------------------------
    # STAGE 2: CANCER
    # -----------------------------
    cancer_genes = load_gene_set("scgpt/research/data/cancer_genes/cancer_gene_list.txt")

    cancer_policy = CancerWeightedMaskingPolicy(
        cancer_genes,
        cancer_weight=5.0,
    )

    print("STARTING TRAINING -- CANCER WEIGHTED")
    train(
        model,
        gene_ids,
        values,
        gene_names,
        cancer_policy,
        pad_token_id,
        EPOCHS_STAGE2,
        optimizer,
        "Cancer",
    )

    print("SAVING TRAINING -- CANCER WEIGHTED")
    save_checkpoint(model, vocab, "stage2_cancer")


if __name__ == "__main__":
    main()