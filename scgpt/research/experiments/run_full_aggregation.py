import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr, pearsonr

BASE_DIR = Path("scgpt/research/results/cancer_predictions/")

def split_by_threshold(t, p, threshold=35):
    mask_left = t <= threshold
    mask_right = t > threshold

    return (t[mask_left], p[mask_left]), (t[mask_right], p[mask_right])

def aggregate_policy(policy_dir):
    targets, preds = [], []

    for f in policy_dir.glob("batch_*.json"):
        data = json.load(open(f))
        targets.extend(data["targets"])
        preds.extend(data["preds"])
    
    return np.array(targets), np.array(preds)

def plot(targets, preds, out, xlim=None, ylim=None):
    pearson_corr, _ = pearsonr(targets, preds)
    spearman_corr, _ = spearmanr(targets, preds)

    bins = len(np.unique(targets))

    plt.figure()
    plt.hist2d(targets, preds, bins=bins, norm=LogNorm())
    plt.colorbar()

    max_val = max(targets.max(), preds.max())

    if xlim:
        plt.xlim(*xlim)
    else:
        plt.xlim(0, max_val)

    if ylim:
        plt.ylim(*ylim)
    else:
        plt.ylim(0, max_val)

    x0, x1 = plt.xlim()
    plt.plot([x0, x1], [x0, x1], 'r--')

    plt.xlabel("Target")
    plt.ylabel("Predicted")

    plt.text(
        x0 + 0.05 * (x1 - x0),
        ylim[1] - 0.05 * (ylim[1] - ylim[0]) if ylim else x1 * 0.95,
        f"Pearson: {pearson_corr:.3f}\nSpearman: {spearman_corr:.3f}",
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.savefig(out, dpi=300)
    plt.close()
    return pearson_corr, spearman_corr

def plot_hist(t, policy):
    plt.figure()
    plt.hist(t, bins=50)
    plt.title("Target distribution")
    plt.savefig(policy / "target_hist.png")
    plt.close()

def run():
    summary = []

    for model_dir in BASE_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for query_dir in model_dir.iterdir():
            if not query_dir.is_dir():
                continue

            for policy_dir in query_dir.iterdir():
                if not policy_dir.is_dir():
                    continue

                print(f"Aggregating {model_dir.name} / {query_dir.name} / {policy_dir.name}")

                t, p = aggregate_policy(policy_dir)
                # CHANGE MADE IN ORDER TO SPLIT THE DATA BY THRESHOLD
                (left_t, left_p), (right_t, right_p) = split_by_threshold(t, p, threshold=35)

                if len(t) == 0:
                    print("Skipping empty:", policy_dir)
                    continue

                pearson_corr, spearman_corr =  plot(t, p, policy_dir / "aggregated.png")
                plot_hist(t, policy_dir)

                if len(left_t) > 0:
                    plot(left_t, left_p, policy_dir / "aggregated_low.png", xlim=(0, 36), ylim=(15,50))

                if len(right_t) > 0:
                    plot(right_t, right_p, policy_dir / "aggregated_high.png", xlim=(35, 51), ylim=(15, 55))

                summary.append({
                    "model": model_dir.name,
                    "query": query_dir.name,
                    "policy": policy_dir.name,
                    "n": len(t),
                    "pearson": float(pearson_corr),
                    "spearman": float(spearman_corr)
                })
    
    out_file = BASE_DIR / "summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary -> {out_file}")

if __name__ == "__main__":
    run()