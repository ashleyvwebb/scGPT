import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def aggregate(policy_dir):
    targets, preds = [], []

    for f in policy_dir.glob("batch_*.json"):
        data = json.load(open(f))
        targets.extend(data["targets"])
        preds.extend(data["preds"])
    
    return np.array(targets), np.array(preds)

def plot(targets, preds, out):
    plt.figure(figsize=(6,6))
    plt.hist2d(targets, preds, bins=100, norm=LogNorm())
    plt.colorbar()
    plt.plot([targets.min(), targets.max()],
             [targets.min(), targets.max()], 'r--')
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.savefig(out, dpi=300)
    plt.close()

def main(base):
    base = Path(base)
    for policy in base.iterdir():
        if not policy.is_dir(): continue
        t, p = aggregate(policy)
        plot(t, p, policy / "aggregated.png")
    
        plt.figure()
        plt.hist(t, bins=50)
        plt.title("Target distribution")
        plt.savefig(policy / "target_hist.png")
        plt.close()

if __name__ == "__main__":
    main("scgpt/research/results/batched_predictions/uniform_model_test_no_zero_70")