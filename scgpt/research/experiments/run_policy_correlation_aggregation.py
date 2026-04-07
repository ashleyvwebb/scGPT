import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def aggregate(policy_dir):
    targets, preds = [], []

    for f in policy_dir.glob("batch_*.json"):
        data = json.load(open(f))
        targets.extend(data["targets"])
        preds.extend(data["preds"])
    
    return np.array(targets), np.array(preds)

def plot(targets, preds, out):
    plt.figure(figsize=(6,6))
    plt.hexbin(targets, preds, gridsize=50, bins='log')
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
        missing = [i for i in range(int(t.min()), int(t.max())+1)
           if i not in t]
        print(missing)
        print(np.unique(t))
    
        plt.figure()
        plt.hist(t, bins=50)
        plt.title("Target distribution")
        plt.savefig(policy / "target_hist.png")
        plt.close()

        plt.figure()
        plt.hist2d(t, p, bins=50, cmap='coolwarm')
        plt.colorbar()
        plt.savefig(policy / "hist2d_attemp.png")
        plt.close()

if __name__ == "__main__":
    main("scgpt/research/results/batched_predictions/uniform_model_test_no_zero_70")