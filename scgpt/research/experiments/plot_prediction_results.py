from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scgpt.research.experiments.common_prediction import SingleCellPredictionResult, masked_pearsonr


def plot_single_cell_predictions(
    result: SingleCellPredictionResult,
    output_path: str | Path,
    discrete: bool = False,
    title_prefix: str = "",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = result.target_values[result.masked_indices]
    y = result.predicted_values[result.masked_indices]
    r = masked_pearsonr(x, y)

    plt.figure(figsize=(6, 6))

    if discrete:
        jitter = 0.12
        x_j = x + np.random.uniform(-jitter, jitter, size=len(x))
        y_j = y + np.random.uniform(-jitter, jitter, size=len(y))
        plt.scatter(x_j, y_j, alpha=0.4, s=12)
    else:
        plt.scatter(x, y, alpha=0.4, s=12)

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    title = f"{title_prefix}{result.policy_name}"
    if result.cell_id is not None:
        title += f" | cell={result.cell_id}"
    title += f" | Pearson r={r:.3f}"

    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_cell_density(
    result: SingleCellPredictionResult,
    output_path: str | Path,
    title_prefix: str = "",
    gridsize: int = 40,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = result.target_values[result.masked_indices]
    y = result.predicted_values[result.masked_indices]
    r = masked_pearsonr(x, y)

    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(x, y, gridsize=gridsize, bins="log", mincnt=1)
    plt.colorbar(hb, label="log10(count)")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    title = f"{title_prefix}{result.policy_name}"
    if result.cell_id is not None:
        title += f" | cell={result.cell_id}"
    title += f" | Pearson r={r:.3f}"

    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()