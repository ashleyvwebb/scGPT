from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch


@dataclass
class SingleCellPredictionResult:
    gene_names: list[str]
    target_values: np.ndarray          # shape: (G,)
    predicted_values: np.ndarray       # shape: (G,)
    predicted_bins: np.ndarray | None  # shape: (G,)
    masked_indices: np.ndarray         # indices predicted under this policy
    policy_name: str
    cell_id: str | None = None


def masked_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def decode_binned_prediction(
    logits: np.ndarray,
    bin_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    logits: (G, B)
    bin_centers: (B,)
    returns:
      predicted_bin: (G,)
      expected_value: (G,)
    """
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    pred_bin = probs.argmax(axis=1)
    expected_value = probs @ bin_centers
    return pred_bin, expected_value


def apply_mask_to_values(
    values: np.ndarray,
    mask: np.ndarray,
    mask_token_value: int | float,
) -> np.ndarray:
    masked = np.array(values, copy=True)
    masked[mask] = mask_token_value
    return masked


def build_valid_mask(values: np.ndarray, pad_value: int | float | None = None) -> np.ndarray:
    values = values[0].toarray()
    print(type(values))
    if pad_value is None:
        return np.ones_like(values, dtype=bool)
    print(values.shape)
    print(values)
    return values != pad_value