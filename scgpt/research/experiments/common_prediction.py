from __future__ import annotations

import numpy as np


def apply_mask_to_values(
    values: np.ndarray,
    mask: np.ndarray,
    mask_token_value: int | float,
) -> np.ndarray:
    masked = values.copy()
    masked[mask] = mask_token_value
    return masked