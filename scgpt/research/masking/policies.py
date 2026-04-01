from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class MaskingResult:
    mask: np.ndarray              # shape: (n_genes,), dtype=bool
    probabilities: np.ndarray     # shape: (n_genes,), dtype=float
    masked_indices: np.ndarray    # shape: (k,), dtype=int


class MaskingPolicy:
    name: str = "base"

    def get_probabilities(
        self,
        gene_names: Sequence[str],
        values: np.ndarray,
        valid_mask: np.ndarray | None,
    ) -> np.ndarray:
        raise NotImplementedError

    def sample_mask(
        self,
        gene_names: Sequence[str],
        values: np.ndarray,
        mask_ratio: float,
        rng: np.random.Generator,
        valid_mask: np.ndarray | None,
        min_masks: int = 1,
    ) -> MaskingResult:
        n = len(gene_names)
        if n == 0:
            raise ValueError("No genes provided to masking policy.")

        probs = self.get_probabilities(gene_names, values, valid_mask=valid_mask).astype(float)

        if valid_mask is None:
            valid_mask = np.ones(n, dtype=bool)
        else:
            valid_mask = valid_mask.astype(bool)

        probs = probs.copy()
        probs[~valid_mask] = 0.0

        total = probs.sum()
        if total <= 0:
            raise ValueError("Masking probabilities sum to zero after applying valid_mask.")

        probs /= total

        k = max(min_masks, int(round(mask_ratio * valid_mask.sum())))
        k = min(k, int(valid_mask.sum()))

        valid_indices = np.where(valid_mask)[0]
        chosen = rng.choice(valid_indices, size=k, replace=False, p=probs[valid_indices] / probs[valid_indices].sum())

        mask = np.zeros(n, dtype=bool)
        mask[chosen] = True

        return MaskingResult(
            mask=mask,
            probabilities=probs,
            masked_indices=np.sort(chosen),
        )


class UniformMaskingPolicy(MaskingPolicy):
    name = "uniform"

    def get_probabilities(
        self,
        gene_names: Sequence[str],
        values: np.ndarray,
        valid_mask: np.ndarray | None,
    ) -> np.ndarray:
        n = len(gene_names)
        probs = np.ones(n, dtype=float)
        if valid_mask is not None:
            probs[~valid_mask.astype(bool)] = 0.0
        return probs


class CancerWeightedMaskingPolicy(MaskingPolicy):
    name = "cancer_weighted"

    def __init__(
        self,
        cancer_gene_set: set[str],
        cancer_weight: float = 5.0,
        non_cancer_weight: float = 1.0,
        uppercase: bool = True,
    ):
        if cancer_weight <= 0 or non_cancer_weight <= 0:
            raise ValueError("Weights must be positive.")
        self.cancer_gene_set = cancer_gene_set
        self.cancer_weight = cancer_weight
        self.non_cancer_weight = non_cancer_weight
        self.uppercase = uppercase

    def _normalise(self, gene: str) -> str:
        return gene.upper() if self.uppercase else gene

    def get_probabilities(
        self,
        gene_names: Sequence[str],
        values: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        probs = np.array(
            [
                self.cancer_weight if self._normalise(g) in self.cancer_gene_set else self.non_cancer_weight
                for g in gene_names
            ],
            dtype=float,
        )
        if valid_mask is not None:
            probs[~valid_mask.astype(bool)] = 0.0
        return probs


class ValueAwareCancerWeightedMaskingPolicy(CancerWeightedMaskingPolicy):
    name = "value_aware_cancer_weighted"

    def __init__(
        self,
        cancer_gene_set: set[str],
        cancer_weight: float = 5.0,
        non_cancer_weight: float = 1.0,
        value_power: float = 0.5,
        uppercase: bool = True,
    ):
        super().__init__(
            cancer_gene_set=cancer_gene_set,
            cancer_weight=cancer_weight,
            non_cancer_weight=non_cancer_weight,
            uppercase=uppercase,
        )
        self.value_power = value_power

    def get_probabilities(
        self,
        gene_names: Sequence[str],
        values: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        base = super().get_probabilities(gene_names, values, valid_mask=valid_mask)
        print("a)")
        print(values)
        values = np.asarray(values, dtype=float)
        print("b)")
        value_scale = np.power(np.clip(values, a_min=0.0, a_max=None) + 1.0, self.value_power)
        print("c)")
        probs = base * value_scale
        print("d)")
        if valid_mask is not None:
            print("e)")
            probs[~valid_mask.astype(bool)] = 0.0
        return probs