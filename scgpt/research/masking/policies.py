from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class MaskingResult:
    mask: np.ndarray              # shape: (n_genes,), dtype=bool
    masked_indices: np.ndarray    # shape: (k,), dtype=int


class MaskingPolicy:
    name: str = "base"

    def build_probability_matrix(
            self,
            gene_names,
            values,
            valid_mask,
            mask_ratio
    ) -> np.ndarray:
        raise NotImplementedError

    def sample_mask(
        self,
        gene_names,
        values: np.ndarray,
        mask_ratio: float,
        valid_mask: np.ndarray | None,
    ) -> MaskingResult:

        probs = self.build_probability_matrix(gene_names, values, valid_mask, mask_ratio)

        probs_t = torch.tensor(probs, dtype=torch.float32)

        mask = torch.bernoulli(probs_t).bool().cpu().numpy()

        return MaskingResult(
            mask=mask,
            masked_indices=np.where(mask)[0],
        )
    

class UniformMaskingPolicy(MaskingPolicy):
    name = "uniform"

    def build_probability_matrix(
            self,
            gene_names,
            values,
            valid_mask,
            mask_ratio
    ) -> np.ndarray:
        probs = np.zeros_like(values, dtype=float)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return probs
        
        probs[valid_indices] = mask_ratio

        return np.clip(probs, 0, 1)

    
class CancerWeightedMaskingPolicy(MaskingPolicy):
    name = "cancer_weighted"

    def __init__(
        self,
        cancer_gene_set: set[str],
        cancer_weight: float = 5.0,
        non_cancer_weight: float = 1.0,
    ):
        if cancer_weight <= 0 or non_cancer_weight <= 0:
            raise ValueError("Weights must be positive.")
        self.cancer_gene_set = cancer_gene_set
        self.cancer_weight = cancer_weight
        self.non_cancer_weight = non_cancer_weight

    def build_probability_matrix(
            self,
            gene_names,
            values,
            valid_mask,
            mask_ratio
    ) -> np.ndarray:
        probs = np.zeros_like(values, dtype=float)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return probs
        
        weights = np.zeros_like(values, dtype=float)

        for i in valid_indices:
            gene = gene_names[i]

            if gene in self.cancer_gene_set:
                weights[i] = self.cancer_weight
            else:
                weights[i] = self.non_cancer_weight
        
        weights_sum = weights.sum()

        if weights_sum == 0:
            return probs
        
        probs = weights / weights_sum

        probs = probs * (mask_ratio * len(valid_indices))

        return np.clip(probs, 0, 1)

class HVGMaskingPolicy(MaskingPolicy):
    name = "hvg"

    def __init__(self, hvg_gene_set: set[str]):
        self.hvg_gene_set = hvg_gene_set

    def build_probability_matrix(
            self, 
            gene_names, 
            values, 
            valid_mask, 
            mask_ratio
    ) -> np.ndarray:
        probs = np.zeros_like(values, dtype=float)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return probs
        
        hvg_indices = [
            i for i in valid_indices
            if gene_names[i] in self.hvg_gene_set
        ]

        if len(hvg_indices) == 0:
            return probs
        
        # This could be changed to mask_ratio, but for the case of our experiments we want to mask all the HVG genes not only a fraction of them
        probs[hvg_indices] = 1

        return np.clip(probs, 0, 1)