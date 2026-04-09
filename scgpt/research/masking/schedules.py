from __future__ import annotations

from dataclasses import dataclass
from typing import List

from scgpt.research.masking.policies import MaskingPolicy


@dataclass
class ScheduledPolicy:
    policy: MaskingPolicy
    mask_ratio: float


class MaskingSchedule:
    name: str = "base_schedule"

    def get(self, epoch: int) -> ScheduledPolicy:
        raise NotImplementedError


class ConstantSchedule(MaskingSchedule):
    name = "constant"

    def __init__(self, policy: MaskingPolicy, mask_ratio: float):
        self.policy = policy
        self.mask_ratio = mask_ratio

    def get(self, epoch: int) -> ScheduledPolicy:
        return ScheduledPolicy(policy=self.policy, mask_ratio=self.mask_ratio)


class DynamicSchedule(MaskingSchedule):
    """
    Generalised multi-stage masking schedule.

    Example:
        policies      = [uniform, cancer, hvgs]
        mask_ratios   = [0.15, 0.20, 0.10]
        switch_epochs = [5, 10]

    Behaviour:
        epoch < 5   -> stage 0 (uniform)
        5 <= epoch < 10 -> stage 1 (cancer)
        epoch >= 10 -> stage 2 (hvgs)
    """

    name = "dynamic"

    def __init__(
        self,
        policies: List[MaskingPolicy],
        mask_ratios: List[float],
        switch_epochs: List[int],
    ):
        # --- Validation ---
        if len(policies) == 0:
            raise ValueError("policies must not be empty")

        if len(policies) != len(mask_ratios):
            raise ValueError("policies and mask_ratios must have the same length")

        if len(switch_epochs) != len(policies) - 1:
            raise ValueError(
                "switch_epochs must have length len(policies) - 1"
            )

        # Ensure strictly increasing switch epochs
        for i in range(1, len(switch_epochs)):
            if switch_epochs[i] <= switch_epochs[i - 1]:
                raise ValueError("switch_epochs must be strictly increasing")

        # Ensure all switch epochs are >= 1
        for e in switch_epochs:
            if e < 1:
                raise ValueError("switch_epochs must all be >= 1")

        self.policies = policies
        self.mask_ratios = mask_ratios
        self.switch_epochs = switch_epochs

    def get(self, epoch: int) -> ScheduledPolicy:
        """
        Determine which stage the current epoch belongs to.

        We iterate through switch_epochs and return the first stage
        whose boundary has not yet been crossed.
        """

        # Find stage index
        for i, switch_epoch in enumerate(self.switch_epochs):
            if epoch < switch_epoch:
                return ScheduledPolicy(
                    policy=self.policies[i],
                    mask_ratio=self.mask_ratios[i],
                )

        # If past all switch points → final stage
        return ScheduledPolicy(
            policy=self.policies[-1],
            mask_ratio=self.mask_ratios[-1],
        )