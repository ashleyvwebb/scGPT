from __future__ import annotations

from dataclasses import dataclass

from research.masking.policies import MaskingPolicy


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


class TwoStageSchedule(MaskingSchedule):
    name = "two_stage"

    def __init__(
        self,
        stage1_policy: MaskingPolicy,
        stage2_policy: MaskingPolicy,
        stage1_mask_ratio: float,
        stage2_mask_ratio: float,
        switch_epoch: int,
    ):
        if switch_epoch < 1:
            raise ValueError("switch_epoch must be >= 1")
        self.stage1_policy = stage1_policy
        self.stage2_policy = stage2_policy
        self.stage1_mask_ratio = stage1_mask_ratio
        self.stage2_mask_ratio = stage2_mask_ratio
        self.switch_epoch = switch_epoch

    def get(self, epoch: int) -> ScheduledPolicy:
        if epoch < self.switch_epoch:
            return ScheduledPolicy(
                policy=self.stage1_policy,
                mask_ratio=self.stage1_mask_ratio,
            )
        return ScheduledPolicy(
            policy=self.stage2_policy,
            mask_ratio=self.stage2_mask_ratio,
        )