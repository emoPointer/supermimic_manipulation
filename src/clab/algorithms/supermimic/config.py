"""SuperMimic algorithm defaults with explicit manipulation-task adaptations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SuperMimicConfig:
    """FP32 parameters with explicit MLP autocast; collection budgets are separate.

    The fixed manipulation task has no reference-motion encoder. Its existing
    state-based observation (including history and goal) feeds the decoder.
    """

    wm_hidden: tuple[int, ...] = (2048, 2048, 2048, 2048, 2048)
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    wm_window: int = 8
    horizon: int = 16
    wm_batch_size: int = 2048
    policy_batch_size: int = 4096
    learning_rate: float = 2e-4
    gamma: float = 0.99
    lambda_return: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    max_grad_norm: float = 1.0
    policy_grad_value_clip: float = 0.1
    normalizer_max_count: float = 2e8
    amp: str = "bf16"
    tf32: bool = True
    # 1 keeps the single world model. K>1 trains K independently initialized
    # members; imagination advances each sample with a random member per step
    # and subtracts uncertainty_penalty x member disagreement from its reward.
    wm_ensemble_size: int = 1
    uncertainty_penalty: float = 0.0
    # Fraction of imagination starts drawn from episode-start rows (reset
    # states) rather than uniformly over replay, where each episode contributes
    # one start among hundreds of rows. A reset curriculum places its hardest
    # decisions there (for example whether to keep a grasped object).
    reset_start_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("wm_hidden", "policy_hidden", "critic_hidden"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self.validate()

    def validate(self) -> None:
        if self.amp not in ("none", "bf16"):
            raise ValueError("amp must be 'none' or 'bf16'")
        if type(self.tf32) is not bool:
            raise ValueError("tf32 must be an explicit boolean")
        for name in ("wm_hidden", "policy_hidden", "critic_hidden"):
            values = getattr(self, name)
            if not values or any(type(v) is not int or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integer widths")
        for name in (
            "wm_window",
            "horizon",
            "wm_batch_size",
            "policy_batch_size",
            "critic_epochs",
            "critic_minibatches",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "max_grad_norm",
            "policy_grad_value_clip",
            "normalizer_max_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            isinstance(self.reset_start_fraction, bool)
            or not math.isfinite(self.reset_start_fraction)
            or not 0 <= self.reset_start_fraction < 1
        ):
            raise ValueError("reset_start_fraction must be finite and in [0, 1)")
        if not math.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise ValueError("gamma must be finite and in (0, 1)")
        if (
            isinstance(self.lambda_return, bool)
            or not math.isfinite(self.lambda_return)
            or not 0 <= self.lambda_return <= 1
        ):
            raise ValueError("lambda_return must be finite and in [0, 1]")
        if type(self.wm_ensemble_size) is not int or self.wm_ensemble_size < 1:
            raise ValueError("wm_ensemble_size must be a positive integer")
        penalty = self.uncertainty_penalty
        if isinstance(penalty, bool) or not math.isfinite(penalty) or penalty < 0:
            raise ValueError("uncertainty_penalty must be finite and nonnegative")
        if penalty > 0 and self.wm_ensemble_size < 2:
            raise ValueError("uncertainty_penalty requires wm_ensemble_size of at least 2")
        if self.critic_minibatches > self.horizon * self.policy_batch_size:
            raise ValueError("critic_minibatches cannot exceed the imagined transition count")
