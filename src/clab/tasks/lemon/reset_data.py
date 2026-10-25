"""Strict, task-owned Lemon reset-bank contract and structural validation.

Banks are local immutable tensors, not demonstration trajectories. Validation
checks structure and metadata; the collector separately checks physical states.
Pass a compiled model to additionally check the current robot's joint limits.
"""

from __future__ import annotations

import torch

from clab.tasks import rigid_resets

from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT

SCHEMA = "clab.lemon.reset_bank/1"
SPEC = rigid_resets.RigidResetSpec(
    "lemon",
    SCHEMA,
    DEFAULT_ROBOT,
    RESET_TYPES,
    "lemon",
    "plate",
    pose_frame="environment_world",
)
FIELDS, WIDTHS = SPEC.fields, SPEC.widths


def bank_contract() -> dict:
    return SPEC.contract()


def validate_bank(bank: object, model=None) -> dict[str, dict[str, torch.Tensor]]:
    return rigid_resets.validate_bank(bank, SPEC, model)


def bank_metadata(bank: dict) -> dict:
    return rigid_resets.bank_metadata(bank, SPEC)
