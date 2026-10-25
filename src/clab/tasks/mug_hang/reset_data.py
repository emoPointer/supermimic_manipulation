"""MugHang-specific state and category contract over shared rigid-bank mechanics."""

import torch

from clab.tasks import rigid_resets

from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT

SCHEMA = "clab.mug_hang.reset_bank/1"
SPEC = rigid_resets.RigidResetSpec(
    "mug_hang", SCHEMA, DEFAULT_ROBOT, RESET_TYPES, "mug", "tree", pose_frame="environment_world"
)
FIELDS, WIDTHS = SPEC.fields, SPEC.widths


def bank_contract():
    return SPEC.contract()


def validate_bank(bank, model=None):
    states = rigid_resets.validate_bank(bank, SPEC, model)
    subtypes = bank.get("insertion_subtypes")
    if subtypes is not None:
        labels = subtypes.get("labels")
        count = len(states[RESET_TYPES[3]]["joint_pos"])
        if (
            subtypes.get("names") != ["outside_tip", "threaded"]
            or not isinstance(labels, torch.Tensor)
            or labels.device.type != "cpu"
            or labels.dtype != torch.int64
            or labels.shape != (count,)
            or not bool(((labels == 0) | (labels == 1)).all())
        ):
            raise ValueError("Invalid MugHang insertion subtype labels")
        counts = torch.bincount(labels, minlength=2).tolist()
        if min(counts) < 1 or counts != subtypes.get("counts"):
            raise ValueError("MugHang insertion subtype counts disagree with labels")
    return states


def bank_metadata(bank):
    return rigid_resets.bank_metadata(bank, SPEC)
