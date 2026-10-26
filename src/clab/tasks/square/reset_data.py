"""Square reset poses use environment-local world axes, not robot-base axes."""

import torch

from clab.tasks import rigid_resets

from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT

SCHEMA = "clab.square.reset_bank/1"
SPEC = rigid_resets.RigidResetSpec(
    "square",
    SCHEMA,
    DEFAULT_ROBOT,
    RESET_TYPES,
    "nut",
    "peg",
    pose_frame="environment_world",
)
FIELDS = SPEC.fields
WIDTHS = SPEC.widths
bank_contract = SPEC.contract


def validate_bank(bank, model=None):
    states = rigid_resets.validate_bank(bank, SPEC, model)
    subtypes = bank.get("insertion_subtypes")
    if subtypes is not None:
        labels = subtypes.get("labels")
        names = subtypes.get("names")
        count = len(states[RESET_TYPES[3]]["joint_pos"])
        if (
            names not in (["near_peg", "above_peg", "threaded"], ["above_peg", "threaded"])
            or not isinstance(labels, torch.Tensor)
            or labels.device.type != "cpu"
            or labels.dtype != torch.int64
            or labels.shape != (count,)
            or not bool(((labels >= 0) & (labels < len(names))).all())
        ):
            raise ValueError("Invalid Square insertion subtype labels")
        counts = torch.bincount(labels, minlength=len(names)).tolist()
        if min(counts) < 1 or counts != subtypes.get("counts"):
            raise ValueError("Square insertion subtype counts must match their labels")
    return states


def bank_metadata(bank):
    validate_bank(bank)
    result = rigid_resets.bank_metadata(bank, SPEC)
    if "insertion_subtypes" in bank:
        result["insertion_subtypes"] = {
            key: bank["insertion_subtypes"][key] for key in ("names", "counts")
        }
    return result
