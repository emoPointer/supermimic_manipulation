"""BoxInBin reset-bank contract and structural validation (robot-base-local poses).

Validation checks structure and metadata, not physical reachability or grasp
stability; the collector checks physical holds before export.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT

FIELDS = ("joint_pos", "joint_vel", "box_pose", "box_vel", "bin_pose")
WIDTHS = (len(DEFAULT_ROBOT.joints), len(DEFAULT_ROBOT.joints), 7, 6, 7)


def bank_contract() -> dict:
    return {
        "task": "box_in_bin",
        "robot": DEFAULT_ROBOT.name,
        "joint_order": list(DEFAULT_ROBOT.joints),
        "frame": "robot_base_local",
        "pose": "xyz_wxyz",
        "joint_units": ["rad"] * len(DEFAULT_ROBOT.arm_joints)
        + ["m"] * len(DEFAULT_ROBOT.finger_joints),
        "joint_velocity_units": ["rad/s"] * len(DEFAULT_ROBOT.arm_joints)
        + ["m/s"] * len(DEFAULT_ROBOT.finger_joints),
        "position_unit": "m",
        "box_velocity": "local_linear_xyz_angular_xyz",
        "box_velocity_units": ["m/s"] * 3 + ["rad/s"] * 3,
        "categories": list(RESET_TYPES),
    }


def bank_metadata(bank: dict) -> dict:
    validate_bank(bank)
    return {
        "contract": deepcopy(bank["contract"]),
        "provenance": deepcopy(bank["provenance"]),
        "split": bank["split"],
        "counts": {kind: len(rows["joint_pos"]) for kind, rows in bank["states"].items()},
    }


def validate_bank(bank: object, model=None) -> dict[str, dict[str, torch.Tensor]]:
    if not isinstance(bank, dict) or bank.get("schema") != "clab.box_in_bin.reset_bank/1":
        raise ValueError("Unsupported BoxInBin reset bank schema")
    if bank.get("contract") != bank_contract():
        raise ValueError(f"Incompatible reset bank contract: {bank.get('contract')}")
    if (
        not isinstance(bank.get("provenance"), dict)
        or not bank["provenance"].get("method")
        or not isinstance(bank.get("split"), str)
        or not bank["split"]
    ):
        raise ValueError("A BoxInBin reset bank requires provenance.method and split")
    states = bank.get("states")
    if not isinstance(states, dict) or tuple(states) != RESET_TYPES:
        raise ValueError(f"Reset bank must contain exactly {RESET_TYPES}")
    for kind, rows in states.items():
        if not isinstance(rows, dict) or tuple(rows) != FIELDS:
            raise ValueError(f"{kind}: expected fields {FIELDS}")
        count = None
        for field, width in zip(FIELDS, WIDTHS, strict=True):
            value = rows[field]
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f"{kind}.{field}: expected [N, {width}] tensor")
            if (
                value.device.type != "cpu"
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"{kind}.{field}: expected finite CPU float32 tensor")
            if count is None:
                count = value.shape[0]
            elif value.shape[0] != count:
                raise ValueError(f"{kind}: inconsistent row count")
        for field in ("box_pose", "bin_pose"):
            norms = torch.linalg.vector_norm(rows[field][:, 3:], dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=0):
                raise ValueError(f"{kind}.{field}: quaternions must be normalized wxyz")
        if not count:
            raise ValueError(f"{kind}: empty reset set")
    if model is not None:
        DEFAULT_ROBOT.validate_model(model)
        DEFAULT_ROBOT.validate_states(model, states)
    return states
