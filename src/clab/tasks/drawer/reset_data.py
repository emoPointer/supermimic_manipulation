"""Articulated Drawer reset contract, including explicit offline process credit.

Structural validation does not prove that a prefix happened physically. A bank
collector must retain and validate real preceding trajectories. The physical
collector accepts rows only after a complete sequence; hand-assigned prefix
flags are not valid data.
"""

import math
from copy import deepcopy

import torch

from .settings import (
    EPISODE_SECONDS,
    INITIAL_SLIDE_RANGE,
    PREFIX_FLAGS,
    RESET_TYPES,
    SLIDE_RANGE,
    SUCCESS_SECONDS,
)
from .variants import DEFAULT_ROBOT

SCHEMA = "clab.drawer.reset_bank/1"
FIELDS = (
    "joint_pos",
    "joint_vel",
    "pod_pose",
    "pod_vel",
    "housing_pose",
    "slide_pos",
    "slide_vel",
    "sequence_prefix",
    "initial_drawer_qpos",
    "initial_pod_height",
    "close_start_qpos",
)
WIDTHS = (8, 8, 7, 6, 7, 1, 1, 2, 1, 1, 1)


def bank_contract():
    return {
        "task": "drawer",
        "objective": "initially_open_stable_grasp_lift_inside_release_contact_close_hold",
        "initial_slide_range_m": list(INITIAL_SLIDE_RANGE),
        "robot": DEFAULT_ROBOT.name,
        "finger_collision_geometry": DEFAULT_ROBOT.finger_collision_contract(),
        "joint_order": list(DEFAULT_ROBOT.joints),
        "fields": list(FIELDS),
        "frame": "environment_local_world",
        "pose": "xyz_wxyz",
        "position_unit": "m",
        "joint_units": ["rad"] * 6 + ["m"] * 2,
        "joint_velocity_units": ["rad/s"] * 6 + ["m/s"] * 2,
        "pod_velocity": "root_link_origin_environment_local_world_linear_xyz_angular_xyz",
        "pod_velocity_units": ["m/s"] * 3 + ["rad/s"] * 3,
        "housing": "fixed_mocap_pose_constant_within_episode",
        "slide": "passive_housing_local_positive_x_metres",
        "slide_velocity_unit": "m/s",
        "sequence_prefix_order": ["stable_grasp_lift", "released_inside"],
        "prefix_means": "physically_verified_offline_not_policy_episode_accomplishment",
        "reference_heights": "environment_local_world_z_metres",
        "categories": list(RESET_TYPES),
    }


def validate_bank(bank, model=None):
    if not isinstance(bank, dict) or bank.get("schema") != SCHEMA:
        raise ValueError("Unsupported Drawer reset schema")
    if bank.get("contract") != bank_contract():
        raise ValueError(
            "Drawer reset contract mismatch: incompatible task objective or finger collision geometry; "
            "opening-task banks are incompatible with the open-start task"
        )
    provenance = bank.get("provenance", {})
    if not isinstance(provenance, dict) or not provenance.get("method") or not bank.get("split"):
        raise ValueError("Drawer bank needs explicit method and split")
    if provenance.get("prefix_validation") != "physical_trajectory_witness":
        raise ValueError("Drawer prefix flags require physical trajectory witness provenance")
    states = bank.get("states")
    if not isinstance(states, dict) or tuple(states) != RESET_TYPES:
        raise ValueError(f"Drawer bank requires ordered categories {RESET_TYPES}")
    initial_rows = states[RESET_TYPES[0]]
    for category, (name, rows) in enumerate(states.items()):
        if not isinstance(rows, dict) or tuple(rows) != FIELDS:
            raise ValueError(f"{name}: expected ordered fields {FIELDS}")
        count = None
        for field, width in zip(FIELDS, WIDTHS, strict=True):
            value = rows[field]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[1] != width
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"{name}.{field}: expected finite CPU float32 [N,{width}]")
            if count is None:
                count = len(value)
            if not count or len(value) != count:
                raise ValueError(f"{name}: empty or inconsistent rows")
        if category > 0:
            if count != len(initial_rows["joint_pos"]):
                raise ValueError(f"{name}: paired Drawer categories need equal row counts")
            for field in ("housing_pose", "initial_drawer_qpos", "initial_pod_height"):
                if not torch.equal(rows[field], initial_rows[field]):
                    raise ValueError(f"{name}.{field}: paired trajectory reference mismatch")
        for field in ("pod_pose", "housing_pose"):
            if not torch.allclose(
                rows[field][:, 3:].norm(dim=-1), torch.ones(count), atol=1e-3, rtol=0
            ):
                raise ValueError(f"{name}.{field}: nonunit wxyz quaternion")
        expected = torch.tensor(PREFIX_FLAGS[category]).expand(count, -1)
        if not torch.equal(rows["sequence_prefix"], expected):
            raise ValueError(f"{name}: incorrect inherited process prefix")
        slide = rows["slide_pos"][:, 0]
        if ((slide < SLIDE_RANGE[0]) | (slide > SLIDE_RANGE[1])).any():
            raise ValueError(f"{name}: slide outside passive joint limits")
        initial = rows["initial_drawer_qpos"][:, 0]
        if (
            (initial < INITIAL_SLIDE_RANGE[0] - 1e-5) | (initial > INITIAL_SLIDE_RANGE[1] + 1e-5)
        ).any():
            raise ValueError(f"{name}: initial open-start slide reference must be 0.13 m")
        if category == 0 and not torch.allclose(slide, initial, atol=1e-5, rtol=0):
            raise ValueError("Full initial reset must preserve the initial open slide")
        if PREFIX_FLAGS[category][1]:
            if (slide < 0.03 + 0.015).any():
                raise ValueError("Late Drawer reset must remain open, not already successful")
            if not torch.allclose(rows["close_start_qpos"][:, 0], slide, atol=1e-5, rtol=0):
                raise ValueError("Released-pod reset must use its current slide as close reference")
    if model is not None:
        DEFAULT_ROBOT.validate_model(model)
        DEFAULT_ROBOT.validate_states(model, states)
    horizon = provenance.get("horizon_acceptance")
    if horizon is not None:
        control_dt = provenance.get("control_dt")
        if not isinstance(control_dt, (float, int)) or control_dt <= 0:
            raise ValueError("Drawer horizon provenance needs a positive control_dt")
        maximum = math.floor(EPISODE_SECONDS / control_dt + 1e-9)
        if horizon != {
            "reference": "task0_snapshot",
            "maximum_remaining_control_steps": maximum,
            "includes_continuous_final_success": True,
        }:
            raise ValueError("Drawer recorded horizon acceptance differs from the task")
        witnesses = provenance.get("physical_witnesses")
        if not isinstance(witnesses, list) or len(witnesses) != len(initial_rows["joint_pos"]):
            raise ValueError("Drawer needs one paired physical witness per accepted trajectory")
        for witness in witnesses:
            total, start, remaining = (
                witness.get(key)
                for key in ("control_steps", "task0_control_step", "remaining_control_steps")
            )
            if (
                not all(isinstance(value, int) for value in (total, start, remaining))
                or not 0 < remaining <= maximum
                or total - start != remaining
            ):
                raise ValueError("Drawer complete success is outside its task0-relative horizon")
            events = witness.get("events")
            if not isinstance(events, list) or [event.get("stage") for event in events] != [
                1,
                2,
                3,
            ]:
                raise ValueError("Drawer witness lacks the complete ordered physical stages")
            steps = [event.get("step") for event in events]
            if not all(isinstance(step, int) for step in steps) or not all(
                left < right
                for left, right in zip([start, *steps], [*steps, total + 1], strict=True)
            ):
                raise ValueError("Drawer physical stage times are not ordered after task0")
            if total - events[-1]["step"] + 1 < math.ceil(SUCCESS_SECONDS / control_dt - 1e-9):
                raise ValueError("Drawer witness lacks the continuous final success duration")
    return states


def bank_metadata(bank):
    states = validate_bank(bank)
    return {
        "contract": deepcopy(bank["contract"]),
        "provenance": deepcopy(bank["provenance"]),
        "split": bank["split"],
        "counts": {name: len(rows["joint_pos"]) for name, rows in states.items()},
    }
