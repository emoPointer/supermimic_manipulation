"""Lemon placement with control-interval contact sampling and a continuous hold."""

from __future__ import annotations

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply

from clab.observations.state import pose
from clab.tasks import rewards

from .settings import (
    SUCCESS_MAX_HEIGHT,
    SUCCESS_MIN_HEIGHT,
    SUCCESS_XY_TOLERANCE,
)
from .variants import DEFAULT_ROBOT


def placement_geometry(lemon_pos: torch.Tensor, plate_pos: torch.Tensor) -> torch.Tensor:
    """Placement thresholds on body origins, not bounding boxes."""
    delta = lemon_pos - plate_pos
    return (
        (torch.linalg.vector_norm(delta[:, :2], dim=-1) < SUCCESS_XY_TOLERANCE)
        & (delta[:, 2] > SUCCESS_MIN_HEIGHT)
        & (delta[:, 2] < SUCCESS_MAX_HEIGHT)
    )


def placement_candidate(
    lemon_pos: torch.Tensor,
    plate_pos: torch.Tensor,
    plate_contact: torch.Tensor,
    left_finger_contact: torch.Tensor,
    right_finger_contact: torch.Tensor,
) -> torch.Tensor:
    """Contact is required; a two-finger grasp is forbidden, a single touch is not.

    Deliberately no velocity threshold. StableSuccess measures plate contact
    across the current control interval and applies the configured hold duration.
    """
    return (
        placement_geometry(lemon_pos, plate_pos)
        & plate_contact.bool()
        & ~(left_finger_contact.bool() & right_finger_contact.bool())
    )


def contact_found(env, name: str) -> torch.Tensor:
    return (env.scene[name].data.found > 0).any(dim=-1)


def plate_contact_during_step(data, *, substeps: int) -> torch.Tensor:
    """Actual plate contact within this action, never a grace period across actions.

    The native sensor's zero distance also represents an empty slot, so only
    strictly negative history values establish historical contact. A current
    ``found`` additionally covers exact-touch contacts. The sensor clears its
    history on reset and overwrites all entries during each decimation interval.
    """
    history = data.dist_history
    if history is None or history.ndim != 3 or history.shape[-1] != substeps:
        raise ValueError("Lemon plate contact history must cover exactly one control step")
    return (data.found > 0).any(dim=-1) | (history < 0).any(dim=(-2, -1))


def narrow_grasp_alignment(env) -> torch.Tensor:
    """Absolute dot of jaw closure with the lemon's 39.7 mm body-local X axis.

    Used for reset acceptance only; it does not alter the source success rule.
    """
    _, ee_quat = pose(env, "robot", DEFAULT_ROBOT.ee_body)
    _, lemon_quat = pose(env, "lemon")
    jaw_axis = torch.tensor(DEFAULT_ROBOT.closing_axis, device=env.device).expand(env.num_envs, -1)
    narrow_axis = torch.tensor((1.0, 0.0, 0.0), device=env.device).expand(env.num_envs, -1)
    return (quat_apply(ee_quat, jaw_axis) * quat_apply(lemon_quat, narrow_axis)).sum(-1).abs()


class StableSuccess(ManagerTermBase):
    """Continuous placement for the configured duration, checked at control steps."""

    def __init__(self, cfg, env) -> None:
        super().__init__(env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.candidate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.success = torch.zeros_like(self.candidate)
        stable_seconds = cfg.params["stable_seconds"]
        if not math.isfinite(stable_seconds) or stable_seconds <= 0:
            raise ValueError("Lemon stable_seconds must be finite and positive")
        self.required_steps = max(1, math.ceil(stable_seconds / env.step_dt - 1e-9))
        self.plate_contact_substeps = cfg.params["plate_contact_substeps"]
        if (
            isinstance(self.plate_contact_substeps, bool)
            or not isinstance(self.plate_contact_substeps, int)
            or self.plate_contact_substeps < 1
            or self.plate_contact_substeps != env.cfg.decimation
        ):
            raise ValueError("Lemon plate_contact_substeps must equal environment decimation")

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        indices = env_ids if env_ids is not None else slice(None)
        self.counter[indices] = 0
        self.candidate[indices] = False
        self.success[indices] = False

    def __call__(self, env, *, stable_seconds: float, plate_contact_substeps: int) -> torch.Tensor:
        del stable_seconds, plate_contact_substeps  # Read from the saved config at construction.
        lemon_pos, _ = pose(env, "lemon")
        plate_pos, _ = pose(env, "plate")
        self.candidate[:] = placement_candidate(
            lemon_pos,
            plate_pos,
            plate_contact_during_step(
                env.scene["lemon_plate_contact"].data, substeps=self.plate_contact_substeps
            ),
            contact_found(env, "left_finger_contact"),
            contact_found(env, "right_finger_contact"),
        )
        self.counter[:] = torch.where(self.candidate, self.counter + 1, 0)
        self.success[:] = self.counter >= self.required_steps
        return self.success


def reaching_reward(env) -> torch.Tensor:
    ee_pos, ee_quat = pose(env, "robot", DEFAULT_ROBOT.ee_body)
    offset = torch.tensor(DEFAULT_ROBOT.grasp_offset, device=env.device).expand_as(ee_pos)
    grasp_pos = ee_pos + quat_apply(ee_quat, offset)
    lemon_pos, _ = pose(env, "lemon")
    return 1.0 - torch.tanh(torch.linalg.vector_norm(grasp_pos - lemon_pos, dim=-1))


def dense_goal_reward(env, goal_height: float) -> torch.Tensor:
    """Dense distance shaping towards the resting pose above the plate."""
    lemon_pos, _ = pose(env, "lemon")
    plate_pos, _ = pose(env, "plate")
    goal = plate_pos + torch.tensor((0.0, 0.0, goal_height), device=env.device)
    return torch.exp(-torch.linalg.vector_norm(lemon_pos - goal, dim=-1))


def success_reward(env) -> torch.Tensor:
    return env.termination_manager.get_term("success").float()


def action_l2(env) -> torch.Tensor:
    return rewards.action_l2(env)


def action_rate_l2(env) -> torch.Tensor:
    return rewards.action_rate_l2(env)


def arm_velocity_l2(env) -> torch.Tensor:
    return rewards.joint_velocity_l2(env, num_joints=len(DEFAULT_ROBOT.arm_joints))
