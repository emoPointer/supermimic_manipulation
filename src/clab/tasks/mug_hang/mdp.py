"""Hanging success, including handle engagement and release."""

from __future__ import annotations

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from clab.observations.state import pose
from clab.tasks import rewards

from .props import (
    BRANCH_INNER,
    BRANCH_LENGTH,
    BRANCH_OUTER,
    HANDLE_ANCHOR,
    HANG_TARGET,
    MUG_GRASP_POINT,
)
from .settings import STABLE_SECONDS
from .variants import DEFAULT_ROBOT


def handle_geometry(mug_pos, mug_quat, tree_pos, tree_quat):
    """Finite-segment anchor distance, projection and signed handle-axis alignment.

    Equivalent to collection planner handle_engagement_report, not the relaxed
    environment's root-position envelope alone. All quaternions are wxyz.
    """
    inner = torch.tensor(BRANCH_INNER, device=mug_pos.device, dtype=mug_pos.dtype).expand_as(
        mug_pos
    )
    outer = torch.tensor(BRANCH_OUTER, device=mug_pos.device, dtype=mug_pos.dtype).expand_as(
        mug_pos
    )
    inner_mug = quat_apply_inverse(mug_quat, tree_pos + quat_apply(tree_quat, inner) - mug_pos)
    outer_mug = quat_apply_inverse(mug_quat, tree_pos + quat_apply(tree_quat, outer) - mug_pos)
    direction = (outer_mug - inner_mug) / BRANCH_LENGTH
    anchor = torch.tensor(HANDLE_ANCHOR, device=mug_pos.device, dtype=mug_pos.dtype)
    projection = ((anchor - inner_mug) * direction).sum(-1)
    closest = inner_mug + projection.clamp(0, BRANCH_LENGTH)[:, None] * direction
    distance = (anchor - closest).norm(dim=-1)
    return distance, projection, direction[:, 0]


def placement_candidate(
    mug_pos, mug_quat, tree_pos, tree_quat, mug_velocity, left, right, table_z=0.0
):
    relative = quat_apply_inverse(tree_quat, mug_pos - tree_pos)
    radial, projected, alignment = handle_geometry(mug_pos, mug_quat, tree_pos, tree_quat)
    return (
        (relative[:, 0].abs() < 0.085)
        & (relative[:, 1].abs() < 0.05)
        & (mug_pos[:, 2] > table_z + 0.11)
        & (mug_velocity.norm(dim=-1) < 0.01)
        & ~(left.bool() & right.bool())
        & (radial <= 0.018)
        & (projected >= -0.005)
        & (projected <= BRANCH_LENGTH + 0.005)
        & (alignment >= 0.75)
    )


def contact_found(env, name):
    return (env.scene[name].data.found > 0).any(dim=-1)


def success_candidate(env):
    mug_pos, mug_quat = pose(env, "mug")
    tree_pos, tree_quat = pose(env, "tree")
    return placement_candidate(
        mug_pos,
        mug_quat,
        tree_pos,
        tree_quat,
        env.scene["mug"].data.root_link_lin_vel_w,
        contact_found(env, "left_finger_contact"),
        contact_found(env, "right_finger_contact"),
        table_z=env.scene.env_origins[:, 2],
    )


class StableSuccess(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        seconds = cfg.params["stable_seconds"] if cfg is not None else STABLE_SECONDS
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("MugHang stable_seconds must be finite and positive")
        self.required_steps = math.ceil(seconds / env.step_dt - 1e-9)

    def reset(self, env_ids=None):
        self.counter[env_ids] = 0

    def __call__(self, env, stable_seconds=STABLE_SECONDS):
        del stable_seconds  # Read once from the saved termination configuration.
        self.counter[:] = torch.where(success_candidate(env), self.counter + 1, 0)
        return self.counter >= self.required_steps


def success_reward(env):
    return env.termination_manager.get_term("success").float()


def reaching_reward(env):
    mug_pos, mug_quat = pose(env, "mug")
    ee_pos, ee_quat = pose(env, "robot", DEFAULT_ROBOT.ee_body)
    offset = torch.tensor(DEFAULT_ROBOT.grasp_offset, device=env.device).expand_as(ee_pos)
    grasp = ee_pos + quat_apply(ee_quat, offset)
    target = mug_pos + quat_apply(
        mug_quat, torch.tensor(MUG_GRASP_POINT, device=env.device).expand_as(mug_pos)
    )
    return 1.0 - torch.tanh(10.0 * (grasp - target).norm(dim=-1))


def handle_to_hang_target(env):
    """Tree-frame target-minus-handle xyz in metres, toward the rod interior.

    The handle anchor is mug-local; the target is the branch's tree-local
    midpoint after insertion, not its outer entrance or either body root.
    Signed axis alignment remains available through ``mug_in_tree_frame``'s
    relative rotation vector. This position feature alone does not certify
    engagement, release or stable task success.
    """
    mug_pos, mug_quat = pose(env, "mug")
    tree_pos, tree_quat = pose(env, "tree")
    handle = mug_pos + quat_apply(mug_quat, mug_pos.new_tensor(HANDLE_ANCHOR).expand_as(mug_pos))
    target = tree_pos + quat_apply(tree_quat, tree_pos.new_tensor(HANG_TARGET).expand_as(tree_pos))
    return quat_apply_inverse(tree_quat, target - handle)


def threading_reward(env):
    mug_pos, mug_quat = pose(env, "mug")
    tree_pos, tree_quat = pose(env, "tree")
    _, _, alignment = handle_geometry(mug_pos, mug_quat, tree_pos, tree_quat)
    anchor = torch.tensor(HANDLE_ANCHOR, device=env.device).expand_as(mug_pos)
    midpoint = torch.tensor(HANG_TARGET, device=env.device).expand_as(tree_pos)
    distance = (
        mug_pos + quat_apply(mug_quat, anchor) - tree_pos - quat_apply(tree_quat, midpoint)
    ).norm(dim=-1)
    # Guide toward the rod interior, not a flat optimum at its outside tip.
    # This bounded shaping is not a replacement for physical hanging success.
    return torch.exp(-10.0 * distance) * alignment.clamp(0, 1)


def action_l2(env):
    return rewards.action_l2(env)


def action_rate_l2(env):
    return rewards.action_rate_l2(env)


def arm_velocity_l2(env):
    return rewards.joint_velocity_l2(env, num_joints=len(DEFAULT_ROBOT.arm_joints))
