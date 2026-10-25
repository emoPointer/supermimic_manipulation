"""BoxInBin observations, rewards, and strict 2 s success for mjlab."""

from __future__ import annotations

import math

import torch
from mjlab.entity.data import compute_velocity_from_cvel
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_apply,
    subtract_frame_transforms,
)

from clab.tasks import rewards

from .settings import (
    BIN_INNER_HALF_EXTENTS,
    BOX_HALF_EXTENTS,
    STABLE_SECONDS,
    SUPPORT_WEIGHT_FRACTION,
)
from .variants import DEFAULT_ROBOT


def _pose(env, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "ee":
        body_id = env.sim.mj_model.body(f"robot/{DEFAULT_ROBOT.ee_body}").id
        return env.sim.data.xpos[:, body_id], env.sim.data.xquat[:, body_id]
    entity = env.scene[name]
    return entity.data.root_link_pos_w, entity.data.root_link_quat_w


def relative_pose(env, target: str, reference: str) -> torch.Tensor:
    """Target pose in reference frame as xyz + axis-angle, in the declared observation order."""
    from clab.observations.state import observation_pose

    def observed(name):
        if name == "ee":
            return observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)
        return _pose(env, name)

    target_pos, target_quat = observed(target)
    reference_pos, reference_quat = observed(reference)
    pos, quat = subtract_frame_transforms(reference_pos, reference_quat, target_pos, target_quat)
    return torch.cat((pos, axis_angle_from_quat(quat)), dim=-1)


def joint_pos(env) -> torch.Tensor:
    from clab.observations.state import joint_pos as observed_joint_pos

    return observed_joint_pos(env)


def previous_actions(env) -> torch.Tensor:
    return env.action_manager.action


def time_left(env) -> torch.Tensor:
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length)[:, None]


def joint_vel(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel


def end_effector_vel_lin_ang_b(env) -> torch.Tensor:
    """COM velocities with BoxInBin's explicit point-transform convention.

    The root position is subtracted as for a point, not just a vector rotation.
    This is the task's current numerical observation contract.
    """
    body_id = env.sim.mj_model.body(f"robot/{DEFAULT_ROBOT.ee_body}").id
    velocity_w = compute_velocity_from_cvel(
        env.sim.data.xipos[:, body_id],
        env.sim.data.subtree_com[:, env.scene["robot"].indexing.root_body_id],
        env.sim.data.cvel[:, body_id],
    )
    root_pos, root_quat = _pose(env, "robot")
    linear, _ = subtract_frame_transforms(root_pos, root_quat, velocity_w[:, :3])
    angular, _ = subtract_frame_transforms(root_pos, root_quat, velocity_w[:, 3:])
    return torch.cat((linear, angular), dim=-1)


_ROBOT_COLLISION_GEOMS = DEFAULT_ROBOT.collision_geoms
_BIN_COLLISION_GEOMS = (
    "floor",
    "wall_negative_y",
    "wall_negative_x",
    "wall_positive_y",
    "wall_positive_x",
)
_ROBOT_RIGID_BODIES = DEFAULT_ROBOT.rigid_bodies


def material_properties(env, entity: str) -> torch.Tensor:
    """Collision-material triplets in checkpoint order: static, dynamic, restitution.

    MuJoCo has one sliding coefficient, not separate static/dynamic coefficients.
    Both friction slots therefore report the *actual* sliding coefficient; the
    restitution slot is zero for this inelastic MuJoCo model. This is a
    cross-engine approximation, not a claim of identical contact laws.
    """
    if entity == "robot":
        names = _ROBOT_COLLISION_GEOMS
    elif entity == "box":
        names = ("box_collision",)
    elif entity == "bin":
        names = _BIN_COLLISION_GEOMS
    elif entity == "table":
        names = ("table_collision",)
    else:
        raise ValueError(f"Unknown material entity: {entity}")
    ids = torch.tensor(
        [env.sim.mj_model.geom(f"{entity}/{name}").id for name in names],
        dtype=torch.long,
        device=env.device,
    )
    sliding = env.sim.model.geom_friction[:, ids, 0]
    triplets = torch.stack((sliding, sliding, torch.zeros_like(sliding)), dim=-1)
    values = triplets.flatten(start_dim=1)
    # The checkpoint contract repeats the table triplet into a 21-D slot.
    return values.repeat(1, 7) if entity == "table" else values


def rigid_body_mass(env, entity: str) -> torch.Tensor:
    names = _ROBOT_RIGID_BODIES if entity == "robot" else (entity,)
    ids = torch.tensor(
        [env.sim.mj_model.body(f"{entity}/{name}").id for name in names],
        dtype=torch.long,
        device=env.device,
    )
    return env.sim.model.body_mass[:, ids]


def robot_joint_friction(env) -> torch.Tensor:
    return DEFAULT_ROBOT.robot_joint_friction(env)


def robot_joint_armature(env) -> torch.Tensor:
    return DEFAULT_ROBOT.robot_joint_armature(env)


def robot_joint_stiffness(env) -> torch.Tensor:
    return DEFAULT_ROBOT.robot_joint_stiffness(env)


def robot_joint_damping(env) -> torch.Tensor:
    return DEFAULT_ROBOT.robot_joint_damping(env)


def placement_geometry(
    box_pos: torch.Tensor, box_quat: torch.Tensor, bin_pos: torch.Tensor, bin_quat: torch.Tensor
) -> torch.Tensor:
    """Entire rotated collision box must fit and rest on the bin floor."""
    # Exact 3D rotated-box projection onto the yaw-only bin frame.
    bw, bx, by, bz = bin_quat.unbind(-1)
    yaw = torch.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by.square() + bz.square()))
    cy, sy = yaw.cos(), yaw.sin()
    delta = box_pos - bin_pos
    local_x = cy * delta[:, 0] + sy * delta[:, 1]
    local_y = -sy * delta[:, 0] + cy * delta[:, 1]
    basis = torch.eye(3, dtype=box_pos.dtype, device=box_pos.device).expand(len(box_pos), -1, -1)
    axes = torch.stack([quat_apply(box_quat, basis[:, i]) for i in range(3)], dim=1)
    rotated_x = cy[:, None] * axes[:, :, 0] + sy[:, None] * axes[:, :, 1]
    rotated_y = -sy[:, None] * axes[:, :, 0] + cy[:, None] * axes[:, :, 1]
    half = torch.tensor(BOX_HALF_EXTENTS, dtype=box_pos.dtype, device=box_pos.device)
    ext_x = (rotated_x.abs() * half).sum(-1)
    ext_y = (rotated_y.abs() * half).sum(-1)
    ext_z = (axes[:, :, 2].abs() * half).sum(-1)
    bottom_z = box_pos[:, 2] - ext_z
    floor_top_z = bin_pos[:, 2] - 0.010
    return (
        (local_x.abs() + ext_x < BIN_INNER_HALF_EXTENTS[0])
        & (local_y.abs() + ext_y < BIN_INNER_HALF_EXTENTS[1])
        & ((bottom_z - floor_top_z).abs() <= 0.005)
        & (box_pos[:, 2] > 0.0)
        & (box_pos[:, 2] < 0.100)
    )


class StableSuccess(ManagerTermBase):
    """Track stable supported placement for terminal success."""

    def __init__(self, cfg, env) -> None:
        super().__init__(env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.candidate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        seconds = cfg.params["stable_seconds"] if cfg is not None else STABLE_SECONDS
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("BoxInBin stable_seconds must be finite and positive")
        self.required_steps = max(1, math.ceil(seconds / env.step_dt - 1e-9))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        indices = env_ids if env_ids is not None else slice(None)
        self.counter[indices] = 0
        self.candidate[indices] = False
        self.success[indices] = False

    def __call__(
        self, env, support_weight_fraction=SUPPORT_WEIGHT_FRACTION, stable_seconds=STABLE_SECONDS
    ) -> torch.Tensor:
        del stable_seconds  # Read once from the saved termination configuration.
        if (
            isinstance(support_weight_fraction, bool)
            or not math.isfinite(support_weight_fraction)
            or not 0 < support_weight_fraction <= 1
        ):
            raise ValueError("Support weight fraction must be finite and in (0, 1]")
        box = env.scene["box"]
        box_pos, box_quat = _pose(env, "box")
        bin_pos, bin_quat = _pose(env, "bin")
        placed = placement_geometry(box_pos, box_quat, bin_pos, bin_quat)
        left = env.scene["left_finger_contact"].data.found[:, 0] > 0
        right = env.scene["right_finger_contact"].data.found[:, 0] > 0
        # mjlab reports box→bin force; both it and gravity point downward
        # under static support. Read live per-world mass/gravity, not CPU defaults.
        gravity = env.sim.model.opt.gravity[:]
        gravity_magnitude = torch.linalg.vector_norm(gravity, dim=-1)
        mass = rigid_body_mass(env, "box")[:, 0]
        contact_force = env.scene["box_bin_contact"].data.force[:, 0]
        supported_weight = (contact_force * gravity).sum(dim=-1) / gravity_magnitude.clamp_min(
            1e-12
        )
        support = (
            (mass > 0)
            & (gravity_magnitude > 0)
            & (supported_weight >= support_weight_fraction * mass * gravity_magnitude)
        )
        linear_slow = torch.linalg.vector_norm(box.data.root_link_lin_vel_w, dim=-1) <= 0.01
        angular_slow = torch.linalg.vector_norm(box.data.root_link_ang_vel_w, dim=-1) <= 0.1
        self.candidate[:] = placed & ~left & ~right & support & linear_slow & angular_slow
        self.counter[:] = torch.where(self.candidate, self.counter + 1, 0)
        self.success[:] = self.counter >= self.required_steps
        return self.success


def reaching_reward(env) -> torch.Tensor:
    pos, quat = _pose(env, "ee")
    offset = torch.tensor(DEFAULT_ROBOT.grasp_offset, device=env.device).expand_as(pos)
    grasp_pos = pos + quat_apply(quat, offset)
    box_pos, _ = _pose(env, "box")
    return 1.0 - torch.tanh(torch.linalg.vector_norm(grasp_pos - box_pos, dim=-1))


def dense_goal_reward(env) -> torch.Tensor:
    box_pos, _ = _pose(env, "box")
    bin_pos, _ = _pose(env, "bin")
    goal = bin_pos + torch.tensor((0.0, 0.0, 0.0405), device=env.device)
    return torch.exp(-torch.linalg.vector_norm(box_pos - goal, dim=-1))


def success_reward(env) -> torch.Tensor:
    return env.termination_manager.get_term("success").float()


def action_l2(env) -> torch.Tensor:
    return rewards.action_l2(env)


def action_rate_l2(env) -> torch.Tensor:
    return rewards.action_rate_l2(env)


def arm_velocity_l2(env) -> torch.Tensor:
    return rewards.joint_velocity_l2(env, num_joints=len(DEFAULT_ROBOT.arm_joints))
