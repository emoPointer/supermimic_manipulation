"""Rigid state observations; velocities are vectors, never translated points."""

from __future__ import annotations

import torch


def pose(env, entity: str, body: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """World pose of an entity root or an explicitly named link origin."""
    if body is None:
        data = env.scene[entity].data
        return data.root_link_pos_w, data.root_link_quat_w
    body_id = env.sim.mj_model.body(f"{entity}/{body}").id
    return env.sim.data.xpos[:, body_id], env.sim.data.xquat[:, body_id]


def observation_pose(env, entity: str, body: str | None = None):
    """Pose observed by the policy and used for controller targets (physical)."""
    return pose(env, entity, body)


def relative_pose(
    env,
    target: str,
    reference: str,
    target_body: str | None = None,
    reference_body: str | None = None,
) -> torch.Tensor:
    """Target in reference coordinates, ordered xyz (m), rotation vector (rad)."""
    # Import at use time: mjlab discovers task entry points during its import,
    # and those tasks also import these reusable observation functions.
    from mjlab.utils.lab_api.math import axis_angle_from_quat, subtract_frame_transforms

    target_pos, target_quat = observation_pose(env, target, target_body)
    reference_pos, reference_quat = observation_pose(env, reference, reference_body)
    pos, quat = subtract_frame_transforms(reference_pos, reference_quat, target_pos, target_quat)
    return torch.cat((pos, axis_angle_from_quat(quat)), dim=-1)


def velocity(env, entity: str, reference: str, body: str | None = None) -> torch.Tensor:
    """Link-origin linear/angular velocity resolved in the reference root frame.

    This changes coordinates only; it is not relative velocity with a moving
    reference. The intended reference is a fixed robot base.
    """
    from mjlab.entity.data import compute_velocity_from_cvel
    from mjlab.utils.lab_api.math import quat_apply_inverse

    if body is None:
        data = env.scene[entity].data
        linear, angular = data.root_link_lin_vel_w, data.root_link_ang_vel_w
    else:
        body_id = env.sim.mj_model.body(f"{entity}/{body}").id
        root_id = env.scene[entity].indexing.root_body_id
        world = compute_velocity_from_cvel(
            env.sim.data.xpos[:, body_id],
            env.sim.data.subtree_com[:, root_id],
            env.sim.data.cvel[:, body_id],
        )
        linear, angular = world[:, :3], world[:, 3:]
    _, reference_quat = pose(env, reference)
    return torch.cat(
        (quat_apply_inverse(reference_quat, linear), quat_apply_inverse(reference_quat, angular)),
        dim=-1,
    )


def joint_pos(env, entity: str = "robot") -> torch.Tensor:
    return env.scene[entity].data.joint_pos


def joint_vel(env, entity: str = "robot") -> torch.Tensor:
    return env.scene[entity].data.joint_vel


def previous_actions(env) -> torch.Tensor:
    return env.action_manager.action


def time_left(env) -> torch.Tensor:
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length)[:, None]


def geom_friction(env, entity: str, names: tuple[str, ...]) -> torch.Tensor:
    """Actual MuJoCo sliding, torsional and rolling coefficients per named geom."""
    ids = [env.sim.mj_model.geom(f"{entity}/{name}").id for name in names]
    return env.sim.model.geom_friction[:, ids].flatten(start_dim=1)


def body_mass(env, entity: str, names: tuple[str, ...]) -> torch.Tensor:
    """Masses in kg, in the caller's explicit body order."""
    ids = [env.sim.mj_model.body(f"{entity}/{name}").id for name in names]
    return env.sim.model.body_mass[:, ids]
