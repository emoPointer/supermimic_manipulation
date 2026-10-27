"""Square assembly: physical seating, release and stable success; any grasp of the nut.

The old source position predicate is deliberately not the active success rule.
Contact rules are measured at control boundaries, not every physics substep.
"""

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from clab.observations.state import pose
from clab.tasks import rewards
from clab.tensors import constant

from .props import (
    HOLE_HALF_WIDTH,
    NUT_GRASP_POINT,
    NUT_HALF_HEIGHT,
    NUT_SEATED_OFFSET,
    PEG_HALF_HEIGHT,
    PEG_HALF_WIDTH,
    handle_contact_bounds,
)
from .settings import AssemblyCriteria
from .variants import DEFAULT_ROBOT


def contact_found(env, name):
    return (env.scene[name].data.found > 0).any(dim=-1)


def threaded_geometry(nut_pos, nut_quat, peg_pos, peg_quat, *, axial_tolerance=0.0):
    """Diagnostic: the rod crosses both nut slab faces inside the square opening.

    Check all four longitudinal rod edges at z=+/-nut_half_height in the nut
    frame. This permits every physically fitting yaw, including a centred 45°
    peg, and rejects merely resting on the peg tip. The explicit axial endpoint
    tolerance permits the small base-contact compression already allowed by the
    penetration gate; it never enlarges the square opening. Not a stability test.
    """
    corners = constant(nut_pos, [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, -1)])
    lower = corners * constant(nut_pos, (PEG_HALF_WIDTH, PEG_HALF_WIDTH, PEG_HALF_HEIGHT))
    upper = lower.clone()
    upper[:, 2] = PEG_HALF_HEIGHT
    n = len(nut_pos)
    rotation = peg_quat[:, None].expand(n, 4, 4)
    inverse = nut_quat[:, None].expand(n, 4, 4)
    lower = quat_apply_inverse(
        inverse, peg_pos[:, None] + quat_apply(rotation, lower.expand(n, -1, -1)) - nut_pos[:, None]
    )
    upper = quat_apply_inverse(
        inverse, peg_pos[:, None] + quat_apply(rotation, upper.expand(n, -1, -1)) - nut_pos[:, None]
    )
    direction = upper - lower
    denominator = direction[..., 2]
    valid = (denominator.abs() > 1e-8).all(dim=-1)
    safe_denominator = torch.where(denominator.abs() > 1e-8, denominator, 1.0)
    fraction_tolerance = axial_tolerance / safe_denominator.abs()
    for height in (-NUT_HALF_HEIGHT, NUT_HALF_HEIGHT):
        fraction = (height - lower[..., 2]) / safe_denominator
        intersection = lower + fraction[..., None] * direction
        valid &= (
            (fraction >= -fraction_tolerance)
            & (fraction <= 1 + fraction_tolerance)
            & (intersection[..., :2].abs().amax(dim=-1) < HOLE_HALF_WIDTH)
        ).all(dim=-1)
    return valid


def insertion_diagnostic(env):
    return threaded_geometry(*pose(env, "nut"), *pose(env, "peg"))


def classify_finger_contacts(data, nut_pos, nut_quat, criteria):
    """Return force-bearing contact and exclusively exposed-handle contact.

    Each nut geom retains four strongest contact slots. Checking positions, not
    just geom4's name, excludes the handle/ring collision-box overlap. A tiny
    contact below the declared force threshold is not classified as a grasp.
    """
    lower, upper = handle_contact_bounds(criteria.handle_clearance, criteria.contact_tolerance)
    rotation = nut_quat[:, None].expand(-1, data.pos.shape[1], -1)
    local = quat_apply_inverse(rotation, data.pos - nut_pos[:, None])
    inside = ((local >= constant(local, lower)) & (local <= constant(local, upper))).all(-1)
    active = (data.found > 0) & (data.force.norm(dim=-1) >= criteria.finger_force_min)
    touching = active.any(-1)
    legal = (active & inside).any(-1) & ~(active & ~inside).any(-1)
    return touching, legal


def grasp_status(env, criteria):
    """Both fingers bear on the nut anywhere, and both bear only on the exposed handle."""
    nut_pos, nut_quat = pose(env, "nut")
    left, left_handle = classify_finger_contacts(
        env.scene["left_grasp_contact"].data, nut_pos, nut_quat, criteria
    )
    right, right_handle = classify_finger_contacts(
        env.scene["right_grasp_contact"].data, nut_pos, nut_quat, criteria
    )
    return left & right, left_handle & right_handle


def handle_grasp_status(env, criteria):
    """Reset collection, whose grasped rows hold the handle: handle grasp, any other grasp."""
    grasped, handle = grasp_status(env, criteria)
    return handle, grasped & ~handle


def seating_geometry(nut_pos, nut_quat, peg_pos, peg_quat, criteria):
    """Real rod-through-hole plus base-top height and a nearly horizontal slab.

    Nut faces are geometrically symmetric, so either face may point upwards;
    no arbitrary yaw or upright-face restriction is introduced.
    """
    relative = quat_apply_inverse(peg_quat, nut_pos - peg_pos)
    normal = constant(nut_pos, (0.0, 0.0, 1.0)).expand_as(nut_pos)
    nut_normal = quat_apply_inverse(peg_quat, quat_apply(nut_quat, normal))
    level = nut_normal[:, 2].abs() >= math.cos(math.radians(criteria.max_tilt_degrees))
    height = (relative[:, 2] - NUT_SEATED_OFFSET).abs() <= criteria.seated_height_tolerance
    return (
        threaded_geometry(
            nut_pos, nut_quat, peg_pos, peg_quat, axial_tolerance=criteria.max_penetration
        )
        & level
        & height
    )


class StableSuccess(ManagerTermBase):
    """Seated, supported, at rest and released for stable_seconds, however it was grasped.

    The per-episode grasp, handle-grasp and threading flags are diagnostics only.
    """

    flags = (
        "grasp_seen",
        "handle_grasp_seen",
        "threaded_seen",
        "grasped",
        "threaded",
        "seated",
        "candidate",
    )

    def __init__(self, cfg, env):
        super().__init__(env)
        self.criteria = AssemblyCriteria(**cfg.params["criteria"])
        self.counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.required_steps = math.ceil(self.criteria.stable_seconds / env.step_dt - 1e-9)
        for name in self.flags:
            setattr(self, name, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    def reset(self, env_ids=None):
        indices = env_ids if env_ids is not None else slice(None)
        self.counter[indices] = 0
        for name in self.flags:
            getattr(self, name)[indices] = False

    def __call__(self, env, *, criteria):
        del criteria  # Validated, immutable copy from the recorded term config.
        self.grasped[:], handle = grasp_status(env, self.criteria)
        self.grasp_seen |= self.grasped
        self.handle_grasp_seen |= handle
        self.threaded[:] = threaded_geometry(
            *pose(env, "nut"), *pose(env, "peg"), axial_tolerance=self.criteria.max_penetration
        )
        self.threaded_seen |= self.threaded
        nut_pos, nut_quat = pose(env, "nut")
        peg_pos, peg_quat = pose(env, "peg")
        base = env.scene["nut_base_support"].data
        peg_normal = quat_apply(peg_quat, constant(peg_pos, (0.0, 0.0, 1.0)).expand_as(peg_pos))
        support = (base.force * peg_normal[:, None]).sum(-1).abs().sum(-1)
        supported = (base.found > 0).any(-1) & (support >= self.criteria.support_force_min)
        safe = torch.ones_like(self.seated)
        for name in ("nut_peg_contact", "nut_table_contact", "nut_robot_contact"):
            data = env.scene[name].data
            safe &= ((data.found == 0) | (data.dist >= -self.criteria.max_penetration)).all(-1)
        self.seated[:] = (
            seating_geometry(nut_pos, nut_quat, peg_pos, peg_quat, self.criteria) & supported & safe
        )
        nut = env.scene["nut"].data
        stable = (nut.root_link_lin_vel_w.norm(dim=-1) <= self.criteria.linear_speed_max) & (
            nut.root_link_ang_vel_w.norm(dim=-1) <= self.criteria.angular_speed_max
        )
        released = ~contact_found(env, "nut_robot_contact")
        self.candidate[:] = self.seated & stable & released
        self.counter[:] = torch.where(self.candidate, self.counter + 1, 0)
        return self.counter >= self.required_steps


def progress(env):
    """Read the task-owned state updated by success before rewards."""
    return env.termination_manager.get_term_cfg("success").func


def reaching_reward(env, distance_scale):
    ee_pos, ee_quat = pose(env, "robot", DEFAULT_ROBOT.ee_body)
    nut_pos, nut_quat = pose(env, "nut")
    grasp = ee_pos + quat_apply(
        ee_quat, constant(ee_pos, DEFAULT_ROBOT.grasp_offset).expand_as(ee_pos)
    )
    handle = nut_pos + quat_apply(nut_quat, constant(nut_pos, NUT_GRASP_POINT).expand_as(nut_pos))
    state = progress(env)
    approach = ~state.grasped & ~state.seated
    return (1.0 - torch.tanh((grasp - handle).norm(dim=-1) / distance_scale)) * approach


def grasp_reward(env):
    return progress(env).grasped.float()


def alignment_reward(env, distance_scale, tilt_scale):
    nut_pos, nut_quat = pose(env, "nut")
    peg_pos, peg_quat = pose(env, "peg")
    relative = quat_apply_inverse(peg_quat, nut_pos - peg_pos)
    normal = constant(nut_pos, (0.0, 0.0, 1.0)).expand_as(nut_pos)
    cosine = quat_apply_inverse(peg_quat, quat_apply(nut_quat, normal))[:, 2].abs().clamp(0, 1)
    tilt = torch.acos(cosine)
    alignment = torch.exp(
        -(relative[:, :2].norm(dim=-1) / distance_scale).square() - (tilt / tilt_scale).square()
    )
    return alignment * progress(env).grasped


def insertion_reward(env, distance_scale):
    nut_pos, _ = pose(env, "nut")
    peg_pos, peg_quat = pose(env, "peg")
    offset = constant(peg_pos, (0.0, 0.0, NUT_SEATED_OFFSET)).expand_as(peg_pos)
    target = peg_pos + quat_apply(peg_quat, offset)
    return (
        torch.exp(-((nut_pos - target).norm(dim=-1) / distance_scale).square())
        * progress(env).threaded
    )


def release_reward(env):
    return progress(env).candidate.float()


def success_reward(env):
    return env.termination_manager.get_term("success").float()


def action_l2(env):
    return rewards.action_l2(env)


def action_rate_l2(env):
    return rewards.action_rate_l2(env)


def arm_velocity_l2(env):
    return rewards.joint_velocity_l2(env, num_joints=len(DEFAULT_ROBOT.arm_joints))
