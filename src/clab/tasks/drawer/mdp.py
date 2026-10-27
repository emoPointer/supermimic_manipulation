"""Drawer physical process, source final predicate and new staged RL shaping."""

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from clab.observations.state import pose
from clab.tasks import rewards

from .process import PhysicalSequence, final_candidate, pod_inside
from .reset_data import bank_contract
from .variants import DEFAULT_ROBOT


def sequence(env):
    """Task-owned history, initialized before observation shape inference."""
    if not hasattr(env, "drawer_sequence"):
        env.drawer_sequence = PhysicalSequence(env.num_envs, env.device, env.step_dt)
    return env.drawer_sequence


def process_observation(env):
    return sequence(env).observation()


def slide_state(env):
    data = env.scene["fixture"].data
    return torch.cat((data.joint_pos, data.joint_vel), -1)


def contact_found(env, name):
    return (env.scene[name].data.found > 0).any(-1)


def geometry(env):
    pod_pos, _ = pose(env, "pod")
    housing_pos, housing_quat = pose(env, "fixture", "housing")
    local = quat_apply_inverse(housing_quat, pod_pos - housing_pos)
    height = pod_pos[:, 2] - env.scene.env_origins[:, 2]
    slide = env.scene["fixture"].data.joint_pos[:, 0]
    grasped = contact_found(env, "left_pod_contact") & contact_found(env, "right_pod_contact")
    return local, height, slide, grasped


class SequenceSuccess(ManagerTermBase):
    """Advance exactly once per control step before rewards/observations.

    The open-start task needs no insertion geometry. Collision guards remain
    independent of these grasp/lift, release and contact-close milestones.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        params = getattr(cfg, "params", None)
        if not isinstance(params, dict) or params.get("task_contract") != bank_contract():
            raise ValueError(
                "Drawer success task contract mismatch: explicit current task and finger collision geometry required"
            )
        self.state = sequence(env)

    def reset(self, env_ids=None):
        # Bank reset has already initialized the history before manager reset.
        # Clearing here would discard late-reset provenance.
        del env_ids

    def __call__(self, env, task_contract):
        # Managers forward the already construction-validated parameter. Do not
        # reload geometry manifests on each 20 Hz control step.
        del task_contract
        local, height, slide, grasped = geometry(env)
        return self.state.update(
            slide=slide,
            pod_z=height,
            inside=pod_inside(local, height, slide),
            grasped=grasped,
            drawer_contact=contact_found(env, "left_drawer_contact")
            | contact_found(env, "right_drawer_contact"),
            final=final_candidate(local, height, slide, grasped),
        )


def success_reward(env):
    return env.termination_manager.get_term("success").float()


def staged_reaching(env):
    """Dense distances to a physical milestone; not the source scripted reward."""
    state = sequence(env)
    ee_pos, ee_quat = pose(env, "robot", DEFAULT_ROBOT.ee_body)
    grip = ee_pos + quat_apply(
        ee_quat, torch.tensor(DEFAULT_ROBOT.grasp_offset, device=env.device).expand_as(ee_pos)
    )
    housing_pos, housing_quat = pose(env, "fixture", "housing")
    pod_pos, _ = pose(env, "pod")
    slide = env.scene["fixture"].data.joint_pos[:, 0]
    goal_x = slide + 0.099
    goal_local = torch.stack((goal_x, torch.zeros_like(slide), torch.full_like(slide, 0.04)), -1)
    panel = housing_pos + quat_apply(housing_quat, goal_local)
    target = torch.where((state.stage < 2)[:, None], pod_pos, panel)
    return 1 - torch.tanh(10 * torch.linalg.vector_norm(grip - target, dim=-1))


def staged_progress(env):
    state = sequence(env)
    local, height, slide, _ = geometry(env)
    lift_progress = ((height - state.initial_pod_z) / 0.04).clamp(0, 1)
    target = torch.stack((slide, torch.zeros_like(slide), torch.full_like(slide, 0.03)), -1)
    place_progress = torch.exp(-10 * torch.linalg.vector_norm(local - target, dim=-1))
    close_progress = 1 - torch.tanh(10 * slide.abs())
    return torch.where(
        state.stage == 0,
        lift_progress,
        torch.where(state.stage == 1, place_progress, close_progress),
    )


def action_cost(env):
    return rewards.action_l2(env)


def action_rate_cost(env):
    return rewards.action_rate_l2(env)


def arm_velocity_cost(env):
    return rewards.joint_velocity_l2(env, num_joints=len(DEFAULT_ROBOT.arm_joints))
