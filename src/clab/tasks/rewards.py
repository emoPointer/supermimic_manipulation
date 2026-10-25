"""Shared raw-action and joint-velocity costs; task weights remain task-owned.

Task MDP functions delegate here but retain their recorded configuration paths.
Costs include the raw binary gripper action and exclude finger joint velocities,
matching the existing formulas; controller clipping does not change these inputs.
"""

import torch


def action_l2(env) -> torch.Tensor:
    return env.action_manager.action.square().sum(-1).clamp_max(1e4)


def action_rate_l2(env) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    return delta.square().sum(-1).clamp_max(1e4)


def joint_velocity_l2(env, *, num_joints: int, entity_name: str = "robot") -> torch.Tensor:
    return env.scene[entity_name].data.joint_vel[:, :num_joints].square().sum(-1).clamp_max(1e4)
