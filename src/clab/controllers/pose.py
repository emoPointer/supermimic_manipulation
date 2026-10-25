"""Convert world pose targets into the relative OSC action."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .osc import OscActionCfg


def pose_target_action(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    *,
    action_cfg: OscActionCfg,
    gripper_command: float,
) -> torch.Tensor:
    """World-frame pose error as scaled xyz / axis-angle commands plus the gripper.

    This does not move bodies, change gains, apply forces, or clip the action.
    Repeating a fixed target holds that pose; zero commands re-anchor each step.
    """
    from mjlab.utils.lab_api.math import compute_pose_error

    position_error, rotation_error = compute_pose_error(
        current_pos, current_quat, target_pos, target_quat
    )
    return torch.cat(
        (
            position_error / action_cfg.position_scale,
            rotation_error / action_cfg.orientation_scale,
            torch.full_like(position_error[:, :1], gripper_command),
        ),
        dim=-1,
    )
