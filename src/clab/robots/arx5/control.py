"""ARX5 operational-space control preset and simulation timing.

Finger gains and the 5 N per-finger force limit belong to the robot asset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .description import ARM_JOINTS, EE_BODY, FINGER_JOINTS

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

    from clab.controllers.osc import OscActionCfg

PHYSICS_DT = 1.0 / 480.0
DECIMATION = 24  # 20 Hz policy.


def osc_action_cfg() -> OscActionCfg:
    """20 Hz world-axis relative pose commands with a binary gripper."""
    from clab.controllers.osc import OscActionCfg

    return OscActionCfg(
        entity_name="robot",
        arm_joints=ARM_JOINTS,
        finger_joints=FINGER_JOINTS,
        ee_body=EE_BODY,
        position_scale=0.02,
        orientation_scale=0.2,
        motion_stiffness=(200.0, 200.0, 200.0, 300.0, 300.0, 300.0),
        motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
        effort_limit=50.0,
        inertia_solver="pinv",
        uncouple_pos_ori=True,
        closed_position=0.002,
        open_position=0.044,
        input_limit=12.0,
        linear_velocity_limit=1.5,
        angular_velocity_limit=8.0,
    )


def robot_and_action_cfg(
    layout: dict | None = None,
    *,
    action_cfg: OscActionCfg | None = None,
    finger_collision_geometry: str = "source_mesh_convex_hull",
) -> tuple[EntityCfg, OscActionCfg]:
    from .model import robot_cfg

    return (
        robot_cfg(layout, finger_collision_geometry=finger_collision_geometry),
        osc_action_cfg() if action_cfg is None else action_cfg,
    )
