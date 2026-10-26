"""State-only Lemon composition using reusable robot, controller and scene modules."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, nan_detection, time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from clab.observations import state
from clab.scenes import workcell
from clab.scenes.robosuite_table import ground_cfg

from . import mdp
from .props import (
    LEMON_BOTTOM_OFFSET,
    LEMON_COLLISION_GEOMS,
    PLATE_CENTRE_FLOOR_Z,
    PLATE_COLLISION_GEOMS,
    lemon_cfg,
    plate_cfg,
)
from .resets import ResetFromBank
from .settings import EPISODE_SECONDS, STABLE_SECONDS
from .variants import DEFAULT_ROBOT


def _contact_sensors() -> tuple[ContactSensorCfg, ...]:
    lemon = ContactMatch(mode="body", pattern="lemon", entity="lemon")
    plate = ContactMatch(mode="body", pattern="plate", entity="plate")
    table = ContactMatch(mode="body", pattern="table", entity="table")
    robot = ContactMatch(mode="subtree", pattern=DEFAULT_ROBOT.root_body, entity="robot")
    pairs = (
        (
            "left_finger_contact",
            ContactMatch(mode="body", pattern=DEFAULT_ROBOT.finger_bodies[0], entity="robot"),
            lemon,
        ),
        (
            "right_finger_contact",
            ContactMatch(mode="body", pattern=DEFAULT_ROBOT.finger_bodies[1], entity="robot"),
            lemon,
        ),
        ("lemon_plate_contact", lemon, plate),
        ("lemon_table_contact", lemon, table),
        ("lemon_robot_contact", lemon, robot),
        ("robot_table_contact", robot, table),
        ("robot_contact", robot, None),
    )
    return tuple(
        ContactSensorCfg(
            name=name,
            primary=primary,
            secondary=secondary,
            fields=("found", "dist"),
            reduce="mindist",
            # Observe the whole action interval, not only its final 0.5 ms.
            history_length=DEFAULT_ROBOT.decimation if name == "lemon_plate_contact" else 0,
        )
        for name, primary, secondary in pairs
    )


def make_env_cfg(
    *, play: bool = False, reset_bank: str | Path | None = None
) -> ManagerBasedRlEnvCfg:
    """Finite 32 s task: half-second placement success terminates with actual +10.

    Actor terms have five historical frames each, oldest to newest, concatenated
    in the order below. Critic sees current actor state plus time, velocities and
    actual physical parameters.
    """
    actor_terms = {
        "lemon_in_plate_frame": ObservationTermCfg(
            func=state.relative_pose, params={"target": "lemon", "reference": "plate"}
        ),
        "previous_actions": ObservationTermCfg(func=state.previous_actions),
        "joint_pos": ObservationTermCfg(func=state.joint_pos),
        "ee_in_base_frame": ObservationTermCfg(
            func=state.relative_pose,
            params={"target": "robot", "target_body": DEFAULT_ROBOT.ee_body, "reference": "robot"},
        ),
        "lemon_in_ee_frame": ObservationTermCfg(
            func=state.relative_pose,
            params={
                "target": "lemon",
                "reference": "robot",
                "reference_body": DEFAULT_ROBOT.ee_body,
            },
        ),
        "plate_in_ee_frame": ObservationTermCfg(
            func=state.relative_pose,
            params={
                "target": "plate",
                "reference": "robot",
                "reference_body": DEFAULT_ROBOT.ee_body,
            },
        ),
    }
    # 39 + 1 + 8 + 6 + 6 + (9+2+5+1)*3 + (9+1) + 8*4 = 153.
    critic_terms = {
        **actor_terms,
        "time_left": ObservationTermCfg(func=state.time_left),
        "joint_vel": ObservationTermCfg(func=state.joint_vel),
        "ee_velocity_base": ObservationTermCfg(
            func=state.velocity,
            params={"entity": "robot", "body": DEFAULT_ROBOT.ee_body, "reference": "robot"},
        ),
        "lemon_velocity_base": ObservationTermCfg(
            func=state.velocity, params={"entity": "lemon", "reference": "robot"}
        ),
        **{
            f"{entity}_friction": ObservationTermCfg(
                func=state.geom_friction, params={"entity": entity, "names": names}
            )
            for entity, names in (
                ("robot", DEFAULT_ROBOT.collision_geoms),
                ("lemon", LEMON_COLLISION_GEOMS),
                ("plate", PLATE_COLLISION_GEOMS),
                ("table", ("table_collision",)),
            )
        },
        "robot_mass": ObservationTermCfg(
            func=state.body_mass,
            params={"entity": "robot", "names": DEFAULT_ROBOT.rigid_bodies},
        ),
        "lemon_mass": ObservationTermCfg(
            func=state.body_mass, params={"entity": "lemon", "names": ("lemon",)}
        ),
        **{
            f"joint_{name}": ObservationTermCfg(func=getattr(DEFAULT_ROBOT, f"robot_joint_{name}"))
            for name in ("friction", "armature", "stiffness", "damping")
        },
    }
    bank_path = None if reset_bank is None else str(reset_bank)
    layout = workcell.load_layout()
    robot, action = DEFAULT_ROBOT.robot_and_action_cfg(layout=layout)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1 if play else 128,
            env_spacing=2.0,
            entities={
                "robot": robot,
                "table": workcell.table_cfg(layout),
                "ground": ground_cfg(-layout["table"]["height"]),
                "room": workcell.room_cfg(layout),
                "lemon": lemon_cfg(),
                "plate": plate_cfg(),
            },
            sensors=_contact_sensors(),
        ),
        observations={
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                history_length=5,
                enable_corruption=False,
                nan_policy="error",
                nan_check_per_term=False,
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                history_length=1,
                enable_corruption=False,
                nan_policy="error",
                nan_check_per_term=False,
            ),
        },
        actions={DEFAULT_ROBOT.action_name: action},
        events={
            "robot_material": EventTermCfg(
                func=dr.geom_friction,
                mode="startup",
                params={
                    "ranges": (0.2, 1.0),
                    "asset_cfg": SceneEntityCfg("robot", geom_names=".*_collision"),
                },
            ),
            "table_material": EventTermCfg(
                func=dr.geom_friction,
                mode="startup",
                params={
                    "ranges": (0.2, 0.5),
                    "asset_cfg": SceneEntityCfg("table", geom_names="table_collision"),
                },
            ),
            "robot_mass": EventTermCfg(
                func=DEFAULT_ROBOT.randomize_mass,
                mode="startup",
                params={"scale_range": (0.7, 1.3)},
            ),
            "reset_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "reset_from_bank": EventTermCfg(
                func=ResetFromBank,
                mode="reset",
                params={"path": bank_path, "reaching_only": play},
            ),
        },
        rewards={
            "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-1e-3),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-2),
            "joint_vel": RewardTermCfg(func=mdp.arm_velocity_l2, weight=-5e-2),
            "reaching": RewardTermCfg(func=mdp.reaching_reward, weight=0.2),
            "placement": RewardTermCfg(
                func=mdp.dense_goal_reward,
                weight=0.2,
                # Shaping reference only: the scanned plate floor is not planar.
                params={"goal_height": PLATE_CENTRE_FLOOR_Z - LEMON_BOTTOM_OFFSET},
            ),
            "success": RewardTermCfg(
                func=mdp.success_reward, weight=10.0 / DEFAULT_ROBOT.control_dt
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            # Rare solver divergence ends only that world; it is not a success.
            "physics_divergence": TerminationTermCfg(func=nan_detection),
            "success": TerminationTermCfg(
                func=mdp.StableSuccess,
                params={
                    "stable_seconds": STABLE_SECONDS,
                    "plate_contact_substeps": DEFAULT_ROBOT.decimation,
                },
            ),
        },
        sim=SimulationCfg(
            nconmax=256,
            njmax=1024,
            mujoco=MujocoCfg(
                timestep=DEFAULT_ROBOT.physics_dt,
                integrator="implicitfast",
                cone="elliptic",
                impratio=5.0,
                iterations=16,
                ls_iterations=20,
            ),
        ),
        viewer=ViewerConfig(
            lookat=(0.27, 0.02, 0.15),
            distance=1.15,
            azimuth=130.0,
            elevation=-25.0,
            origin_type=ViewerConfig.OriginType.WORLD,
        ),
        is_finite_horizon=True,
        decimation=DEFAULT_ROBOT.decimation,
        episode_length_s=EPISODE_SECONDS,
    )
    cfg.sim.mujoco.gravity = DEFAULT_ROBOT.gravity
    return cfg
