"""Compose the BoxInBin task with its explicitly selected robot binding."""

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

from clab.scenes import workcell
from clab.scenes.robosuite_table import ground_cfg

from . import mdp
from .props import bin_cfg, box_cfg
from .resets import ResetFromBank
from .settings import (
    EPISODE_SECONDS,
    STABLE_SECONDS,
    SUPPORT_WEIGHT_FRACTION,
)
from .variants import DEFAULT_ROBOT


def _contact_sensors() -> tuple[ContactSensorCfg, ...]:
    object_body = ContactMatch(mode="body", pattern="box", entity="box")
    return (
        ContactSensorCfg(
            name="left_finger_contact",
            primary=ContactMatch(
                mode="body", pattern=DEFAULT_ROBOT.finger_bodies[0], entity="robot"
            ),
            secondary=object_body,
            fields=("found",),
            reduce="maxforce",
        ),
        ContactSensorCfg(
            name="right_finger_contact",
            primary=ContactMatch(
                mode="body", pattern=DEFAULT_ROBOT.finger_bodies[1], entity="robot"
            ),
            secondary=object_body,
            fields=("found",),
            reduce="maxforce",
        ),
        ContactSensorCfg(
            name="box_bin_contact",
            primary=object_body,
            secondary=ContactMatch(mode="subtree", pattern="bin", entity="bin"),
            fields=("found", "force"),
            reduce="netforce",
            global_frame=True,
        ),
    )


def make_env_cfg(
    *,
    play: bool = False,
    reset_bank: str | Path | None = None,
) -> ManagerBasedRlEnvCfg:
    """OSC control, finite 16 s episodes and released 2 s success.

    Actor/critic dimensions are 195/164. Actor history uses five 20 Hz samples
    spanning 0.2 s from the oldest observation to the current one.
    """
    actor_terms = {
        # Observation term order is part of the checkpoint contract.
        "insertive_asset_in_receptive_asset_frame": ObservationTermCfg(
            func=mdp.relative_pose, params={"target": "box", "reference": "bin"}
        ),
        "prev_actions": ObservationTermCfg(func=mdp.previous_actions),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos),
        "end_effector_pose": ObservationTermCfg(
            func=mdp.relative_pose, params={"target": "ee", "reference": "robot"}
        ),
        "insertive_asset_pose": ObservationTermCfg(
            func=mdp.relative_pose, params={"target": "box", "reference": "ee"}
        ),
        "receptive_asset_pose": ObservationTermCfg(
            func=mdp.relative_pose, params={"target": "bin", "reference": "ee"}
        ),
    }
    critic_terms = {
        **actor_terms,
        "time_left": ObservationTermCfg(func=mdp.time_left),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel),
        "end_effector_vel_lin_ang_b": ObservationTermCfg(func=mdp.end_effector_vel_lin_ang_b),
        "robot_material_properties": ObservationTermCfg(
            func=mdp.material_properties, params={"entity": "robot"}
        ),
        "insertive_object_material_properties": ObservationTermCfg(
            func=mdp.material_properties, params={"entity": "box"}
        ),
        "receptive_object_material_properties": ObservationTermCfg(
            func=mdp.material_properties, params={"entity": "bin"}
        ),
        "table_material_properties": ObservationTermCfg(
            func=mdp.material_properties, params={"entity": "table"}
        ),
        "robot_mass": ObservationTermCfg(func=mdp.rigid_body_mass, params={"entity": "robot"}),
        "insertive_object_mass": ObservationTermCfg(
            func=mdp.rigid_body_mass, params={"entity": "box"}
        ),
        "receptive_object_mass": ObservationTermCfg(
            func=mdp.rigid_body_mass, params={"entity": "bin"}
        ),
        "table_mass": ObservationTermCfg(func=mdp.rigid_body_mass, params={"entity": "table"}),
        "robot_joint_friction": ObservationTermCfg(func=mdp.robot_joint_friction),
        "robot_joint_armature": ObservationTermCfg(func=mdp.robot_joint_armature),
        "robot_joint_stiffness": ObservationTermCfg(func=mdp.robot_joint_stiffness),
        "robot_joint_damping": ObservationTermCfg(func=mdp.robot_joint_damping),
    }
    bank_path = None if reset_bank is None else str(reset_bank)
    layout = workcell.load_layout()
    robot, action = DEFAULT_ROBOT.robot_and_action_cfg(layout=layout)
    entities = {
        "robot": robot,
        "table": workcell.table_cfg(layout),
        "ground": ground_cfg(-layout["table"]["height"]),
        "box": box_cfg(),
        "bin": bin_cfg(),
    }
    entities["room"] = workcell.room_cfg(layout)
    # Resting visual examples on the new tabletop, in front of the robot.
    # These are defaults only; reset-bank records are never rewritten.
    entities["box"].init_state.pos = (0.28, -0.10, entities["box"].init_state.pos[2])
    entities["bin"].init_state.pos = (0.35, 0.13, entities["bin"].init_state.pos[2])
    for name in ("box", "bin"):
        entities[name].init_state.rot = (0.004852061427585483, 0.0, 0.0, 0.9999882286806695)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1 if play else 128,
            env_spacing=2.0,
            entities=entities,
            sensors=_contact_sensors(),
        ),
        observations={
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                history_length=5,
                enable_corruption=not play,
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
        # Use identical control parameters in training and evaluation.
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
            "box_material": EventTermCfg(
                func=dr.geom_friction,
                mode="startup",
                params={
                    "ranges": (0.9, 1.9),
                    "asset_cfg": SceneEntityCfg("box", geom_names="box_collision"),
                },
            ),
            "bin_material": EventTermCfg(
                func=dr.geom_friction,
                mode="startup",
                params={
                    "ranges": (0.15, 0.5),
                    "asset_cfg": SceneEntityCfg("bin", geom_names=".*"),
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
                params={
                    "path": bank_path,
                    "reaching_only": play,
                },
            ),
        },
        rewards={
            "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-1e-4),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-3),
            "joint_vel": RewardTermCfg(func=mdp.arm_velocity_l2, weight=-1e-2),
            "reaching": RewardTermCfg(func=mdp.reaching_reward, weight=0.2),
            "placement": RewardTermCfg(func=mdp.dense_goal_reward, weight=0.2),
            "success": RewardTermCfg(
                func=mdp.success_reward,
                weight=10.0 / DEFAULT_ROBOT.control_dt,
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            # Rare solver divergence ends only that world; it is not a success.
            "physics_divergence": TerminationTermCfg(func=nan_detection),
            "success": TerminationTermCfg(
                func=mdp.StableSuccess,
                params={
                    "support_weight_fraction": SUPPORT_WEIGHT_FRACTION,
                    "stable_seconds": STABLE_SECONDS,
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
    DEFAULT_ROBOT.configure_gripper_friction(cfg.scene, 2.0)
    return cfg
