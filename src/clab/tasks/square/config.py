"""Compose Square's state-only task from common ARX5, observation and scene components."""

from __future__ import annotations

from dataclasses import asdict
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
from .props import NUT_COLLISION_GEOMS, PEG_COLLISION_GEOMS, nut_cfg, peg_cfg
from .resets import ResetFromBank
from .settings import EPISODE_SECONDS, AssemblyCriteria
from .variants import DEFAULT_ROBOT


def _contact_sensors():
    nut = ContactMatch(mode="body", pattern="nut", entity="nut")
    peg = ContactMatch(mode="body", pattern="peg", entity="peg")
    table = ContactMatch(mode="body", pattern="table", entity="table")
    robot = ContactMatch(mode="subtree", pattern=DEFAULT_ROBOT.root_body, entity="robot")
    pairs = [
        (f"{side}_finger_contact", ContactMatch(mode="body", pattern=body, entity="robot"), nut)
        for side, body in zip(("left", "right"), DEFAULT_ROBOT.finger_bodies, strict=True)
    ]
    pairs += [
        (
            "nut_arm_contact",
            nut,
            ContactMatch(
                mode="body",
                pattern="(" + "|".join(DEFAULT_ROBOT.rigid_bodies[:-2]) + ")",
                entity="robot",
            ),
        ),
        (
            "left_handle_contact",
            ContactMatch(mode="body", pattern=DEFAULT_ROBOT.finger_bodies[0], entity="robot"),
            ContactMatch(mode="geom", pattern="nut_collision_4", entity="nut"),
        ),
        (
            "right_handle_contact",
            ContactMatch(mode="body", pattern=DEFAULT_ROBOT.finger_bodies[1], entity="robot"),
            ContactMatch(mode="geom", pattern="nut_collision_4", entity="nut"),
        ),
        ("nut_peg_contact", nut, peg),
        ("nut_table_contact", nut, table),
        ("nut_robot_contact", nut, robot),
        ("robot_table_contact", robot, table),
        ("robot_peg_contact", robot, peg),
        ("robot_contact", robot, None),
    ]
    sensors = tuple(
        ContactSensorCfg(
            name=name, primary=a, secondary=b, fields=("found", "dist"), reduce="mindist"
        )
        for name, a, b in pairs
    )
    grasp_sensors = tuple(
        ContactSensorCfg(
            name=f"{side}_grasp_contact",
            primary=ContactMatch(mode="geom", pattern=NUT_COLLISION_GEOMS, entity="nut"),
            secondary=ContactMatch(mode="body", pattern=body, entity="robot"),
            fields=("found", "force", "pos"),
            reduce="maxforce",
            num_slots=4,
            secondary_policy="error",
        )
        for side, body in zip(("left", "right"), DEFAULT_ROBOT.finger_bodies, strict=True)
    )
    support = ContactSensorCfg(
        name="nut_base_support",
        primary=nut,
        secondary=ContactMatch(mode="geom", pattern="peg_collision_0", entity="peg"),
        fields=("found", "force"),
        reduce="netforce",  # Summed force is in world coordinates.
        secondary_policy="error",
    )
    return (*sensors, *grasp_sensors, support)


def make_env_cfg(*, play: bool = False, reset_bank: str | Path | None = None):
    """20 Hz, finite 16 s episodes; released, seated assembly for 1 s earns +10.

    Actor is six ordered current-state terms (39) with five history frames (195).
    Critic is current state, time, velocities and physical parameters (153).
    Neither observation contract implies compatibility with BoxInBin weights.
    """

    def relative(**params):
        return ObservationTermCfg(func=state.relative_pose, params=params)

    actor = {
        "nut_in_peg_frame": relative(target="nut", reference="peg"),
        "previous_actions": ObservationTermCfg(func=state.previous_actions),
        "joint_pos": ObservationTermCfg(func=state.joint_pos),
        "ee_in_base_frame": relative(
            target="robot", target_body=DEFAULT_ROBOT.ee_body, reference="robot"
        ),
        "nut_in_ee_frame": relative(
            target="nut", reference="robot", reference_body=DEFAULT_ROBOT.ee_body
        ),
        "peg_in_ee_frame": relative(
            target="peg", reference="robot", reference_body=DEFAULT_ROBOT.ee_body
        ),
    }
    critic = {
        **actor,
        "time_left": ObservationTermCfg(func=state.time_left),
        "joint_vel": ObservationTermCfg(func=state.joint_vel),
        "ee_velocity_base": ObservationTermCfg(
            func=state.velocity,
            params={"entity": "robot", "body": DEFAULT_ROBOT.ee_body, "reference": "robot"},
        ),
        "nut_velocity_base": ObservationTermCfg(
            func=state.velocity, params={"entity": "nut", "reference": "robot"}
        ),
        **{
            f"{entity}_friction": ObservationTermCfg(
                func=state.geom_friction, params={"entity": entity, "names": names}
            )
            for entity, names in (
                ("robot", DEFAULT_ROBOT.collision_geoms),
                ("nut", NUT_COLLISION_GEOMS),
                ("peg", PEG_COLLISION_GEOMS),
                ("table", ("table_collision",)),
            )
        },
        "robot_mass": ObservationTermCfg(
            func=state.body_mass, params={"entity": "robot", "names": DEFAULT_ROBOT.rigid_bodies}
        ),
        "nut_mass": ObservationTermCfg(
            func=state.body_mass, params={"entity": "nut", "names": ("nut",)}
        ),
        **{
            f"joint_{name}": ObservationTermCfg(func=getattr(DEFAULT_ROBOT, f"robot_joint_{name}"))
            for name in ("friction", "armature", "stiffness", "damping")
        },
    }
    path = None if reset_bank is None else str(reset_bank)
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
                "nut": nut_cfg(),
                "peg": peg_cfg(),
            },
            sensors=_contact_sensors(),
        ),
        observations={
            "actor": ObservationGroupCfg(
                terms=actor, history_length=5, enable_corruption=False, nan_policy="error"
            ),
            "critic": ObservationGroupCfg(
                terms=critic, history_length=1, enable_corruption=False, nan_policy="error"
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
                func=ResetFromBank, mode="reset", params={"path": path, "reaching_only": play}
            ),
        },
        rewards={
            "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-1e-4),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-3),
            "joint_vel": RewardTermCfg(func=mdp.arm_velocity_l2, weight=-1e-2),
            "reaching": RewardTermCfg(
                func=mdp.reaching_reward, weight=0.2, params={"distance_scale": 0.05}
            ),
            "handle_grasp": RewardTermCfg(func=mdp.handle_grasp_reward, weight=0.1),
            "alignment": RewardTermCfg(
                func=mdp.alignment_reward,
                weight=0.1,
                params={"distance_scale": 0.03, "tilt_scale": 0.25},
            ),
            "insertion": RewardTermCfg(
                func=mdp.insertion_reward, weight=0.2, params={"distance_scale": 0.03}
            ),
            "release": RewardTermCfg(func=mdp.release_reward, weight=0.1),
            "invalid_grasp": RewardTermCfg(
                func=mdp.invalid_grasp_penalty, weight=-2.0 / DEFAULT_ROBOT.control_dt
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
                func=mdp.StableSuccess, params={"criteria": asdict(AssemblyCriteria())}
            ),
            "invalid_grasp": TerminationTermCfg(func=mdp.invalid_grasp),
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
            lookat=(0.27, 0.02, 0.12),
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
