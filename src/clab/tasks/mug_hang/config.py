"""State-only MugHang: shared robot and controller, geometric hanging objective."""

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
from .props import MUG_COLLISION_GEOMS, TREE_COLLISION_GEOMS, mug_cfg, tree_cfg
from .resets import ResetFromBank
from .settings import EPISODE_SECONDS, STABLE_SECONDS
from .variants import DEFAULT_ROBOT


def _contact_sensors():
    mug = ContactMatch(mode="body", pattern="mug", entity="mug")
    tree = ContactMatch(mode="body", pattern="tree", entity="tree")
    table = ContactMatch(mode="body", pattern="table", entity="table")
    robot = ContactMatch(mode="subtree", pattern=DEFAULT_ROBOT.root_body, entity="robot")
    pairs = [
        (f"{side}_finger_contact", ContactMatch(mode="body", pattern=finger, entity="robot"), mug)
        for side, finger in zip(("left", "right"), DEFAULT_ROBOT.finger_bodies, strict=True)
    ]
    pairs += [
        ("mug_tree_contact", mug, tree),
        ("mug_table_contact", mug, table),
        ("mug_robot_contact", mug, robot),
        ("robot_contact", robot, None),
    ]
    return tuple(
        ContactSensorCfg(
            name=name, primary=a, secondary=b, fields=("found", "dist"), reduce="mindist"
        )
        for name, a, b in pairs
    )


def make_env_cfg(*, play=False, reset_bank: str | Path | None = None):
    """210-D actor, 240-D critic; finite 16 s horizon, actual +10 on 1 s success.

    Actor current 42-D state: existing mug/tree relative pose, previous actions,
    joints, EE/base, mug/EE, tree/EE plus tree-frame handle-to-hang-target xyz;
    each term has five oldest-to-newest frames. Existing relative rotations
    retain the handle's signed axis alignment. Critic
    adds time, velocities and explicit physical parameters (including 32 mug
    collision frictions), then the same xyz target vector.
    """

    def relative(**params):
        return ObservationTermCfg(func=state.relative_pose, params=params)

    actor = {
        "mug_in_tree_frame": relative(target="mug", reference="tree"),
        "previous_actions": ObservationTermCfg(func=state.previous_actions),
        "joint_pos": ObservationTermCfg(func=state.joint_pos),
        "ee_in_base_frame": relative(
            target="robot", target_body=DEFAULT_ROBOT.ee_body, reference="robot"
        ),
        "mug_in_ee_frame": relative(
            target="mug", reference="robot", reference_body=DEFAULT_ROBOT.ee_body
        ),
        "tree_in_ee_frame": relative(
            target="tree", reference="robot", reference_body=DEFAULT_ROBOT.ee_body
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
        "mug_velocity_base": ObservationTermCfg(
            func=state.velocity, params={"entity": "mug", "reference": "robot"}
        ),
        **{
            f"{entity}_friction": ObservationTermCfg(
                func=state.geom_friction, params={"entity": entity, "names": names}
            )
            for entity, names in (
                ("robot", DEFAULT_ROBOT.collision_geoms),
                ("mug", MUG_COLLISION_GEOMS),
                ("tree", TREE_COLLISION_GEOMS),
                ("table", ("table_collision",)),
            )
        },
        "robot_mass": ObservationTermCfg(
            func=state.body_mass, params={"entity": "robot", "names": DEFAULT_ROBOT.rigid_bodies}
        ),
        "mug_mass": ObservationTermCfg(
            func=state.body_mass, params={"entity": "mug", "names": ("mug",)}
        ),
        **{
            f"joint_{name}": ObservationTermCfg(func=getattr(DEFAULT_ROBOT, f"robot_joint_{name}"))
            for name in ("friction", "armature", "stiffness", "damping")
        },
    }
    # Append separately so every existing actor/critic field keeps its position.
    actor["handle_to_hang_target"] = ObservationTermCfg(func=mdp.handle_to_hang_target)
    critic["handle_to_hang_target"] = ObservationTermCfg(func=mdp.handle_to_hang_target)
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
                "mug": mug_cfg(),
                "tree": tree_cfg(),
            },
            sensors=_contact_sensors(),
        ),
        observations={
            "actor": ObservationGroupCfg(
                terms=actor,
                history_length=5,
                enable_corruption=False,
                nan_policy="error",
                nan_check_per_term=False,
            ),
            "critic": ObservationGroupCfg(
                terms=critic,
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
            "robot_mass": EventTermCfg(
                func=DEFAULT_ROBOT.randomize_mass,
                mode="startup",
                params={"scale_range": (0.7, 1.3)},
            ),
            "reset_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "reset_from_bank": EventTermCfg(
                func=ResetFromBank, mode="reset", params={"path": bank_path, "reaching_only": play}
            ),
        },
        rewards={
            "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-1e-4),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-3),
            "joint_vel": RewardTermCfg(func=mdp.arm_velocity_l2, weight=-1e-2),
            "reaching": RewardTermCfg(func=mdp.reaching_reward, weight=0.2),
            "threading": RewardTermCfg(func=mdp.threading_reward, weight=0.2),
            "success": RewardTermCfg(
                func=mdp.success_reward, weight=10.0 / DEFAULT_ROBOT.control_dt
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            # Rare solver divergence ends only that world; it is not a success.
            "physics_divergence": TerminationTermCfg(func=nan_detection),
            "success": TerminationTermCfg(
                func=mdp.StableSuccess, params={"stable_seconds": STABLE_SECONDS}
            ),
        },
        sim=SimulationCfg(
            nconmax=512,
            njmax=2048,
            # The 32-piece mug can make more than 64 matches against the table.
            contact_sensor_maxmatch=128,
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
            lookat=(0.25, 0.08, 0.16),
            distance=1.15,
            azimuth=130,
            elevation=-25,
            origin_type=ViewerConfig.OriginType.WORLD,
        ),
        is_finite_horizon=True,
        decimation=DEFAULT_ROBOT.decimation,
        episode_length_s=EPISODE_SECONDS,
    )
    cfg.sim.mujoco.gravity = DEFAULT_ROBOT.gravity
    return cfg
