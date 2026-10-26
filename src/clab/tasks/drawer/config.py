"""Independent nominal-physics Drawer composition.

No startup parameter randomization is enabled yet. Formal randomized training
needs the corresponding critic physical parameters and separate validation;
common controller settings alone do not establish experimental equivalence.
OSC control uses the shared 20 Hz policy preset. Surroundings and robot
placement use the common workcell layout.
The task starts with the passive drawer already open and requires grasp/lift,
release inside and contact-driven closure.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import nan_detection, time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from clab.observations import state
from clab.scenes import workcell
from clab.scenes.robosuite_table import ground_cfg

from . import mdp
from .props import drawer_cfg, pod_cfg
from .reset_data import bank_contract
from .resets import ResetFromBank
from .settings import EPISODE_SECONDS
from .variants import DEFAULT_ROBOT


def make_env_cfg(*, play=False, reset_bank=None):
    actor = {
        "pod_in_housing": ObservationTermCfg(
            func=state.relative_pose, params={"target": "pod", "reference": "fixture"}
        ),
        "previous_actions": ObservationTermCfg(func=state.previous_actions),
        "joint_pos": ObservationTermCfg(func=state.joint_pos),
        "ee_in_base": ObservationTermCfg(
            func=state.relative_pose,
            params={"target": "robot", "target_body": DEFAULT_ROBOT.ee_body, "reference": "robot"},
        ),
        "pod_in_ee": ObservationTermCfg(
            func=state.relative_pose,
            params={"target": "pod", "reference": "robot", "reference_body": DEFAULT_ROBOT.ee_body},
        ),
        "housing_in_ee": ObservationTermCfg(
            func=state.relative_pose,
            params={
                "target": "fixture",
                "reference": "robot",
                "reference_body": DEFAULT_ROBOT.ee_body,
            },
        ),
        "slide_state": ObservationTermCfg(func=mdp.slide_state),
        "process_state": ObservationTermCfg(func=mdp.process_observation),
    }
    critic = {
        **actor,
        "time_left": ObservationTermCfg(func=state.time_left),
        "joint_vel": ObservationTermCfg(func=state.joint_vel),
        "ee_velocity": ObservationTermCfg(
            func=state.velocity,
            params={"entity": "robot", "body": DEFAULT_ROBOT.ee_body, "reference": "robot"},
        ),
        "pod_velocity": ObservationTermCfg(
            func=state.velocity, params={"entity": "pod", "reference": "robot"}
        ),
    }
    contacts = []
    for side, body in zip(("left", "right"), DEFAULT_ROBOT.finger_bodies, strict=True):
        for target, entity, other_body in (("pod", "pod", "pod"), ("drawer", "fixture", "drawer")):
            contacts.append(
                ContactSensorCfg(
                    name=f"{side}_{target}_contact",
                    primary=ContactMatch(mode="body", entity="robot", pattern=body),
                    secondary=ContactMatch(mode="body", entity=entity, pattern=other_body),
                    fields=("found", "dist"),
                    reduce="mindist",
                )
            )
    for name, primary, secondary in (
        (
            "robot_contact",
            ContactMatch(mode="subtree", entity="robot", pattern=DEFAULT_ROBOT.root_body),
            None,
        ),
        (
            "pod_table_contact",
            ContactMatch(mode="body", entity="pod", pattern="pod"),
            ContactMatch(mode="body", entity="table", pattern="table"),
        ),
        (
            "pod_fixture_contact",
            ContactMatch(mode="body", entity="pod", pattern="pod"),
            ContactMatch(mode="subtree", entity="fixture", pattern="housing"),
        ),
    ):
        contacts.append(
            ContactSensorCfg(
                name=name,
                primary=primary,
                secondary=secondary,
                fields=("found", "dist"),
                reduce="mindist",
            )
        )
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
                "fixture": drawer_cfg(),
                "pod": pod_cfg(),
            },
            sensors=tuple(contacts),
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
            "reset_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "reset_from_bank": EventTermCfg(
                func=ResetFromBank,
                mode="reset",
                params={
                    "path": None if reset_bank is None else str(reset_bank),
                    "reaching_only": play,
                },
            ),
        },
        rewards={
            "reaching": RewardTermCfg(func=mdp.staged_reaching, weight=0.2),
            "progress": RewardTermCfg(func=mdp.staged_progress, weight=0.2),
            "action_magnitude": RewardTermCfg(func=mdp.action_cost, weight=-1e-4),
            "action_rate": RewardTermCfg(func=mdp.action_rate_cost, weight=-1e-3),
            "joint_vel": RewardTermCfg(func=mdp.arm_velocity_cost, weight=-1e-2),
            "success": RewardTermCfg(func=mdp.success_reward, weight=10 / DEFAULT_ROBOT.control_dt),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            # Rare solver divergence ends only that world; it is not a success.
            "physics_divergence": TerminationTermCfg(func=nan_detection),
            "success": TerminationTermCfg(
                func=mdp.SequenceSuccess, params={"task_contract": bank_contract()}
            ),
            "process_failure": TerminationTermCfg(func=mdp.process_failure),
        },
        sim=SimulationCfg(
            # Pushing the drawer with compound fingertips needed more than 1024
            # constraint rows in one world once the policy acted.
            nconmax=512,
            njmax=2048,
            # Compound fingertips against the multi-piece housing can match many
            # contacts in one sensor; 327 were observed once the policy acted.
            # This is sensor storage, not a change to contact forces/physics.
            contact_sensor_maxmatch=1024,
            mujoco=MujocoCfg(
                timestep=DEFAULT_ROBOT.physics_dt,
                integrator="implicitfast",
                cone="elliptic",
                impratio=5.0,
                iterations=16,
                ls_iterations=20,
            ),
        ),
        viewer=ViewerConfig(lookat=(0.27, 0.02, 0.1), distance=1.2, azimuth=130, elevation=-25),
        is_finite_horizon=True,
        decimation=DEFAULT_ROBOT.decimation,
        episode_length_s=EPISODE_SECONDS,
    )
    cfg.sim.mujoco.gravity = DEFAULT_ROBOT.gravity
    return cfg
