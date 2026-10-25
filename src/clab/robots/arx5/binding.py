"""ARX5 robot binding; task variants subclass it and declare their observation contract."""

import numpy as np
import torch

from . import description as robot
from . import parameters
from .control import DECIMATION, PHYSICS_DT, osc_action_cfg, robot_and_action_cfg


class Arx5Binding:
    """Robot-owned names, physical parameter access, and representation checks.

    Task subclasses supply ``task_id``, ``actor_dim``, ``critic_dim``,
    ``experiment_name`` and ``wandb_project``. No task or algorithm is imported.
    """

    name = "arx5"
    joints = robot.JOINTS
    arm_joints = robot.ARM_JOINTS
    finger_joints = robot.FINGER_JOINTS
    finger_actuators = robot.FINGER_ACTUATORS
    root_body = robot.ROOT_BODY
    ee_body = robot.EE_BODY
    finger_bodies = robot.FINGER_BODIES
    rigid_bodies = robot.RIGID_BODIES
    collision_geoms = robot.COLLISION_GEOMS
    grasp_site = robot.GRASP_SITE
    grasp_offset = robot.GRASP_OFFSET
    closing_axis = robot.GRIPPER_CLOSING_AXIS
    home = robot.HOME
    action_dim = 7
    # Simulation timing and gravity shared by every task.
    physics_dt = PHYSICS_DT
    decimation = DECIMATION
    gravity = (0.0, 0.0, -9.81)
    # Explicitly approved reset-data projection; not a physical/control limit.
    reset_finger_upper_tolerance = 0.0004
    action_name = "arx5_osc"
    finger_collision_geometry = "source_mesh_convex_hull"
    actor_dim: int
    critic_dim: int
    action_cfg = staticmethod(osc_action_cfg)
    robot_joint_friction = staticmethod(parameters.robot_joint_friction)
    robot_joint_armature = staticmethod(parameters.robot_joint_armature)
    robot_joint_stiffness = staticmethod(parameters.robot_joint_stiffness)
    robot_joint_damping = staticmethod(parameters.robot_joint_damping)

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation

    @classmethod
    def robot_and_action_cfg(cls, layout=None, *, action_cfg=None):
        return robot_and_action_cfg(
            layout,
            action_cfg=action_cfg,
            finger_collision_geometry=cls.finger_collision_geometry,
        )

    @classmethod
    def finger_collision_contract(cls):
        from .model import finger_collision_contract

        return finger_collision_contract(cls.finger_collision_geometry)

    @staticmethod
    def gripper_contract(model, *, entity_name="robot"):
        from .model import gripper_actuator_contract

        return gripper_actuator_contract(model, entity_name=entity_name)

    @staticmethod
    def gripper_contact_contract(model, *, entity_name="robot"):
        from .model import gripper_contact_contract

        return gripper_contact_contract(model, entity_name=entity_name)

    @staticmethod
    def configure_gripper_friction(scene_cfg, coefficient, *, entity_name="robot"):
        from .model import configure_gripper_friction

        configure_gripper_friction(scene_cfg, coefficient, entity_name=entity_name)

    @property
    def randomize_mass(self):
        # Return the decorated function itself: managers inspect its model-field
        # requirements, so wrapping only its invocation would lose metadata.
        from .randomization import robot_mass

        return robot_mass

    def validate_model(self, model) -> None:
        actual = tuple(model.joint(index).name for index in range(len(self.joints)))
        if actual != tuple(f"robot/{name}" for name in self.joints):
            raise ValueError(f"ARX5 joint order mismatch: {actual}")
        for name in self.rigid_bodies:
            model.body(f"robot/{name}")
        for name in self.collision_geoms:
            model.geom(f"robot/{name}")
        contract = self.finger_collision_contract()
        for body in self.finger_bodies:
            body_id = model.body(f"robot/{body}").id
            actual = {
                model.geom(index).name
                for index in range(model.ngeom)
                if model.geom_bodyid[index] == body_id
                and (model.geom_contype[index] or model.geom_conaffinity[index])
            }
            count = (
                len(contract["decomposition"][body]["parts"])
                if contract["mode"] == "convex_decomposition"
                else 1
            )
            expected = {f"robot/{body}_collision"} | {
                f"robot/{body}_part_{index:03d}_collision" for index in range(1, count)
            }
            if actual != expected:
                raise ValueError(
                    f"ARX5 {body} collision-piece mismatch: {actual}, expected {expected}"
                )

    def validate_states(self, model, states) -> None:
        limits = torch.from_numpy(
            np.stack([model.joint(f"robot/{name}").range for name in self.joints])
        )
        for kind, rows in states.items():
            q = rows["joint_pos"]
            if bool(((q < limits[:, 0] - 1e-5) | (q > limits[:, 1] + 1e-5)).any()):
                raise ValueError(f"{kind}: reset joint positions exceed current ARX5 limits")

    def validate_env(self, env) -> None:
        dimensions = env.observation_manager.group_obs_dim
        expected = {"actor": (self.actor_dim,), "critic": (self.critic_dim,)}
        if any(dimensions.get(key) != shape for key, shape in expected.items()):
            raise ValueError(f"Observation contract mismatch: {dimensions}, expected {expected}")
        if env.action_manager.total_action_dim != self.action_dim:
            raise ValueError("ARX5 requires six relative pose commands and one binary gripper")

    def reset_joint_position_mask(self, positions, limits, finger_upper_tolerance):
        """Keep arm bounds strict; allow bounded finger upper-stop overshoot."""
        if not np.isfinite(finger_upper_tolerance) or finger_upper_tolerance < 0:
            raise ValueError("Finger overshoot tolerance must be finite and nonnegative")
        upper = limits[:, 1].clone() + 1e-5
        upper[len(self.arm_joints) :] = limits[len(self.arm_joints) :, 1] + finger_upper_tolerance
        return (
            torch.isfinite(positions).all(-1)
            & (positions >= limits[:, 0] - 1e-5).all(-1)
            & (positions <= upper).all(-1)
        )

    def project_reset_fingers(self, bank, limits, finger_upper_tolerance):
        """Project exported data after physical acceptance; never alter simulation."""
        indices = slice(len(self.arm_joints), len(self.joints))
        statistics = {}
        for kind, rows in bank["states"].items():
            q = rows["joint_pos"]
            if not bool(self.reset_joint_position_mask(q, limits, finger_upper_tolerance).all()):
                raise ValueError(f"{kind}: reset exceeds arm bounds or permitted finger overshoot")
            before = q[:, indices].clone()
            after = before.clamp(limits[indices, 0], limits[indices, 1])
            q[:, indices] = after
            delta = (after - before).abs()
            statistics[kind] = {
                "changed_rows": int((delta != 0).any(-1).sum()),
                "max_correction_m": float(delta.max()),
            }
        bank["provenance"]["finger_position_projection"] = {
            "joints": list(self.finger_joints),
            "bounds_m": limits[indices].tolist(),
            "upper_overshoot_tolerance_m": finger_upper_tolerance,
            "statistics": statistics,
            "simulation_state_modified": False,
            "projected_contact_hold_revalidated": False,
        }
