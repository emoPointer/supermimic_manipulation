"""Read effective robot joint parameters from the runtime physical model."""

import torch

from .description import FINGER_ACTUATORS, JOINTS


def _robot_joint_ids(env) -> tuple[torch.Tensor, torch.Tensor]:
    joints = [env.sim.mj_model.joint(f"robot/{name}") for name in JOINTS]
    joint_ids = torch.tensor([joint.id for joint in joints], device=env.device)
    dof_ids = torch.tensor([joint.dofadr[0] for joint in joints], device=env.device)
    return joint_ids, dof_ids


def robot_joint_friction(env) -> torch.Tensor:
    _, dof_ids = _robot_joint_ids(env)
    return env.sim.model.dof_frictionloss[:, dof_ids]


def robot_joint_armature(env) -> torch.Tensor:
    _, dof_ids = _robot_joint_ids(env)
    return env.sim.model.dof_armature[:, dof_ids]


def robot_joint_stiffness(env) -> torch.Tensor:
    joint_ids, _ = _robot_joint_ids(env)
    value = env.sim.model.jnt_stiffness[:, joint_ids].clone()
    actuator_ids = [env.sim.mj_model.actuator(f"robot/{name}").id for name in FINGER_ACTUATORS]
    value[:, -2:] += env.sim.model.actuator_gainprm[:, actuator_ids, 0]
    return value


def robot_joint_damping(env) -> torch.Tensor:
    _, dof_ids = _robot_joint_ids(env)
    value = env.sim.model.dof_damping[:, dof_ids].clone()
    actuator_ids = [env.sim.mj_model.actuator(f"robot/{name}").id for name in FINGER_ACTUATORS]
    value[:, -2:] -= env.sim.model.actuator_biasprm[:, actuator_ids, 2]
    return value
