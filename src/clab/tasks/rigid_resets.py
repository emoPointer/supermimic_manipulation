"""Shared mechanics for tasks with one free prop and one fixed mocap target.

Task modules declare names, schema, robot binding and category semantics. This
module validates representation, writes environment-local state and counts completed
episodes; it does not decide physical acceptance, reachability or task success.
Articulated fixtures need their own extra-state contract, not this rigid schema.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _indices(env, env_ids):
    return (
        torch.arange(env.num_envs, device=env.device, dtype=torch.int)
        if env_ids is None
        else env_ids
    )


@dataclass(frozen=True)
class RigidResetSpec:
    task: str
    schema: str
    robot: Any
    reset_types: tuple[str, ...]
    object_name: str
    target_name: str
    pose_frame: str = "robot_base_local"

    @property
    def fields(self) -> tuple[str, ...]:
        return (
            "joint_pos",
            "joint_vel",
            f"{self.object_name}_pose",
            f"{self.object_name}_vel",
            f"{self.target_name}_pose",
        )

    @property
    def widths(self) -> tuple[int, ...]:
        return (len(self.robot.joints), len(self.robot.joints), 7, 6, 7)

    def contract(self) -> dict:
        return {
            "task": self.task,
            "robot": self.robot.name,
            "joint_order": list(self.robot.joints),
            "fields": list(self.fields),
            "frame": self.pose_frame,
            "pose": "xyz_wxyz",
            "joint_units": ["rad"] * len(self.robot.arm_joints)
            + ["m"] * len(self.robot.finger_joints),
            "joint_velocity_units": ["rad/s"] * len(self.robot.arm_joints)
            + ["m/s"] * len(self.robot.finger_joints),
            "position_unit": "m",
            f"{self.object_name}_velocity": "root_origin_local_linear_xyz_angular_xyz",
            f"{self.object_name}_velocity_units": ["m/s"] * 3 + ["rad/s"] * 3,
            self.target_name: "fixed_mocap_pose_constant_within_episode",
            "categories": list(self.reset_types),
        }


def validate_bank(bank: object, spec: RigidResetSpec, model=None) -> dict:
    """Check the explicit tensor contract, not a physics/reachability guarantee."""
    if not isinstance(bank, dict) or bank.get("schema") != spec.schema:
        raise ValueError(f"Unsupported {spec.task} reset bank schema")
    if bank.get("contract") != spec.contract():
        raise ValueError(f"Incompatible {spec.task} reset bank contract: {bank.get('contract')}")
    provenance = bank.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("method"), str)
        or not provenance["method"]
        or not isinstance(bank.get("split"), str)
        or not bank["split"]
    ):
        raise ValueError(f"{spec.task} reset bank requires explicit provenance.method and split")
    states = bank.get("states")
    if not isinstance(states, dict) or tuple(states) != spec.reset_types:
        raise ValueError(f"{spec.task} reset bank requires ordered categories: {spec.reset_types}")
    for kind, rows in states.items():
        if not isinstance(rows, dict) or tuple(rows) != spec.fields:
            raise ValueError(f"{kind}: expected ordered fields {spec.fields}")
        count = None
        for field, width in zip(spec.fields, spec.widths, strict=True):
            value = rows[field]
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f"{kind}.{field}: expected [N, {width}] tensor")
            if (
                value.device.type != "cpu"
                or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{kind}.{field}: expected finite CPU float32 tensor")
            if count is None:
                count = len(value)
            elif len(value) != count:
                raise ValueError(f"{kind}: inconsistent row count")
        if not count:
            raise ValueError(f"{kind}: empty reset category")
        for field in (f"{spec.object_name}_pose", f"{spec.target_name}_pose"):
            norm = torch.linalg.vector_norm(rows[field][:, 3:], dim=-1)
            if not torch.allclose(norm, torch.ones_like(norm), atol=1e-3, rtol=0):
                raise ValueError(f"{kind}.{field}: quaternions must be normalized wxyz")
    if model is not None:
        spec.robot.validate_model(model)
        spec.robot.validate_states(model, states)
    return states


def bank_metadata(bank: dict, spec: RigidResetSpec) -> dict:
    states = validate_bank(bank, spec)
    return {
        "contract": deepcopy(bank["contract"]),
        "provenance": deepcopy(bank["provenance"]),
        "split": bank["split"],
        "counts": {kind: len(rows["joint_pos"]) for kind, rows in states.items()},
    }


def snapshot(env, spec: RigidResetSpec, env_ids: torch.Tensor | None = None) -> dict:
    """Copy physical states in world axes, subtracting only the environment origin."""
    indices = _indices(env, env_ids)
    robot = env.scene["robot"]
    obj, target = env.scene[spec.object_name], env.scene[spec.target_name]
    object_pose = obj.data.root_link_pose_w[indices].clone()
    target_pose = target.data.root_link_pose_w[indices].clone()
    object_pose[:, :3] -= env.scene.env_origins[indices]
    target_pose[:, :3] -= env.scene.env_origins[indices]
    return dict(
        zip(
            spec.fields,
            (
                robot.data.joint_pos[indices].clone(),
                robot.data.joint_vel[indices].clone(),
                object_pose,
                torch.cat(
                    (obj.data.root_link_lin_vel_w[indices], obj.data.root_link_ang_vel_w[indices]),
                    dim=-1,
                ).clone(),
                target_pose,
            ),
            strict=True,
        )
    )


def reloaded_hold_mask(
    env,
    rows,
    eligible,
    spec,
    *,
    limits,
    finger_upper_tolerance,
    action_cfg,
    gripper_command,
    seed,
    steps,
    required_steps,
    acceptance,
    safety,
):
    """Replay fresh candidate bytes through normal reset before permitting export.

    Task callbacks own all physical predicates. Only a copy's approved finger
    positions are projected; source rows remain untouched. Rejected worlds are
    padded with the first candidate solely to keep a fixed simulation batch;
    the returned mask can never admit those padding rows. Export the input rows
    with the same finger projection, NOT the ending state of this replay.
    """
    from clab.controllers.pose import pose_target_action
    from clab.observations.state import observation_pose

    if steps < required_steps or required_steps < 1:
        raise ValueError("Replay needs a positive continuous acceptance window")
    selected = eligible.nonzero()[:, 0]
    if not len(selected):
        return eligible.clone()
    replay_rows = {name: value.clone() for name, value in rows.items()}
    for value in replay_rows.values():
        value[~eligible] = value[selected[0]].clone()
    projected = {"states": {"candidates": replay_rows}, "provenance": {}}
    spec.robot.project_reset_fingers(projected, limits, finger_upper_tolerance)
    env.reset(seed=seed)
    write_states(env, replay_rows, spec)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    initial = snapshot(env, spec)
    target = tuple(v.clone() for v in observation_pose(env, "robot", spec.robot.ee_body))
    good = eligible.clone() & safety(env)
    stable = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    for _ in range(steps):
        action = pose_target_action(
            *observation_pose(env, "robot", spec.robot.ee_body),
            *target,
            action_cfg=action_cfg,
            gripper_command=gripper_command,
        )
        _, _, terminal, timeout, _ = env.step(action)
        if bool((terminal | timeout).any()) or bool((env.sim.data.overflow != 0).any()):
            raise RuntimeError("Reset replay encountered episode end or capacity overflow")
        if not bool(torch.isfinite(env.sim.data.qpos).all()) or not bool(
            torch.isfinite(env.sim.data.qvel).all()
        ):
            raise RuntimeError("Reset replay encountered non-finite physical state")
        good &= safety(env)
        stable = torch.where(acceptance(env, initial), stable + 1, 0)
    return good & (stable >= required_steps)


def write_states(env, rows: dict, spec: RigidResetSpec, env_ids=None) -> None:
    """Apply validated local states without filtering or clamping values."""
    indices = _indices(env, env_ids)
    count = len(indices)
    if tuple(rows) != spec.fields:
        raise ValueError(f"{spec.task} state fields must be ordered {spec.fields}")
    for field, width in zip(spec.fields, spec.widths, strict=True):
        if rows[field].shape != (count, width):
            raise ValueError(f"{field}: expected {(count, width)}, got {rows[field].shape}")
    robot = env.scene["robot"]
    robot.write_joint_state_to_sim(rows["joint_pos"], rows["joint_vel"], env_ids=indices)
    object_pose = rows[f"{spec.object_name}_pose"].clone()
    target_pose = rows[f"{spec.target_name}_pose"].clone()
    object_pose[:, :3] += env.scene.env_origins[indices]
    target_pose[:, :3] += env.scene.env_origins[indices]
    env.scene[spec.object_name].write_root_state_to_sim(
        torch.cat((object_pose, rows[f"{spec.object_name}_vel"]), dim=-1), env_ids=indices
    )
    env.scene[spec.target_name].write_mocap_pose_to_sim(target_pose, env_ids=indices)
    robot.set_joint_velocity_target(
        torch.zeros((count, len(spec.robot.joints)), device=env.device), env_ids=indices
    )


def record_finished_episodes(env, env_ids: torch.Tensor, task_id: torch.Tensor, count: int) -> None:
    """Capture explicit success outcomes before automatic reset clears them."""
    if env.common_step_counter == 0:
        return
    completed = env_ids[env.reset_buf[env_ids]]
    if len(completed) == 0:
        return
    success = env.termination_manager.get_term("success")[completed]
    previous_types = task_id[completed]
    outcomes = torch.zeros_like(task_id, dtype=torch.bool)
    outcomes[completed] = success
    env.extras["episode_success"] = outcomes
    log = env.extras["log"]
    log["Episode_Metrics/success_rate"] = success.float().mean()
    for type_id in range(count):
        belongs = previous_types == type_id
        if bool(belongs.any()):
            log[f"Episode_Metrics/type_{type_id}_success_rate"] = success[belongs].float().mean()
            log[f"Episode_Metrics/type_{type_id}_episodes"] = belongs.sum()
            log[f"Episode_Metrics/type_{type_id}_successes"] = success[belongs].sum()


class RigidResetFromBank:
    """Class-based mjlab event; tasks bind `spec` and own data semantics.

    mjlab instantiates callable classes directly; inheriting ManagerTermBase is
    unnecessary here. Keeping this representation module independent of mjlab
    imports also avoids cycles during mjlab's eager task-plugin discovery.
    """

    spec: RigidResetSpec

    def __init__(self, cfg, env) -> None:
        path = cfg.params.get("path")
        if not path:
            raise ValueError(f"Provide a {self.spec.task} reset bank path")
        bank_path = Path(path).expanduser()
        if not bank_path.is_file():
            raise FileNotFoundError(f"{self.spec.task} reset bank not found: {bank_path}")
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        states = validate_bank(bank, self.spec, model=env.sim.mj_model)
        self.metadata = bank_metadata(bank, self.spec)
        self.states = {
            kind: {field: value.to(env.device) for field, value in rows.items()}
            for kind, rows in states.items()
        }
        self.task_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.type_override: int | None = None
        self._validated = False

    def __call__(self, env, env_ids, path: str | None, reaching_only: bool = False) -> None:
        del path
        if not self._validated:
            self.spec.robot.validate_env(env)
            self._validated = True
        indices = _indices(env, env_ids)
        count = len(indices)
        record_finished_episodes(env, indices, self.task_id, len(self.spec.reset_types))
        if self.type_override is not None:
            if self.type_override not in range(len(self.spec.reset_types)):
                raise ValueError(f"Invalid {self.spec.task} reset category: {self.type_override}")
            kinds = torch.full((count,), self.type_override, dtype=torch.long, device=env.device)
        elif reaching_only:
            kinds = torch.zeros(count, dtype=torch.long, device=env.device)
        else:
            kinds = torch.randint(len(self.spec.reset_types), (count,), device=env.device)
        self.task_id[indices] = kinds
        for type_id, kind in enumerate(self.spec.reset_types):
            selected = indices[kinds == type_id]
            if len(selected):
                rows = self.states[kind]
                sample = torch.randint(len(rows["joint_pos"]), (len(selected),), device=env.device)
                write_states(
                    env,
                    {field: value[sample] for field, value in rows.items()},
                    self.spec,
                    selected,
                )
