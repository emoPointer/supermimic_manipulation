"""Batched six-joint MuJoCo-Warp OSC for state-control tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import apply_delta_pose, compute_pose_error

from clab.cuda_graphs import graph_break

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class OscActionCfg(ActionTermCfg):
    """Relative end-effector pose (6) plus binary finger position target (1)."""

    arm_joints: tuple[str, ...]
    finger_joints: tuple[str, ...]
    ee_body: str
    closed_position: float = 0.002
    open_position: float = 0.044
    effort_limit: float = 50.0
    inertia_solver: Literal["inverse", "pinv"] = "inverse"
    inertia_rcond: float = 1e-10
    uncouple_pos_ori: bool = False

    position_scale: float = 0.02
    orientation_scale: float = 0.2
    motion_stiffness: tuple[float, ...] = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
    motion_damping_ratio: tuple[float, ...] = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)

    input_limit: float | None = None
    # Bounds on each relative pose command divided by the control period.
    # Like common delta-pose bounds, these are NOT actual velocity caps.
    linear_velocity_limit: float | None = None
    angular_velocity_limit: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_limit",
            "linear_velocity_limit",
            "angular_velocity_limit",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"OSC {name} must be positive and finite, or None")

    def build(self, env: ManagerBasedRlEnv) -> OscAction:
        return OscAction(self, env)


def limit_vector_norm(value: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Bound a vector's length, preserving direction and all in-range values."""
    if limit is None:
        return value
    scale = (
        limit / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)
    ).clamp_max(1.0)
    return value * scale


def _eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.aten._linalg_eigh(matrix, "L", True)


def _pinv_hermitian(matrix: torch.Tensor, rtol: float) -> torch.Tensor:
    """``torch.linalg.pinv(matrix, rtol=rtol, hermitian=True)`` without host synchronizations.

    The same eigendecomposition and thresholding as torch 2.9's linalg_pinv, bitwise;
    ``aten._linalg_eigh`` skips ``linalg.eigh``'s host-side convergence check, which
    synchronized twice per substep. cuSOLVER's batched eigendecomposition cannot be
    captured on this driver, so a substep graph runs it eagerly between segments.
    """
    eigenvalues, vectors = graph_break(_eigh, matrix)
    magnitude = eigenvalues.abs()
    zero = matrix.new_zeros(())
    tolerance = torch.maximum(
        zero.unsqueeze(-1),
        matrix.new_full((), rtol).unsqueeze(-1) * magnitude.amax(-1, keepdim=True),
    )
    inverse = torch.where(magnitude > tolerance, eigenvalues.reciprocal(), zero)
    return torch.matmul(vectors * inverse.unsqueeze(-2), vectors.mH)


def operational_space_torque(
    jacobian: torch.Tensor,
    mass: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    solver: Literal["inverse", "pinv"],
    rcond: float,
    uncouple_pos_ori: bool = False,
) -> torch.Tensor:
    """Compute Jᵀ Λ a; the explicit pinv profile projects near-singular directions.

    The historical inverse profile retains its float32 arithmetic. Pinv builds
    the symmetric inverse task inertia in float64 using a mass solve, discards
    eigenvalues below rcond times the largest magnitude, and returns the input
    dtype. Task axes retain the existing metres/radians convention. No invalid
    state or failed decomposition is replaced, and no solver fallback is used.
    With uncouple_pos_ori, compute independent position/orientation task inertia
    blocks, as in robosuite's default OSC, then sum their joint torques. This is
    a different control profile, not equivalent to the full six-axis inertia.
    """
    if uncouple_pos_ori:
        return sum(
            operational_space_torque(
                jacobian[:, part], mass, acceleration[:, part], solver=solver, rcond=rcond
            )
            for part in (slice(0, 3), slice(3, 6))
        )
    if solver == "inverse":
        inverse_inertia = torch.bmm(
            torch.bmm(jacobian, torch.linalg.inv(mass)), jacobian.transpose(-1, -2)
        )
        inertia = torch.linalg.inv(inverse_inertia)
        wrench = torch.bmm(inertia, acceleration.unsqueeze(-1))
        return torch.bmm(jacobian.transpose(-1, -2), wrench).squeeze(-1)
    if solver != "pinv":
        raise ValueError(f"Unknown OSC inertia solver: {solver}")
    jac, matrix = jacobian.double(), mass.double()
    # solve_ex skips the host-side error check; the mass matrix is positive definite.
    solution = torch.linalg.solve_ex(matrix, jac.transpose(-1, -2), check_errors=False).result
    inverse_inertia = jac @ solution
    inverse_inertia = (inverse_inertia + inverse_inertia.transpose(-1, -2)) / 2
    inertia = _pinv_hermitian(inverse_inertia, rcond)
    torque = jac.transpose(-1, -2) @ inertia @ acceleration.double().unsqueeze(-1)
    return torque.squeeze(-1).to(jacobian.dtype)


class OscAction(ActionTerm):
    """Six-DOF inertia-decoupled OSC; no gravity or null-space compensation."""

    cfg: OscActionCfg

    def __init__(self, cfg: OscActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        if cfg.inertia_solver not in ("inverse", "pinv"):
            raise ValueError(f"Unknown OSC inertia solver: {cfg.inertia_solver}")
        if not math.isfinite(cfg.inertia_rcond) or not 0 < cfg.inertia_rcond < 1:
            raise ValueError("OSC inertia_rcond must be finite and strictly between zero and one")
        robot = self._entity
        arm_joint_ids, _ = robot.find_joints(cfg.arm_joints, preserve_order=True)
        finger_joint_ids, _ = robot.find_joints(cfg.finger_joints, preserve_order=True)
        ee_ids, _ = robot.find_bodies(cfg.ee_body)
        if len(arm_joint_ids) != 6 or len(finger_joint_ids) != 2 or len(ee_ids) != 1:
            raise ValueError(
                "OSC requires six ordered arm joints, two fingers, and one end-effector body"
            )
        self._arm_joint_ids = torch.as_tensor(arm_joint_ids, device=self.device)
        self._finger_joint_ids = torch.as_tensor(finger_joint_ids, device=self.device)
        self._arm_dof_ids = robot.indexing.joint_v_adr[self._arm_joint_ids]
        self._ee_body_id = int(robot.indexing.body_ids[ee_ids[0]])
        self._raw_actions = torch.zeros((self.num_envs, 7), device=self.device)
        self._desired_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._desired_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._desired_quat[:, 0] = 1.0
        self._gripper_pos = torch.full((self.num_envs, 2), cfg.open_position, device=self.device)
        if len(cfg.motion_stiffness) != 6 or len(cfg.motion_damping_ratio) != 6:
            raise ValueError("OSC needs six task-space stiffness and damping ratios")
        self._stiffness = torch.tensor(cfg.motion_stiffness, device=self.device)
        ratio = torch.tensor(cfg.motion_damping_ratio, device=self.device)
        self._damping = 2.0 * self._stiffness.sqrt() * ratio
        nv = env.sim.mj_model.nv
        with wp.ScopedDevice(env.sim.wp_device):
            self._jacp_wp = wp.zeros((self.num_envs, 3, nv), dtype=float)
            self._jacr_wp = wp.zeros((self.num_envs, 3, nv), dtype=float)
            self._point_wp = wp.zeros(self.num_envs, dtype=wp.vec3)
            self._body_wp = wp.zeros(self.num_envs, dtype=wp.int32)
            self._body_wp.fill_(self._ee_body_id)
        self._jacp = wp.to_torch(self._jacp_wp)
        self._jacr = wp.to_torch(self._jacr_wp)
        self._point = wp.to_torch(self._point_wp).view(self.num_envs, 3)
        # Read the six-arm-joint block of the joint-space inertia with one gather
        # from the CSR storage, at the same addresses mujoco_warp's mul_m uses.
        model = env.sim.wp_model
        rowadr, columns, addresses = (
            array.numpy() for array in (model.M_mulm_rowadr, model.M_mulm_col, model.M_mulm_madr)
        )
        arm_dofs = [int(dof) for dof in self._arm_dof_ids.tolist()]
        index = [[-1] * 6 for _ in arm_dofs]
        for row, dof in enumerate(arm_dofs):
            for entry in range(rowadr[dof], rowadr[dof + 1]):
                if int(columns[entry]) in arm_dofs:
                    index[row][arm_dofs.index(int(columns[entry]))] = int(addresses[entry])
        if min(min(row) for row in index) < 0:
            raise ValueError("OSC arm inertia block is missing from the CSR mass matrix")
        self._mass_index = torch.tensor(index, device=self.device)
        self._mass = wp.to_torch(env.sim.wp_data.M)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        data = self._env.sim.data
        current_pos = data.xpos[:, self._ee_body_id]
        current_quat = data.xquat[:, self._ee_body_id]
        delta = actions[:, :6].clone()
        if self.cfg.input_limit is not None:
            delta.clamp_(-self.cfg.input_limit, self.cfg.input_limit)
        delta[:, :3] *= self.cfg.position_scale
        delta[:, 3:] *= self.cfg.orientation_scale
        # Bounded relative pose commands. Convert the configured
        # command speeds to per-action displacements using the control period.
        # This bounds the requested motion, not actual qvel or Cartesian speed.
        linear = self.cfg.linear_velocity_limit
        angular = self.cfg.angular_velocity_limit
        delta[:, :3] = limit_vector_norm(
            delta[:, :3], linear * self._env.step_dt if linear is not None else None
        )
        delta[:, 3:] = limit_vector_norm(
            delta[:, 3:], angular * self._env.step_dt if angular is not None else None
        )
        pos, quat = apply_delta_pose(current_pos, current_quat, delta)
        self._desired_pos[:] = pos
        self._desired_quat[:] = quat
        # Keep the existing binary action: negative closes, positive opens.
        target = torch.where(actions[:, 6] < 0.0, self.cfg.closed_position, self.cfg.open_position)
        self._gripper_pos[:] = target.unsqueeze(-1)

    def apply_actions(self) -> None:
        sim = self._env.sim
        data = sim.data
        pos = data.xpos[:, self._ee_body_id]
        quat = data.xquat[:, self._ee_body_id]
        pos_error, rot_error = compute_pose_error(pos, quat, self._desired_pos, self._desired_quat)
        pose_error = torch.cat((pos_error, rot_error), dim=-1)
        # Jacobian and velocity are evaluated at the center of mass;
        # the target pose remains the body frame origin.
        self._point[:] = data.xipos[:, self._ee_body_id]
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.jac(
                sim.wp_model,
                sim.wp_data,
                self._jacp_wp,
                self._jacr_wp,
                self._point_wp,
                self._body_wp,
            )
        jac = torch.cat((self._jacp, self._jacr), dim=1)[:, :, self._arm_dof_ids]
        ee_velocity = torch.bmm(jac, data.qvel[:, self._arm_dof_ids].unsqueeze(-1)).squeeze(-1)
        desired_accel = self._stiffness * pose_error - self._damping * ee_velocity
        # Invert the six-arm-joint mass block, not the inverse of
        # the full robot+box mass matrix.
        mass_matrix = self._mass[:, self._mass_index]
        # A world whose physics diverged gets zero arm torque (identity Jacobian
        # and mass, zero acceleration) so the batched solve stays well posed;
        # the task's divergence termination resets it. Finite worlds are unchanged.
        finite = (
            torch.isfinite(jac).flatten(1).all(-1)
            & torch.isfinite(mass_matrix).flatten(1).all(-1)
            & torch.isfinite(desired_accel).all(-1)
        )
        eye = torch.eye(6, device=self.device).expand_as(mass_matrix)
        jac = torch.where(finite[:, None, None], jac, eye)
        mass_matrix = torch.where(finite[:, None, None], mass_matrix, eye)
        desired_accel = torch.where(finite[:, None], desired_accel, 0.0)
        joint_torque = operational_space_torque(
            jac,
            mass_matrix,
            desired_accel,
            solver=self.cfg.inertia_solver,
            rcond=self.cfg.inertia_rcond,
            uncouple_pos_ori=self.cfg.uncouple_pos_ori,
        )
        arm_torque = joint_torque.clamp(-self.cfg.effort_limit, self.cfg.effort_limit)
        self._entity.set_joint_effort_target(arm_torque, joint_ids=self._arm_joint_ids)
        self._entity.set_joint_position_target(self._gripper_pos, joint_ids=self._finger_joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._desired_pos[env_ids] = 0.0
        self._desired_quat[env_ids] = 0.0
        self._desired_quat[env_ids, 0] = 1.0
        self._gripper_pos[env_ids] = self.cfg.open_position
