"""Small CPU MuJoCo IK helper for collision-checked reset proposal generators.

This solves a named site's pose, not a dynamics controller. Callers own target
sampling, collision checks and subsequent physical acceptance.
"""

from __future__ import annotations

import mujoco
import numpy as np


class SiteIK:
    """Damped least-squares pose IK with explicit joint order and hard limits."""

    def __init__(self, model: mujoco.MjModel, joint_names: tuple[str, ...], site: str):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.joint(name) for name in joint_names]
        if any(int(j.type[0]) != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("SiteIK currently accepts only scalar hinge joints")
        self.qadr = np.array([int(j.qposadr[0]) for j in joints])
        self.dofs = np.array([int(j.dofadr[0]) for j in joints])
        self.limits = np.stack([j.range.copy() for j in joints])
        self.site_id = model.site(site).id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def solve(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        seed: np.ndarray,
        *,
        iterations: int = 100,
        position_tolerance: float = 0.001,
        angle_tolerance: float = 0.015,
    ) -> np.ndarray | None:
        """Return an in-limit solution, or None; never accept a clipped target pose."""
        if seed.shape != (len(self.qadr),) or not np.isfinite(seed).all():
            raise ValueError("IK seed has incorrect shape or non-finite values")
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("IK target must be xyz and wxyz")
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError("IK target contains non-finite values")
        if not np.isclose(np.linalg.norm(quaternion), 1.0):
            raise ValueError("IK target quaternion must have unit norm")
        q = np.clip(seed, self.limits[:, 0], self.limits[:, 1])
        current_quat = np.empty(4)
        angle_error = np.empty(3)
        for _ in range(iterations):
            self.data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self.site_id])
            # subQuat returns a body-frame rotation; the Jacobian is world-frame.
            mujoco.mju_subQuat(angle_error, quaternion, current_quat)
            rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            angular = rotation @ angle_error
            translation = position - self.data.site_xpos[self.site_id]
            if (
                np.linalg.norm(translation) < position_tolerance
                and np.linalg.norm(angular) < angle_tolerance
            ):
                return q.copy()
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
            # Metres and radians have different magnitudes; this length weight
            # balances pose error without changing either acceptance tolerance.
            jacobian = np.vstack((self.jacp[:, self.dofs], 0.15 * self.jacr[:, self.dofs]))
            error = np.concatenate((translation, 0.15 * angular))
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(6), error)
            scale = min(1.0, 0.15 / max(float(np.max(np.abs(step))), 1e-12))
            q = np.clip(q + scale * step, self.limits[:, 0], self.limits[:, 1])
        return None
