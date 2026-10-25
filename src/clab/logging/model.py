"""Small readers for recording effective compiled physical-model parameters."""

from __future__ import annotations

import mujoco
import numpy as np


def joint_damping_metadata(model: mujoco.MjModel, joint_name: str) -> dict:
    """Read every DOF's linear, quadratic and cubic joint-damping coefficients.

    Values come from the compiled model, not scalar configuration shorthand.
    A free joint has six ordered DOFs (translation then rotation); a ball joint
    has three and a hinge/slide one. Missing coefficients are errors, never
    silently interpreted as zero. This reads the supplied model only: callers
    must not use a nominal CPU template to describe randomized GPU parameters.
    """
    if not isinstance(joint_name, str) or not joint_name:
        raise ValueError("An explicit nonempty joint name is required")
    try:
        joint = model.joint(joint_name)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unknown joint for damping metadata: {joint_name!r}") from error
    count = {
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }[int(joint.type[0])]
    address = int(joint.dofadr[0])
    try:
        linear = np.asarray(model.dof_damping)[address : address + count]
        polynomial = np.asarray(model.dof_dampingpoly)[address : address + count]
    except AttributeError as error:
        raise ValueError(
            "Compiled model must expose linear and polynomial joint damping"
        ) from error
    if linear.shape != (count,) or polynomial.shape != (count, 2):
        raise ValueError(f"Incomplete compiled damping coefficients for joint {joint_name!r}")
    if not np.isfinite(linear).all() or not np.isfinite(polynomial).all():
        raise ValueError(f"Nonfinite compiled damping coefficients for joint {joint_name!r}")
    return {
        "joint": joint_name,
        "dof_address": address,
        "dof_count": count,
        "linear": linear.tolist(),
        "quadratic": polynomial[:, 0].tolist(),
        "cubic": polynomial[:, 1].tolist(),
    }
