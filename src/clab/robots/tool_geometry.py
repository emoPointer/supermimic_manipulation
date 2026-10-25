"""Physical tool envelopes and clearance-limited world-frame rotations.

These geometric queries do not step physics or alter controller/task settings.
A permitted rotation is safe only for the supplied fixed link6 position and
geometry; it is not a guarantee about plant tracking or the rest of the arm.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch


def tool_geometry_points(model, finger_open: float, *, robot) -> np.ndarray:
    """Return all compiled tool mesh vertices in physical link6 coordinates.

    Include visual/collision meshes on link6 and every descendant, at both ends
    of the fingers' prismatic travel. Vertex positions during that travel are
    affine, so their endpoint union also bounds every intermediate opening.
    Only a fresh local MjData is changed; the supplied model is never modified.
    """
    if not math.isfinite(finger_open) or finger_open < 0:
        raise ValueError("Tool finger opening must be finite and nonnegative")
    wrist = model.body(f"robot/{robot.ee_body}").id
    bodies = {wrist}
    for body in range(wrist + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    qids = []
    for name in robot.finger_joints:
        joint = model.joint(f"robot/{name}")
        if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError(f"Tool envelope requires prismatic finger joint {name}")
        if not model.jnt_range[joint.id, 0] <= 0 <= finger_open <= model.jnt_range[joint.id, 1]:
            raise ValueError(f"Tool opening is outside the physical limits of {name}")
        qids.append(int(joint.qposadr[0]))
    data = mujoco.MjData(model)
    points = []
    for opening in (0.0, finger_open):
        mujoco.mj_resetData(model, data)
        data.qpos[qids] = opening
        mujoco.mj_forward(model, data)
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in bodies:
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f"Unsupported non-mesh tool geometry: {model.geom(geom).name}")
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = (
                model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                + data.geom_xpos[geom]
            )
            points.append((world - data.xpos[wrist]) @ data.xmat[wrist].reshape(3, 3))
    if not points:
        raise ValueError("The compiled tool has no mesh vertices")
    result = np.unique(np.concatenate(points), axis=0)
    if not np.isfinite(result).all() or not len(result):
        raise ValueError("The compiled tool geometry must be finite and nonempty")
    return result


def safe_rotation_fraction(
    points_l6: torch.Tensor,
    position_w: torch.Tensor,
    quaternion_w: torch.Tensor,
    rotation_vector_w: torch.Tensor,
    floor_z: torch.Tensor,
) -> torch.Tensor:
    """Largest safe prefix of a left-multiplied world rotation, in [0, 1].

    Inputs are [V,3], [N,3], [N,4] wxyz, [N,3] world rotation vectors and [N]
    minimum allowed heights. Rotation-vector norms must not exceed pi. The
    caller owns the trajectory, plane clearance,
    and any response when the CURRENT geometry is already below that plane.
    Such a state returns zero: this is not a recovery motion or a safety claim.

    Each vertex height along Rodrigues' rotation is C+A*cos(t)+B*sin(t).
    Checking both ends and every interior minimum gives the entire prefix's
    minimum, not merely its endpoint. This prefix predicate is monotone even
    when vertex heights themselves are not; 32 bisections find its boundary.
    Float64 intermediates and downward-rounded partial fractions avoid rounding
    an admissible prefix upward. Inputs are not modified.
    """
    if position_w.ndim != 2 or position_w.shape[1] != 3:
        raise ValueError("Tool positions must have shape [N,3]")
    n = position_w.shape[0]
    expected = {
        "quaternion_w": (quaternion_w, (n, 4)),
        "rotation_vector_w": (rotation_vector_w, (n, 3)),
        "floor_z": (floor_z, (n,)),
    }
    if points_l6.ndim != 2 or points_l6.shape[1] != 3 or not len(points_l6):
        raise ValueError("Tool vertices must have nonempty shape [V,3]")
    for name, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name, value in (
        ("points_l6", points_l6),
        ("position_w", position_w),
        *((name, value) for name, (value, _) in expected.items()),
    ):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite floating-point values")
        if value.device != position_w.device:
            raise ValueError(f"{name} must use the same device as position_w")
    points = points_l6.to(dtype=torch.float64)
    p = position_w.to(dtype=torch.float64)
    q = quaternion_w.to(dtype=torch.float64)
    rv = rotation_vector_w.to(dtype=torch.float64)
    floor = floor_z.to(dtype=torch.float64)
    qnorm = q.norm(dim=-1, keepdim=True)
    if bool((qnorm - 1).abs().gt(1e-4).any()):
        raise ValueError("Tool orientation must be a unit wxyz quaternion")
    q = q / qnorm
    angle = rv.norm(dim=-1)
    if bool((angle > math.pi + 1e-6).any()):
        raise ValueError("Tool rotation must be a shortest rotation of at most pi radians")
    axis = rv / angle[:, None].clamp_min(1e-30)
    points = points[None].expand(n, -1, -1)
    qvector = q[:, None, 1:].expand_as(points)
    cross = 2 * torch.cross(qvector, points, dim=-1)
    world = points + q[:, None, :1] * cross + torch.cross(qvector, cross, dim=-1)
    projection = (world * axis[:, None]).sum(-1)
    c = p[:, 2:3] + axis[:, 2:3] * projection
    a = world[:, :, 2] - axis[:, 2:3] * projection
    b = torch.cross(axis[:, None].expand_as(world), world, dim=-1)[:, :, 2]
    stationary = torch.remainder(torch.atan2(b, a) + math.pi, 2 * math.pi)
    stationary_height = c - torch.sqrt(a.square() + b.square())

    def prefix_height(fraction):
        theta = (fraction * angle)[:, None]
        height = torch.minimum(c + a, c + a * theta.cos() + b * theta.sin())
        height = torch.minimum(
            height,
            torch.where(stationary <= theta, stationary_height, float("inf")),
        )
        return height.amin(dim=1)

    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    current_safe = prefix_height(zero) >= floor
    if not bool(current_safe.any()):
        return zero.to(dtype=position_w.dtype)
    full_safe = prefix_height(one) >= floor
    if bool(full_safe.all()):
        return one.to(dtype=position_w.dtype)
    low, high = zero.clone(), one.clone()
    for _ in range(32):
        middle = (low + high) * 0.5
        safe = prefix_height(middle) >= floor
        low = torch.where(safe, middle, low)
        high = torch.where(safe, high, middle)
    fraction = torch.where(full_safe, one, low)
    fraction = torch.where(current_safe, fraction, zero).to(dtype=position_w.dtype)
    partial = (fraction > 0) & (fraction < 1)
    return torch.where(partial, torch.nextafter(fraction, torch.zeros_like(fraction)), fraction)
