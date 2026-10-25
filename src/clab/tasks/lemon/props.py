"""Lemon task props with packaged, independently loaded mesh assets.

Body frames and baked mesh dimensions follow the data-collection model. MuJoCo may
recenter compiled meshes internally; do not apply an additional body offset.
"""

from __future__ import annotations

from importlib.resources import files
from itertools import product

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

# Source sampling sites, not the minimum/maximum mesh vertices.
LEMON_BOTTOM_OFFSET = -0.0296
PLATE_BOTTOM_OFFSET = -0.0060
# CPU ray at body-local XY=(0, 0); the scanned plate floor is not planar.
PLATE_CENTRE_FLOOR_Z = 0.0015841257320302893
# Lemon mass (the source asset used 0.029 kg).
LEMON_MASS = 0.080
PLATE_MASS = 0.279
LEMON_INERTIA = (0.001, 0.001, 0.001)
PLATE_INERTIA = (0.001, 0.001, 0.001)
LEMON_INITIAL_CENTRE_Z = 0.01 - LEMON_BOTTOM_OFFSET
PLATE_CENTRE_Z = 0.002 - PLATE_BOTTOM_OFFSET
LEMON_COLLISION_GEOMS = ("lemon_collision_0", "lemon_collision_1")
PLATE_COLLISION_GEOMS = tuple(f"plate_collision_{i}" for i in range(5))


def _add_mesh(
    spec: mujoco.MjSpec,
    prop: str,
    filename: str,
    name: str,
) -> None:
    """Load resource bytes into MuJoCo's virtual filesystem, not a checkout path."""
    resource = files(__package__).joinpath("assets", prop, filename)
    asset_name = f"{prop}/{filename}"
    spec.assets[asset_name] = resource.read_bytes()
    spec.add_mesh(name=name, file=asset_name)


def _add_geometries(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    *,
    name: str,
    collision_geoms: tuple[str, ...],
    friction: tuple[float, float, float],
    condim: int,
    rgba: tuple[float, float, float, float],
) -> None:
    visual_mesh = f"{name}_visual_mesh"
    _add_mesh(spec, name, "textured.obj", visual_mesh)
    material = spec.add_material(name=f"{name}_material", rgba=rgba)
    if name == "plate":
        material.specular = 0.1
        material.shininess = 0.1
        material.reflectance = 0.0
    body.add_geom(
        name=f"{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=visual_mesh,
        material=material.name,
        contype=0,
        conaffinity=0,
        group=1,
        mass=0.0,
    )
    for index, geom_name in enumerate(collision_geoms):
        mesh_name = f"{geom_name}_mesh"
        _add_mesh(spec, name, f"textured_coacd_{index}.stl", mesh_name)
        body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            contype=1,
            conaffinity=1,
            condim=condim,
            friction=friction,
            solimp=(0.998, 0.998, 0.001, 0.5, 2.0),
            solref=(0.001, 1.0),
            group=3,
            rgba=rgba,
            mass=0.0,
        )


def lemon_spec() -> mujoco.MjSpec:
    """Movable lemon: two convex collision pieces and a separate visual mesh."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(
        name="lemon",
        mass=LEMON_MASS,
        inertia=LEMON_INERTIA,
        explicitinertial=True,
    )
    # MjSpec damping stores linear, quadratic and cubic coefficients.
    body.add_freejoint(name="lemon_free").damping[:] = (0.01, 0.0, 0.0)
    _add_geometries(
        spec,
        body,
        name="lemon",
        collision_geoms=LEMON_COLLISION_GEOMS,
        friction=(2.0, 0.5, 0.1),
        condim=4,
        rgba=(1.0, 1.0, 0.0, 1.0),
    )
    return spec


def plate_spec() -> mujoco.MjSpec:
    """Jointless plate: reset may move its mocap body, physics cannot."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(
        name="plate",
        mocap=True,
        mass=PLATE_MASS,
        inertia=PLATE_INERTIA,
        explicitinertial=True,
    )
    _add_geometries(
        spec,
        body,
        name="plate",
        collision_geoms=PLATE_COLLISION_GEOMS,
        friction=(0.95, 0.3, 0.1),
        condim=3,
        rgba=(0.45, 0.015, 0.015, 1.0),
    )
    return spec


def lemon_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lemon_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.28, -0.08, LEMON_INITIAL_CENTRE_Z)),
    )


def plate_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=plate_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.30, 0.06, PLATE_CENTRE_Z)),
    )


def workspace_corners(model) -> dict[str, np.ndarray]:
    """Conservative body-local bounds enclosing every visual/collision mesh.

    Compiled mesh vertices include MuJoCo recentering; geom pose restores their
    body-local placement. Rotating the enclosing box may reject extra states,
    but cannot admit a mesh edge outside the shared workspace.
    """
    result = {}
    for name in ("lemon", "plate"):
        body = model.body(f"{name}/{name}").id
        points = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f"Expected mesh geometry for {name}")
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            points.append(
                model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
                + model.geom_pos[geom]
            )
        vertices = np.concatenate(points)
        result[name] = np.array(list(product(*zip(vertices.min(0), vertices.max(0), strict=True))))
    return result


def within_workspace(poses, corners):
    """Batched Torch poses in environment-world coordinates, including rotation."""
    from mjlab.utils.lab_api.math import quat_apply

    from clab.scenes.workspace import contains_bounds

    valid = None
    for name, vertices in corners.items():
        pose = poses[f"{name}_pose"]
        points = quat_apply(
            pose[:, None, 3:].expand(-1, 8, -1),
            pose.new_tensor(vertices)[None].expand(len(pose), -1, -1),
        )
        points += pose[:, None, :3]
        inside = contains_bounds(points.amin(1), points.amax(1))
        valid = inside if valid is None else valid & inside
    return valid
