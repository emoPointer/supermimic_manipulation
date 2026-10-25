"""Reusable robosuite-style table, expressed relative to a configurable local origin."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

TABLE_BODY_CENTRE = (0.535, 0.21, -0.025)
GROUND_Z = -0.868


def table_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")
    body.add_geom(
        name="table_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.6, 0.45, 0.025),
        mass=54.0,
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.75, 0.75, 0.72, 1.0),
    )
    for index, (x, y) in enumerate(
        ((0.5, 0.35), (-0.5, 0.35), (-0.5, -0.35), (0.5, -0.35)), start=1
    ):
        body.add_geom(
            name=f"leg_{index}_visual",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(x, y, -0.3875),
            size=(0.025, 0.3875, 0.0),
            density=0.0,
            contype=0,
            conaffinity=0,
            rgba=(0.55, 0.55, 0.55, 1.0),
        )
    return spec


def table_cfg(position: tuple[float, float, float] = TABLE_BODY_CENTRE) -> EntityCfg:
    return EntityCfg(spec_fn=table_spec, init_state=EntityCfg.InitialStateCfg(pos=position))


def ground_spec() -> mujoco.MjSpec:
    """Collision ground below the table legs."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ground")
    body.add_geom(
        name="ground_collision",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(2.0, 2.0, 0.1),
        friction=(1.0, 0.005, 0.0001),
    )
    return spec


def ground_cfg(height: float = GROUND_Z) -> EntityCfg:
    return EntityCfg(
        spec_fn=ground_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, height))
    )
