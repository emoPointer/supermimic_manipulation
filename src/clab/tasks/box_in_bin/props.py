"""BoxInBin movable box and immovable open-bin MJCF primitives."""

from __future__ import annotations

import math

import mujoco
from mjlab.entity import EntityCfg

from .settings import BIN_CENTRE_Z, BOX_CENTRE_Z


def box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    body.add_freejoint(name="box_free")
    body.add_geom(
        name="box_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.0565, 0.0225, 0.0505),
        mass=0.1,
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.45, 0.015, 0.015, 1.0),
    )
    return spec


def box_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=box_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.16, -0.12, BOX_CENTRE_Z))
    )


def validate_object_friction(coefficient: float) -> float:
    """Validate an explicit dimensionless box sliding-friction override."""
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
        raise ValueError("Box sliding friction must be a positive finite number")
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Box sliding friction must be a positive finite number")
    return float(coefficient)


def configure_object_friction(scene_cfg, coefficient: float) -> None:
    """Change only this scene's movable-box sliding friction, not defaults."""
    coefficient = validate_object_friction(coefficient)
    previous = scene_cfg.spec_fn

    def object_friction_spec(spec):
        if previous is not None:
            previous(spec)
        geom = spec.geom("box/box_collision")
        if geom is None or not (geom.contype or geom.conaffinity):
            raise ValueError("Missing contact-enabled BoxInBin box/box_collision")
        geom.friction[0] = coefficient

    scene_cfg.spec_fn = object_friction_spec


def bin_spec() -> mujoco.MjSpec:
    """Five separate collision boxes leave the top open."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="bin")
    # The fixed bin carries a 0.057455286 kg rigid-body mass.
    # Keep the same physical parameter in the MuJoCo model/critic observation.
    bin_mass_per_part = 0.057455286383628845 / 5
    parts = (
        ("floor", (0.0, 0.0, -0.01475), (0.10725, 0.07225, 0.00475)),
        ("wall_negative_y", (0.0, -0.07225, 0.0), (0.112, 0.00475, 0.0195)),
        ("wall_negative_x", (-0.10725, 0.0, 0.0), (0.00475, 0.077, 0.0195)),
        ("wall_positive_y", (0.0, 0.07225, 0.0), (0.112, 0.00475, 0.0195)),
        ("wall_positive_x", (0.10725, 0.0, 0.0), (0.00475, 0.077, 0.0195)),
    )
    for name, pos, size in parts:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            mass=bin_mass_per_part,
            friction=(1.0, 0.005, 0.0001),
            rgba=(1.0, 0.92, 0.0, 1.0),
        )
    return spec


def bin_cfg() -> EntityCfg:
    # Fixed base becomes a mocap entity in mjlab. Only reset events may move it.
    return EntityCfg(
        spec_fn=bin_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.27, 0.02, BIN_CENTRE_Z))
    )
