"""Workcell geometry shared by all tasks: table, walls and the robot mount.

The layout is metric and Z-up, with the robot support plate's bottom centre at
the world origin and its axes defining the world axes (tabletop at Z = 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

from clab.robots.arx5.model import rotation_quaternion
from clab.scenes.robosuite_table import table_spec

ASSET_DIR = Path(__file__).with_name("assets") / "workcell"


def load_layout() -> dict:
    layout = json.loads((ASSET_DIR / "layout.json").read_text())
    if layout["units"] != "meters" or layout["up"] != "+z":
        raise ValueError("The workcell layout must use meters and +z up")
    for matrix in (layout["table"]["T_table_to_world"], layout["robot"]["T_robot_to_world"]):
        transform = np.asarray(matrix, dtype=float)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0, 0, 0, 1))
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("Layout poses must be proper rigid transforms")
    table = layout["table"]
    if (
        any(
            not np.isfinite(table[k]) or table[k] <= 0
            for k in ("length", "width", "height", "thickness", "leg_width", "leg_inset")
        )
        or table["height"] <= table["thickness"]
    ):
        raise ValueError("Invalid workcell table dimensions")
    return layout


def table_cfg(layout: dict) -> EntityCfg:
    """Tabletop collision box and visual legs with the layout's dimensions and pose."""
    table = layout["table"]
    transform = np.asarray(table["T_table_to_world"])

    def make_spec():
        spec = table_spec()
        top = spec.geom("table_collision")
        top.size[:] = (table["length"] / 2, table["width"] / 2, table["thickness"] / 2)
        top.pos[2] = table["surface_z_m"] - table["thickness"] / 2
        for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1)), 1):
            leg = spec.geom(f"leg_{i}_visual")
            leg.type = mujoco.mjtGeom.mjGEOM_BOX
            leg.size[:] = (
                table["leg_width"] / 2,
                table["leg_width"] / 2,
                (table["height"] - table["thickness"]) / 2,
            )
            leg.pos[:] = (
                sx * (table["length"] / 2 - table["leg_inset"]),
                sy * (table["width"] / 2 - table["leg_inset"]),
                table["surface_z_m"] - (table["height"] + table["thickness"]) / 2,
            )
        return spec

    return EntityCfg(
        spec_fn=make_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(transform[:3, 3]), rot=tuple(rotation_quaternion(transform[:3, :3]))
        ),
    )


def room_cfg(layout: dict) -> EntityCfg:
    """Two visual wall planes and the robot support plate."""

    def make_spec():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="room")
        for wall in layout["walls"]:
            vertices = np.asarray(wall["vertices"], dtype=float)
            horizontal = vertices[1] - vertices[0]
            length = np.linalg.norm(horizontal)
            horizontal /= length
            vertical = np.array((0, 0, 1))
            rotation = np.column_stack((horizontal, np.cross(vertical, horizontal), vertical))
            body.add_geom(
                name=f"wall_{wall['name']}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=vertices.mean(axis=0),
                quat=rotation_quaternion(rotation),
                size=(length / 2, wall["thickness"] / 2, (vertices[2, 2] - vertices[0, 2]) / 2),
                rgba=(0.72, 0.72, 0.70, 1),
                contype=0,
                conaffinity=0,
                density=0,
            )
        # The mount mesh is already placed in world coordinates.
        spec.add_mesh(name="robot_mount", file=str(ASSET_DIR / layout["robot"]["mount"]["mesh"]))
        body.add_geom(
            name="robot_mount",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="robot_mount",
            rgba=(0.32, 0.32, 0.30, 1),
            density=0,
            friction=(1, 0.005, 0.0001),
            contype=1,
            conaffinity=1,
        )
        return spec

    return EntityCfg(spec_fn=make_spec)
