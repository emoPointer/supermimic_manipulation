"""Exact source Square primitives; no external repository is loaded at runtime.

Geometry follows robosuite/models/assets/objects/{square-nut,peg_with_base}.xml.
Explicit body inertias reproduce source compilation with inertiagrouprange=0 0;
overlapping primitive masses are intentionally retained. All lengths are metres.
"""

from itertools import product

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

from clab.tensors import constant

NUT_BOXES = (
    ((-0.03325, 0.0, 0.0), (0.0105, 0.04375, 0.01)),
    ((0.0, 0.03325, 0.0), (0.03125, 0.0105, 0.01)),
    ((0.0, -0.03325, 0.0), (0.03125, 0.0105, 0.01)),
    ((0.03325, 0.0, 0.0), (0.0105, 0.04375, 0.01)),
    ((0.054, 0.0, 0.0), (0.02525, 0.015875, 0.01)),
)
PEG_BOXES = (
    ((0.0, 0.0, -0.045), (0.05, 0.05, 0.015)),
    ((0.0, 0.0, 0.0), (0.015, 0.015, 0.03)),
)
NUT_COLLISION_GEOMS = tuple(f"nut_collision_{index}" for index in range(5))
PEG_COLLISION_GEOMS = tuple(f"peg_collision_{index}" for index in range(2))
NUT_MASS = 0.01580675
NUT_COM = (0.010955098296613789, 0.0, 0.0)
NUT_INERTIA = (1.1482869118489675e-5, 1.87672229799745e-5, 2.9196308765130837e-5)
PEG_MASS = 0.0354
PEG_COM = (0.0, 0.0, -0.038135593220338986)
PEG_INERTIA = (3.854194915254258e-5, 3.854194915254258e-5, 5.081e-5)
NUT_GRASP_POINT = (0.064, 0.0, 0.012)
NUT_HANDLE_SITE = (0.054, 0.0, 0.0)
HOLE_HALF_WIDTH = 0.02275
PEG_HALF_WIDTH = 0.015
PEG_HALF_HEIGHT = 0.03
NUT_HALF_HEIGHT = 0.01
PEG_ROOT_Z = 0.061
NUT_SPAWN_Z = 0.051  # Source bottom sampling site is -0.05, not geometry -0.01.
NUT_SEATED_OFFSET = -0.02  # Nut bottom at the fixed peg base's top surface.


def workspace_corners():
    """Exact corners of all visual/collision boxes, including handle and peg base."""
    signs = np.array(tuple(product((-1.0, 1.0), repeat=3)))
    return {
        name: np.concatenate([np.asarray(pos) + signs * size for pos, size in boxes])
        for name, boxes in (("nut", NUT_BOXES), ("peg", PEG_BOXES))
    }


def within_workspace(poses, corners):
    """Check complete rotated geometry in batched environment-world poses."""
    from mjlab.utils.lab_api.math import quat_apply

    from clab.scenes.workspace import contains_bounds

    valid = None
    for name, vertices in corners.items():
        pose = poses[f"{name}_pose"]
        points = (
            quat_apply(
                pose[:, None, 3:].expand(-1, len(vertices), -1),
                constant(pose, vertices)[None].expand(len(pose), -1, -1),
            )
            + pose[:, None, :3]
        )
        inside = contains_bounds(points.amin(1), points.amax(1))
        valid = inside if valid is None else valid & inside
    return valid


def handle_contact_bounds(clearance: float, tolerance: float):
    """Exposed handle region; exclude its collision-box overlap with the ring."""
    centre, half_size = NUT_BOXES[4]
    ring_outer_x = max(position[0] + size[0] for position, size in NUT_BOXES[:4])
    lower = (
        ring_outer_x + clearance,
        centre[1] - half_size[1] - tolerance,
        centre[2] - half_size[2] - tolerance,
    )
    upper = tuple(
        position + size + tolerance for position, size in zip(centre, half_size, strict=True)
    )
    if lower[0] >= upper[0]:
        raise ValueError("Square handle clearance removes the entire grasp region")
    return lower, upper


def _boxes(body, name, boxes, color):
    for index, (position, half_size) in enumerate(boxes):
        common = dict(type=mujoco.mjtGeom.mjGEOM_BOX, pos=position, size=half_size, mass=0.0)
        body.add_geom(
            name=f"{name}_visual_{index}",
            **common,
            rgba=color,
            contype=0,
            conaffinity=0,
            group=1,
        )
        contact = (
            dict(
                friction=(0.95, 0.3, 0.2 if index == 0 else 0.1),
                condim=4,
                solimp=(0.998, 0.998, 0.001, 0.5, 2.0),
                solref=(0.02, 1.0),
            )
            if name == "nut"
            else dict(friction=(1.0, 0.005, 0.0001), condim=3)
        )
        body.add_geom(
            name=f"{name}_collision_{index}",
            **common,
            **contact,
            rgba=color,
            contype=1,
            conaffinity=1,
            group=3,
        )


def nut_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(
        name="nut", mass=NUT_MASS, ipos=NUT_COM, inertia=NUT_INERTIA, explicitinertial=True
    )
    # Match the source scalar damping: no quadratic or cubic terms.
    body.add_freejoint(name="nut_free").damping[:] = (0.0005, 0.0, 0.0)
    _boxes(body, "nut", NUT_BOXES, (0.04, 0.55, 0.06, 1.0))
    body.add_site(name="handle", pos=NUT_HANDLE_SITE, size=(0.003, 0.003, 0.003), group=4)
    body.add_site(name="grasp", pos=NUT_GRASP_POINT, size=(0.003, 0.003, 0.003), group=4)
    return spec


def peg_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(
        name="peg",
        mocap=True,
        mass=PEG_MASS,
        ipos=PEG_COM,
        inertia=PEG_INERTIA,
        explicitinertial=True,
    )
    _boxes(body, "peg", PEG_BOXES, (0.68, 0.04, 0.04, 1.0))
    return spec


def nut_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=nut_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.25, -0.06, NUT_SPAWN_Z))
    )


def peg_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=peg_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.28, 0.09, PEG_ROOT_Z))
    )
