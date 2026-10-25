"""Full-size source Drawer housing/passive slider and free coffee pod.

Geometry, body origins and unusual source inertias are deliberately preserved.
The pod's source 0.018 mesh scale is baked offline; runtime mesh scale is one.
Housing bottom extends 3 mm below its source origin (tabletop); the housing and
table are both fixed. Drawer floor starts 2 mm above that origin.
"""

from importlib.resources import files

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .settings import INITIAL_SLIDE_RANGE

HOUSING_BOXES = (
    ((0.0, 0.0, -0.001), (0.1, 0.095, 0.002)),
    ((-0.099, 0.0, 0.0375), (0.001, 0.093, 0.0375)),
    ((0.0, 0.094, 0.0375), (0.1, 0.001, 0.0375)),
    ((0.0, -0.094, 0.0375), (0.1, 0.001, 0.0375)),
    ((0.0, 0.0, 0.074), (0.1, 0.093, 0.001)),
    ((0.0, 0.0, 0.001), (0.1, 0.093, 0.001)),
)
DRAWER_BOXES = (
    ((0.099, 0.0, 0.0375), (0.001, 0.093, 0.0355)),
    ((0.0, 0.092, 0.032), (0.098, 0.001, 0.03)),
    ((0.0, -0.092, 0.032), (0.098, 0.001, 0.03)),
    ((0.0, 0.0, 0.003), (0.098, 0.091, 0.001)),
)
HOUSING_GEOMS = tuple(f"housing_collision_{i}" for i in range(len(HOUSING_BOXES)))
DRAWER_GEOMS = tuple(f"drawer_collision_{i}" for i in range(len(DRAWER_BOXES)))

POD_BOTTOM_OFFSET = -0.023103
POD_TOP_OFFSET = 0.023103
POD_INITIAL_Z = 0.001 - POD_BOTTOM_OFFSET


def _boxes(body, boxes, names):
    for (pos, size), name in zip(boxes, names, strict=True):
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            group=3,
            rgba=(0.78, 0.68, 0.06, 1),
            mass=0,
        )
        body.add_geom(
            name=name.replace("collision", "visual"),
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            contype=0,
            conaffinity=0,
            group=1,
            rgba=(0.78, 0.68, 0.06, 1),
            mass=0,
        )


def drawer_spec():
    spec = mujoco.MjSpec()
    housing = spec.worldbody.add_body(
        name="housing",
        mocap=True,
        mass=2.0,
        inertia=(0.5, 0.5, 0.5),
        ipos=(0, 0, -0.001),
        iquat=(0.5, 0.5, 0.5, 0.5),
        explicitinertial=True,
    )
    _boxes(housing, HOUSING_BOXES, HOUSING_GEOMS)
    slider = housing.add_body(
        name="drawer",
        mass=0.2,
        inertia=(0.5, 0.5, 0.5),
        ipos=(0, 0, 0.0375),
        iquat=(0.5, 0.5, 0.5, 0.5),
        explicitinertial=True,
    )
    slider.add_joint(
        name="slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(1, 0, 0),
        limited=True,
        range=(0, 0.14),
        damping=0.1,
    )
    _boxes(slider, DRAWER_BOXES, DRAWER_GEOMS)
    return spec


def pod_spec():
    spec = mujoco.MjSpec()
    asset_name = "coffee_pod.stl"
    spec.assets[asset_name] = files(__package__).joinpath("assets", asset_name).read_bytes()
    spec.add_mesh(name="coffee_pod", file=asset_name)
    pod = spec.worldbody.add_body(name="pod")
    # Match the source scalar damping: no quadratic or cubic terms.
    pod.add_freejoint(name="pod_free").damping[:] = (0.0005, 0.0, 0.0)
    pod.add_geom(
        name="pod_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="coffee_pod",
        solimp=(0.998, 0.998, 0.001, 0.5, 2.0),
        solref=(0.001, 1),
        density=100,
        friction=(1, 1, 1),
        condim=4,
        group=3,
        rgba=(0.04, 0.55, 0.06, 1),
    )
    pod.add_geom(
        name="pod_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="coffee_pod",
        contype=0,
        conaffinity=0,
        group=1,
        mass=0.0001,
        rgba=(0.04, 0.55, 0.06, 1),
    )
    return spec


def drawer_cfg():
    return EntityCfg(
        spec_fn=drawer_spec,
        # Source nominal (-.205, -.27, .8), expressed about the robot base.
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.33, -0.06, 0),
            rot=(0, 0, 0, 1),
            joint_pos={"slide": INITIAL_SLIDE_RANGE[0]},
        ),
        articulation=EntityArticulationInfoCfg(actuators=()),
    )


def pod_cfg():
    return EntityCfg(
        spec_fn=pod_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.14, 0.17, POD_INITIAL_Z)),
    )
