"""Packaged source mug decomposition and a fixed, resettable three-box tree.

The original mesh scale 0.15 is baked offline into all 34 OBJ resources. The
collector's additional mug scale is one. Density, freejoint damping and contact
parameters follow the source model; visual meshes never contribute inertia.
"""

from importlib.resources import files

import mujoco
from mjlab.entity import EntityCfg

MUG_BOTTOM_OFFSET = -0.0420880482
MUG_TOP_OFFSET = 0.0406138479
MUG_INITIAL_Z = 0.001 - MUG_BOTTOM_OFFSET
MUG_GRASP_POINT = (0.0, -0.05, 0.03)
HANDLE_ANCHOR = (0.0, 0.03, 0.0)

TREE_BOTTOM_OFFSET = -0.095
TREE_INITIAL_Z = 0.001 - TREE_BOTTOM_OFFSET
BRANCH_INNER = (0.015, 0.0, 0.055)
BRANCH_OUTER = (0.095, 0.0, 0.055)
BRANCH_LENGTH = 0.08
HANG_TARGET = tuple(
    (inner + outer) / 2 for inner, outer in zip(BRANCH_INNER, BRANCH_OUTER, strict=True)
)
MUG_COLLISION_GEOMS = tuple(f"mug_collision_{i}" for i in range(32))
TREE_COLLISION_GEOMS = ("tree_base_collision", "tree_post_collision", "tree_branch_collision")


def mug_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    # Only the source collision meshes generate inertia. Group 3 excludes
    # these transparent hulls from RGB cameras rendering groups 0/1/2; keep
    # the compiler range matched so the physical mass/inertia stay identical.
    spec.compiler.inertiagrouprange = (3, 3)
    body = spec.worldbody.add_body(name="mug")
    # Match the source scalar damping: no quadratic or cubic terms.
    body.add_freejoint(name="mug_free").damping[:] = (0.0005, 0.0, 0.0)
    for visual, count in ((True, 2), (False, 32)):
        for index in range(count):
            filename = (
                f"visual/model_normalized_{index}.obj"
                if visual
                else f"collision/model_normalized_collision_{index}.obj"
            )
            name = f"mug_visual_{index}" if visual else MUG_COLLISION_GEOMS[index]
            spec.assets[filename] = files(__package__).joinpath("assets", filename).read_bytes()
            spec.add_mesh(name=name, file=filename)
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                density=0.0 if visual else 100.0,
                contype=0 if visual else 1,
                conaffinity=0 if visual else 1,
                group=1 if visual else 3,
                friction=(0.95, 0.3, 0.1),
                solimp=(0.998, 0.998, 0.001, 0.5, 2.0),
                solref=(0.001, 1.0),
                rgba=(0.75, 0.02, 0.02, 1.0) if visual else (0.1, 0.8, 0.2, 0.0),
            )
    return spec


def tree_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="tree", mocap=True)
    # Positions are relative to the source total-height centre, not tabletop.
    for name, centre, halfsize in zip(
        TREE_COLLISION_GEOMS,
        ((0, 0, -0.08), (0, 0, 0.015), (0.055, 0, 0.055)),
        ((0.08, 0.08, 0.015), (0.015, 0.015, 0.08), (0.04, 0.0025, 0.0075)),
        strict=True,
    ):
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=centre,
            size=halfsize,
            density=5000.0,
            rgba=(0.55, 0.32, 0.12, 1.0),
        )
    return spec


def mug_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=mug_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.35, -0.12, MUG_INITIAL_Z)),
    )


def tree_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=tree_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.35, 0.16, TREE_INITIAL_Z), rot=(2**-0.5, 0, 0, -(2**-0.5))
        ),
    )
