"""ARX5 convex-mesh MuJoCo model as a reusable mjlab entity.

The asset keeps ideal body gravity compensation; the operational-space
controller adds no gravity term of its own.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

from .description import FINGER_ACTUATORS, FINGER_BODIES, HOME, JOINTS, ROOT_BODY

MODEL_PATH = Path(__file__).resolve().parent / "assets/arx5.xml"
FINGER_COLLISION_MODES = ("source_mesh_convex_hull", "convex_decomposition")


def rotation_quaternion(matrix) -> tuple[float, ...]:
    """wxyz quaternion of a 3x3 rotation matrix."""
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=float).reshape(9))
    return tuple(float(value) for value in quaternion)


def gripper_actuator_contract(model, *, entity_name="robot") -> dict:
    """Read actual compiled finger settings; never infer them from CLI values."""
    return {
        "force_units": "N per finger",
        "actuators": {
            name: {
                "forcerange": model.actuator_forcerange[
                    model.actuator(f"{entity_name}/{name}").id
                ].tolist(),
                "forcelimited": bool(
                    model.actuator_forcelimited[model.actuator(f"{entity_name}/{name}").id]
                ),
                "ctrlrange": model.actuator_ctrlrange[
                    model.actuator(f"{entity_name}/{name}").id
                ].tolist(),
                "ctrllimited": bool(
                    model.actuator_ctrllimited[model.actuator(f"{entity_name}/{name}").id]
                ),
                "gainprm": model.actuator_gainprm[
                    model.actuator(f"{entity_name}/{name}").id
                ].tolist(),
                "biasprm": model.actuator_biasprm[
                    model.actuator(f"{entity_name}/{name}").id
                ].tolist(),
                "gaintype": int(
                    model.actuator_gaintype[model.actuator(f"{entity_name}/{name}").id]
                ),
                "biastype": int(
                    model.actuator_biastype[model.actuator(f"{entity_name}/{name}").id]
                ),
            }
            for name in FINGER_ACTUATORS
        },
    }


def validate_gripper_friction(coefficient: float) -> float:
    """Validate an explicitly requested dimensionless sliding coefficient."""
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
        raise ValueError("Gripper sliding friction must be a positive finite number")
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Gripper sliding friction must be a positive finite number")
    return float(coefficient)


def configure_gripper_friction(scene_cfg, coefficient: float, *, entity_name="robot") -> None:
    """Override only sliding friction on actual finger collision geoms.

    Select contact-enabled geoms attached to both finger bodies, including all
    compound collision pieces. Keep visual-only geoms, torsional/rolling
    friction, solver/contact softness, priorities and actuators unchanged.
    """
    coefficient = validate_gripper_friction(coefficient)
    previous = scene_cfg.spec_fn

    def gripper_friction_spec(spec):
        if previous is not None:
            previous(spec)
        geoms = []
        for name in FINGER_BODIES:
            body = spec.body(f"{entity_name}/{name}")
            contacts = [] if body is None else [g for g in body.geoms if g.contype or g.conaffinity]
            if not contacts:
                raise ValueError(f"Missing ARX5 finger collision geoms: {entity_name}/{name}")
            geoms.extend(contacts)
        for geom in geoms:
            geom.friction[0] = coefficient

    scene_cfg.spec_fn = gripper_friction_spec


def gripper_contact_contract(model, *, entity_name="robot") -> dict:
    """Read all actual compiled finger collision geoms and friction settings."""
    geoms = {}
    for name in FINGER_BODIES:
        body_name = f"{entity_name}/{name}"
        body_id = model.body(body_name).id
        indices = [
            index
            for index in range(model.ngeom)
            if model.geom_bodyid[index] == body_id
            and (model.geom_contype[index] or model.geom_conaffinity[index])
        ]
        if not indices:
            raise ValueError(f"Missing ARX5 finger collision geoms: {body_name}")
        for index in indices:
            geoms[model.geom(index).name] = {
                "body": body_name,
                "friction": model.geom_friction[index].tolist(),
                "priority": int(model.geom_priority[index]),
                "condim": int(model.geom_condim[index]),
                "contype": int(model.geom_contype[index]),
                "conaffinity": int(model.geom_conaffinity[index]),
            }
    return {"friction_order": ["sliding", "torsional", "rolling"], "geoms": geoms}


def finger_collision_contract(mode: str) -> dict:
    """Resolved robot-owned geometry recipe, without changing global defaults."""
    if mode not in FINGER_COLLISION_MODES:
        raise ValueError(f"Unknown ARX5 finger collision geometry: {mode!r}")
    result = {"mode": mode, "source_meshes": ["link7.obj", "link8.obj"]}
    if mode == "convex_decomposition":
        result["decomposition"] = {}
        directory = files(__package__).joinpath("assets", "collision")
        for body in ("link7", "link8"):
            manifest = json.loads(directory.joinpath(f"{body}.json").read_text(encoding="utf-8"))
            if manifest.get("qa_status") != "passed":
                raise ValueError(
                    f"Unaccepted collision decomposition for {body}; finish offline QA"
                )
            parts = manifest["parts"]
            expected = [f"{body}_part_{index:03d}.obj" for index in range(len(parts))]
            if not parts or [part["file"] for part in parts] != expected:
                raise ValueError(f"Invalid ordered convex-piece manifest for {body}")
            if manifest["coordinate_frame"] != "unchanged_source_body_frame":
                raise ValueError(f"Collision parts must preserve the {body} source frame")
            if manifest["source_mesh"] != f"{body}.obj" or manifest["source_scale"] != [1, 1, 1]:
                raise ValueError(f"Collision parts must match the unscaled {body} visual mesh")
            result["decomposition"][body] = {
                key: manifest[key]
                for key in (
                    "source_mesh",
                    "source_scale",
                    "coordinate_frame",
                    "parameters",
                    "qa_criterion",
                    "parts",
                )
            }
    return result


def _apply_finger_collision_geometry(spec, mode):
    contract = finger_collision_contract(mode)
    if mode == "source_mesh_convex_hull":
        return
    # A nonempty VFS makes nested MjSpec.attach prefix mesh filenames. Include
    # the existing XML meshes too, so mocap wrapping followed by Scene.attach
    # does not leave their newly prefixed filenames pointing only at disk.
    mesh_directory = Path(spec.modelfiledir) / spec.meshdir
    for mesh in spec.meshes:
        if mesh.file:
            spec.assets[mesh.file] = (mesh_directory / mesh.file).read_bytes()
    directory = files(__package__).joinpath("assets", "collision")
    for body, manifest in contract["decomposition"].items():
        original = spec.geom(f"{body}_collision")
        # These are copied from the existing geom, never separately tuned.
        fields = (
            "type",
            "pos",
            "quat",
            "contype",
            "conaffinity",
            "condim",
            "priority",
            "friction",
            "solmix",
            "solref",
            "solimp",
            "margin",
            "gap",
            "mass",
            "density",
            "typeinertia",
            "fluid_ellipsoid",
            "fluid_coefs",
            "material",
            "rgba",
            "group",
        )
        values = {
            name: getattr(original, name).copy()
            if isinstance(getattr(original, name), np.ndarray)
            else getattr(original, name)
            for name in fields
        }
        # The MjsGeom getter exposes this integer mode as a float, while the
        # add_geom binding requires an integer argument. Preserve its value.
        values["fluid_ellipsoid"] = int(values["fluid_ellipsoid"])
        for index, part in enumerate(manifest["parts"]):
            filename = part["file"]
            meshname = f"{body}_collision_mesh_{index:03d}"
            spec.assets[filename] = directory.joinpath(filename).read_bytes()
            spec.add_mesh(name=meshname, file=filename)
            if index == 0:
                original.meshname = meshname
            else:
                spec.body(body).add_geom(
                    name=f"{body}_part_{index:03d}_collision", meshname=meshname, **values
                )


def robot_spec(*, finger_collision_geometry="source_mesh_convex_hull") -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    _apply_finger_collision_geometry(spec, finger_collision_geometry)
    for key in list(spec.keys):
        spec.delete(key)
    # The reusable model starts at its base; the task supplies its world pose.
    spec.body(ROOT_BODY).pos[:] = (0.0, 0.0, 0.0)
    # Only SimulationCfg owns world physics settings after spec attachment.
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    return spec


def robot_cfg(
    layout: dict | None = None,
    *,
    finger_collision_geometry: str = "source_mesh_convex_hull",
) -> EntityCfg:
    """Build a fresh robot entity, optionally mounted by a scene layout."""
    if finger_collision_geometry not in FINGER_COLLISION_MODES:
        raise ValueError(f"Unknown ARX5 finger collision geometry: {finger_collision_geometry!r}")
    from mjlab.actuator.xml_actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    def placed_spec():
        spec = robot_spec(finger_collision_geometry=finger_collision_geometry)
        # Render the visual meshes; collision proxies keep their physical masks.
        for geom in spec.geoms:
            if geom.name.endswith("_collision"):
                geom.group = 3
        return spec

    pose = np.eye(4) if layout is None else np.asarray(layout["robot"]["T_robot_to_world"])
    return EntityCfg(
        spec_fn=placed_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(pose[:3, 3]),
            rot=rotation_quaternion(pose[:3, :3]),
            joint_pos=dict(zip(JOINTS, HOME, strict=True)),
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=("joint[1-6]",)),
                XmlActuatorCfg(target_names_expr=("joint[78]",)),
            ),
        ),
    )
