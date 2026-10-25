"""Shared task-object placement boundary in the pad-aligned world frame.

XY_MIN/MAX constrain complete object geometry for existing reset-bank paths.
SCRIPTED_CENTER_XY_MIN/MAX separately constrain BoxInBin/Lemon scripted initial
root centers; their geometry may extend outside that rectangle. Neither range
certifies robot reachability.
"""

XY_MIN = (0.2, -0.2)
XY_MAX = (0.5, 0.3)

# Explicit scripted initial-center distribution used only by BoxInBin and Lemon.
# The complete-geometry workspace above remains the reset-bank contract.
SCRIPTED_CENTER_XY_MIN = (0.15, -0.2)
SCRIPTED_CENTER_XY_MAX = (0.35, 0.2)


def scripted_center_metadata(entities):
    """Declare initial root-center bounds; visual/collision edges may exceed them."""
    return {
        "pose_frame": "environment_world",
        "proposal_xy_min_m": list(SCRIPTED_CENTER_XY_MIN),
        "proposal_xy_max_m": list(SCRIPTED_CENTER_XY_MAX),
        "center_definition": "object_root_frame_translation",
        "containment": "object_centers_only",
        "complete_geometry_workspace_filter": False,
        "entities": list(entities),
        "units": "m",
    }


def contains_bounds(lower, upper):
    """Contain an XY AABB; supports NumPy arrays and Torch tensors, batched or scalar."""
    return (
        (lower[..., 0] >= XY_MIN[0])
        & (lower[..., 1] >= XY_MIN[1])
        & (upper[..., 0] <= XY_MAX[0])
        & (upper[..., 1] <= XY_MAX[1])
    )


def workspace_metadata():
    return {
        "frame": "world",
        "xy_min": list(XY_MIN),
        "xy_max": list(XY_MAX),
        "containment": "complete_object_geometry",
        "units": "m",
    }


def body_geometry_corners(model, names):
    """Conservative local box enclosing each body's compiled meshes and boxes.

    Includes visual and collision geometry; compiled mesh geom transforms undo
    MuJoCo's mesh recentering. Unsupported geometry fails rather than vanishing.
    """
    from itertools import product

    import mujoco
    import numpy as np

    signs = np.array(tuple(product((-1.0, 1.0), repeat=3)))
    result = {}
    for name in names:
        body = model.body(f"{name}/{name}").id
        points = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start : start + count]
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                vertices = signs * model.geom_size[geom]
            else:
                raise ValueError(
                    f"Unsupported workspace geometry for {name}: {model.geom_type[geom]}"
                )
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            points.append(vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom])
        if not points:
            raise ValueError(f"No workspace geometry for {name}")
        vertices = np.concatenate(points)
        result[name] = np.array(tuple(product(*zip(vertices.min(0), vertices.max(0), strict=True))))
    return result


def poses_within_workspace(poses, corners):
    """Batched environment-world poses; contain full rotated geometry envelopes."""
    from mjlab.utils.lab_api.math import quat_apply

    valid = None
    for name, vertices in corners.items():
        pose = poses[f"{name}_pose"]
        points = (
            quat_apply(
                pose[:, None, 3:].expand(-1, len(vertices), -1),
                pose.new_tensor(vertices)[None].expand(len(pose), -1, -1),
            )
            + pose[:, None, :3]
        )
        inside = contains_bounds(points.amin(1), points.amax(1))
        valid = inside if valid is None else valid & inside
    return valid
