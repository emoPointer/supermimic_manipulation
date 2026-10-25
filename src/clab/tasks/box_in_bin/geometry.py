"""BoxInBin binding for shared physical-tool clearance queries."""

from clab.robots.tool_geometry import safe_rotation_fraction
from clab.robots.tool_geometry import tool_geometry_points as _tool_geometry_points

from .variants import DEFAULT_ROBOT


def tool_geometry_points(model, finger_open):
    return _tool_geometry_points(model, finger_open, robot=DEFAULT_ROBOT)


__all__ = ["safe_rotation_fraction", "tool_geometry_points"]
