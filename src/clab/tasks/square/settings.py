"""Square assembly tolerances and task-specific reset curriculum."""

import math
from dataclasses import dataclass, fields

RESET_TYPES = (
    "NutRestingEEAnywhere",
    "NutRestingHandleGrasped",
    "NutAirborneHandleGrasped",
    "NutPartiallyInsertedHandleGrasped",
    # Open hand just above the handle of a resting nut, derived physically from
    # NutRestingHandleGrasped rows (clab.tasks.pregrasp): the rung between a free
    # gripper and a held handle. Last, so the collected categories keep their indices.
    "NutRestingEEPregrasp",
)
# Categories the collector simulates directly; the last one is derived from them.
COLLECTED_TYPES = RESET_TYPES[:-1]
# Grasped categories start from a collector-verified legal handle grasp
# (unbroken legal contact); success counts it as observed in the episode.
INHERITED_HANDLE_GRASP = tuple(kind.endswith("HandleGrasped") for kind in RESET_TYPES)
EPISODE_SECONDS = 16.0
DESCENT_STEPS = 120  # Six seconds: the clipped controller must catch the moving target.
STABLE_SECONDS = 0.5


@dataclass(frozen=True)
class AssemblyCriteria:
    """Explicit new-task rules, serialized in the success term's configuration.

    Distances are metres, speeds m/s and rad/s, forces newtons. The historical
    source XY/height predicate above is retained only for compatibility tests.
    These tolerances are design choices.
    """

    stable_seconds: float = STABLE_SECONDS
    seated_height_tolerance: float = 0.002
    max_tilt_degrees: float = 5.0
    linear_speed_max: float = 0.01
    angular_speed_max: float = 0.1
    support_force_min: float = 0.05
    finger_force_min: float = 0.05
    max_penetration: float = 0.001
    handle_clearance: float = 0.001
    contact_tolerance: float = 0.002

    def __post_init__(self):
        if any(
            not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) <= 0
            for f in fields(self)
        ):
            raise ValueError("Square assembly tolerances must be finite and positive")
        if self.max_tilt_degrees >= 90:
            raise ValueError("Square seated tilt must be below 90 degrees")
