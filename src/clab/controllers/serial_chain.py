"""Kinematic description of one serial-chain link (body pose, hinge axis, mass)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SerialLinkCfg:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    axis: tuple[float, float, float]
    com: tuple[float, float, float]
    mass: float

    def __post_init__(self):
        for name, size in (("position", 3), ("quaternion", 4), ("axis", 3), ("com", 3)):
            value = getattr(self, name)
            if len(value) != size or not all(math.isfinite(x) for x in value):
                raise ValueError(f"Invalid serial-chain {name}")
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Serial-chain link mass must be positive")
        for name in ("axis", "quaternion"):
            if not math.isclose(sum(x * x for x in getattr(self, name)), 1.0, abs_tol=1e-8):
                raise ValueError(f"Serial-chain {name} must be normalized")
