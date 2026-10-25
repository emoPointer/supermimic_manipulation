"""BoxInBin dimensions, workspace and reset categories.

Five actor frames span 0.2 s from oldest to current.
"""

import math

RESET_TYPES = (
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
)
EPISODE_SECONDS = 16.0
BOX_HALF_EXTENTS = (0.0565, 0.0225, 0.0505)
BIN_OUTER_HALF_EXTENTS = (0.112, 0.077, 0.0195)
BOX_RADIUS = math.hypot(BOX_HALF_EXTENTS[0], BOX_HALF_EXTENTS[1])
BIN_RADIUS = math.hypot(0.224 / 2, 0.154 / 2)
BOX_CENTER_X_RANGE = (0.075 + BOX_RADIUS, 0.455 - BOX_RADIUS)
BOX_CENTER_Y_RANGE = (-0.22 + BOX_RADIUS, 0.26 - BOX_RADIUS)
BIN_INNER_HALF_EXTENTS = (0.097, 0.062)
BIN_CENTRE_Z = 0.0205
BOX_CENTRE_Z = 0.0515
STABLE_SECONDS = 0.5
# Allow 10% contact-force fluctuation while requiring nearly full weight support.
SUPPORT_WEIGHT_FRACTION = 0.9
