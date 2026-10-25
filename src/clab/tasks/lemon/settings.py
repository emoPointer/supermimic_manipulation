"""Lemon data-generation objective and reset categories."""

RESET_TYPES = (
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAirborneEEGrasped",
    "ObjectNearGoalEEGrasped",
)
EPISODE_SECONDS = 32.0
# Success hold: 10 successive checks at 20 Hz.
STABLE_SECONDS = 0.5
SUCCESS_XY_TOLERANCE = 0.045
SUCCESS_MIN_HEIGHT = 0.005
SUCCESS_MAX_HEIGHT = 0.12
