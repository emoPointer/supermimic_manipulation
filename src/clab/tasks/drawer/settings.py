"""Drawer collection objective; process and final success are distinct checks."""

# Snapshots of one complete physical trajectory, in order. Intermediate stages
# shorten the credit-assignment distance of this long-horizon task.
RESET_TYPES = (
    "DrawerOpenedPodOnTable",
    "DrawerOpenedPodGraspedOnTable",
    "DrawerOpenedPodAirborneGrasped",
    "PodHeldAboveOpenDrawer",
    "PodReleasedInOpenDrawer",
    "DrawerHalfClosed",
)
# Stable grasp/lift and inside release. These are inherited witnesses,
# never policy accomplishments in the new episode.
PREFIX_FLAGS = ((0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 1))
# The half-closed snapshot is taken when the pushed slide first reaches this.
HALF_CLOSED_SLIDE = 0.07
# Preserve the finite 1200-step/20 Hz budget after removing opening. This is
# not a claim that every initial state or policy can finish within one minute.
EPISODE_SECONDS = 60.0
GRASP_SECONDS = 0.3
RELEASE_SECONDS = 0.3
SUCCESS_SECONDS = 0.5
MIN_CLOSE_DELTA = 0.015
MIN_POD_LIFT = 0.04
SLIDE_RANGE = (0.0, 0.14)
INITIAL_SLIDE_RANGE = (0.13, 0.13)
