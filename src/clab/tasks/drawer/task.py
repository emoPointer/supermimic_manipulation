"""Drawer: put a pod into an opened drawer, release it and push the drawer closed."""

from clab.tasks.base import ManipulationTask

from .config import make_env_cfg
from .resets import ResetFromBank
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


class DrawerTask(ManipulationTask):
    name = "drawer"
    label = "Drawer"
    robot = DEFAULT_ROBOT
    reset_types = RESET_TYPES
    reset_class = ResetFromBank
    make_env_cfg = staticmethod(make_env_cfg)
    supermimic_adapter = "DrawerAdapter"


TASK = DrawerTask()
