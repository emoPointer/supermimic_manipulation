"""Square: thread a square nut onto its peg by the exposed handle and release it."""

from clab.tasks.base import ManipulationTask

from .config import make_env_cfg
from .resets import ResetFromBank
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


class SquareTask(ManipulationTask):
    name = "square"
    label = "Square"
    robot = DEFAULT_ROBOT
    reset_types = RESET_TYPES
    reset_class = ResetFromBank
    make_env_cfg = staticmethod(make_env_cfg)
    supermimic_adapter = "SquareAdapter"


TASK = SquareTask()
