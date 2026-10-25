"""BoxInBin: carry a box into a bin, release it and let it rest."""

from clab.tasks.base import ManipulationTask

from .config import make_env_cfg
from .resets import ResetFromBank
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


class BoxInBinTask(ManipulationTask):
    name = "box_in_bin"
    label = "BoxInBin"
    robot = DEFAULT_ROBOT
    reset_types = RESET_TYPES
    reset_class = ResetFromBank
    make_env_cfg = staticmethod(make_env_cfg)
    supermimic_adapter = "BoxInBinAdapter"
    ppo_capacity = (24, 128)


TASK = BoxInBinTask()
