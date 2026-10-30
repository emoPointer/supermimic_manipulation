"""MugHang: hang a mug on a tree by its handle and release it."""

from clab.tasks.base import ManipulationTask

from .config import make_env_cfg
from .resets import ResetFromBank
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


class MugHangTask(ManipulationTask):
    name = "mug_hang"
    label = "MugHang"
    robot = DEFAULT_ROBOT
    reset_types = RESET_TYPES
    reset_class = ResetFromBank
    make_env_cfg = staticmethod(make_env_cfg)
    supermimic_adapter = "MugHangAdapter"
    # PPO's 24576 worlds do not fit the environment's capacities; measured at most
    # 61 contacts per world, 234 rows (rollouts at 4096 worlds), with a threefold margin.
    ppo_capacity = (184, 704)


TASK = MugHangTask()
