"""Lemon: place a lemon on a plate, release it and let it rest."""

from clab.tasks.base import ManipulationTask

from .config import make_env_cfg
from .resets import ResetFromBank
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


class LemonTask(ManipulationTask):
    name = "lemon"
    label = "Lemon"
    robot = DEFAULT_ROBOT
    reset_types = RESET_TYPES
    reset_class = ResetFromBank
    make_env_cfg = staticmethod(make_env_cfg)
    supermimic_adapter = "LemonAdapter"
    # PPO's 24576 worlds do not fit the environment's capacities; measured at most
    # 2.5 contacts per world, 28 rows (rollouts at 4096 worlds), with a threefold margin
    # (nconmax at least the initial model's 10 contacts, which mujoco_warp requires).
    ppo_capacity = (16, 96)


TASK = LemonTask()
