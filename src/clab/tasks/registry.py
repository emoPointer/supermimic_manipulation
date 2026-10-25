"""Task discovery: every package under ``clab.tasks`` that contains ``task.py``."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


@cache
def task_names() -> tuple[str, ...]:
    return tuple(sorted(path.parent.name for path in _ROOT.glob("*/task.py")))


def load_task(name: str):
    if name not in task_names():
        raise ValueError(f"Unknown task {name!r}; choose one of {task_names()}")
    return import_module(f"clab.tasks.{name}.task").TASK


def register_mjlab_tasks() -> None:
    """Register every task with mjlab once, for its train/play tools and PPO runner."""
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    from clab.algorithms.ppo.runner import ClabPpoRunner

    registered = set(list_tasks())
    for name in task_names():
        task = load_task(name)
        if task.task_id not in registered:
            register_mjlab_task(
                task_id=task.task_id,
                env_cfg=task.make_env_cfg(),
                play_env_cfg=task.make_env_cfg(play=True),
                rl_cfg=task.make_ppo_cfg(),
                runner_cls=ClabPpoRunner,
            )
