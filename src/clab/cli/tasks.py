"""``clab tasks``: registered tasks with their robot, mjlab ID and reset categories."""

from __future__ import annotations

import argparse

from clab.tasks.registry import load_task, task_names


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    argparse.ArgumentParser(prog=prog, description=__doc__).parse_args(argv)
    for name in task_names():
        task = load_task(name)
        robot = task.robot
        print(f"{name}  ({task.label}, robot {robot.name}, {task.task_id})")
        print(f"    observations: actor {robot.actor_dim}, critic {robot.critic_dim}")
        print(f"    reset types:  {', '.join(task.reset_types)}")
