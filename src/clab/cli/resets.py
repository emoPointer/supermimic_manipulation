"""``clab resets collect``: collect a task's reset bank; ``pregrasp`` derives a category."""

from __future__ import annotations

import argparse

from clab.tasks.registry import load_task, task_names


def collect(argv: list[str], prog: str) -> None:
    """Dispatch to the task's own collector, which defines the remaining options."""
    parser = argparse.ArgumentParser(prog=prog, add_help=False)
    parser.add_argument("--task", choices=task_names())
    args, rest = parser.parse_known_args(argv)
    if args.task is None:
        tasks = ", ".join(task_names())
        print(f"usage: {prog} --task {{{tasks}}} --output NEW_BANK.pt [task options]")
        print(f"Run '{prog} --task TASK --help' for a task's collection options.")
        raise SystemExit(0 if {"-h", "--help"} & set(rest) else 2)
    load_task(args.task).module("collect").main(rest, prog=f"{prog} --task {args.task}")


def pregrasp(argv: list[str], prog: str) -> None:
    """Append the open-gripper pre-grasp category to a bank collected without it."""
    from clab.tasks.pregrasp import main

    main(argv, prog=prog)
