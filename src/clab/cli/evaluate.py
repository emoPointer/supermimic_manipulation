"""``clab eval policy``: complete-episode evaluation of a learned checkpoint.

Inputs are never modified and existing reports are never replaced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clab.tasks.registry import load_task, task_names


def policy(argv: list[str], prog: str) -> None:
    from clab.algorithms.inference import ALGORITHMS

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evaluate a learned checkpoint over complete episodes of every reset category.",
    )
    parser.add_argument("--task", required=True, choices=task_names())
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="ppo")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--episodes-per-type", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split-label",
        choices=("development", "model_selection", "held_out"),
        default="development",
    )
    parser.add_argument("--training-env-config", type=Path, help="Default: RUN/params/env.yaml")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain", "substeps"), default="auto")
    args = parser.parse_args(argv)
    from clab.evaluation.policy import evaluate_policy

    task = load_task(args.task)
    evaluate_policy(task, **{k: v for k, v in vars(args).items() if k != "task"})
