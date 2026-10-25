"""``clab train``: PPO and SuperMimic training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from clab.tasks.registry import load_task, task_names


def ppo(argv: list[str], prog: str) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Train a task's state policy with PPO (mjlab / RSL-RL)."
    )
    parser.add_argument("--task", required=True, choices=task_names())
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=24576)
    parser.add_argument(
        "--iterations", type=int, default=40000, help="PPO iterations (additional on --resume)"
    )
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nconmax", type=int, help="Contacts per world (task training default)")
    parser.add_argument("--njmax", type=int, help="Constraints per world (task training default)")
    parser.add_argument("--run-name", help="Suffix of the timestamped run directory")
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument(
        "--resume", type=Path, help="Continue from LOG_ROOT/<experiment>/RUN/model_N.pt"
    )
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    parser.add_argument("--wandb-entity", help="Also log metrics to this Weights & Biases entity")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    args = parser.parse_args(argv)
    from clab.algorithms.ppo.train import train

    train(
        load_task(args.task),
        reset_bank=args.reset_bank,
        num_envs=args.num_envs,
        iterations=args.iterations,
        save_interval=args.save_interval,
        seed=args.seed,
        nconmax=args.nconmax,
        njmax=args.njmax,
        run_name=args.run_name,
        log_root=args.log_root,
        resume=args.resume,
        cuda_graph_mode=args.cuda_graph_mode,
        wandb_entity=args.wandb_entity,
        gpu_ids=tuple(args.gpu_ids),
    )


def supermimic(argv: list[str], prog: str) -> None:
    from clab.algorithms.supermimic.config import SuperMimicConfig
    from clab.algorithms.supermimic.runner import RunConfig

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train with SuperMimic: backpropagate the task return through a learned "
        "world model. Every algorithm/run setting below defaults to its dataclass value.",
    )
    parser.add_argument("--task", default="box_in_bin", choices=task_names())
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nconmax", type=int)
    parser.add_argument("--njmax", type=int)
    parser.add_argument("--resume", type=Path, help="Approximate continuation from a checkpoint")
    parser.add_argument("--logger", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny networks and budgets; checks save and reload"
    )
    parser.add_argument(
        "--adapter-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Task adapter imagination setting (all tasks: stable_linear_speed, "
        "stable_angular_speed, release_opening, release_weight, success_gradient_scale; "
        "see docs/supermimic.md for task-specific settings)",
    )
    for field, default in asdict(RunConfig()).items():
        _add_field(parser, field, default)
    for field, default in asdict(SuperMimicConfig()).items():
        _add_field(parser, field, default, choices=("none", "bf16") if field == "amp" else None)
    args = parser.parse_args(argv)
    from clab.algorithms.supermimic.train import smoke_profile, train

    algorithm, run = SuperMimicConfig(), RunConfig()
    if args.smoke:
        algorithm, run = smoke_profile(algorithm, run)
    algorithm = replace(algorithm, **_overrides(args, algorithm))
    run = replace(run, **_overrides(args, run))
    options = {}
    for item in args.adapter_option:
        name, separator, value = item.partition("=")
        if not separator:
            parser.error(f"--adapter-option expects NAME=VALUE, got {item!r}")
        options[name] = float(value)
    train(
        load_task(args.task),
        reset_bank=args.reset_bank,
        output=args.output,
        algorithm=algorithm,
        run=run,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        nconmax=args.nconmax,
        njmax=args.njmax,
        resume=args.resume,
        logger=args.logger,
        cuda_graph_mode=args.cuda_graph_mode,
        smoke=args.smoke,
        adapter_options=options,
    )


def _add_field(parser, field: str, default, choices=None) -> None:
    flag = "--" + field.replace("_", "-")
    if type(default) is bool:
        parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
    elif isinstance(default, tuple):
        parser.add_argument(flag, type=type(default[0]), nargs="+")
    else:
        parser.add_argument(flag, type=type(default), choices=choices)


def _overrides(args, config) -> dict:
    values = {}
    for field, default in asdict(config).items():
        value = getattr(args, field, None)
        if value is not None:
            values[field] = tuple(value) if isinstance(default, tuple) else value
    return values
