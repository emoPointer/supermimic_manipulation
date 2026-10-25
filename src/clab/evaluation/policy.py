"""Evaluate a learned checkpoint over complete episodes of every reset category."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import dump_yaml

from clab.algorithms.inference import load_policy
from clab.contracts import checkpoint_environment

from .episodes import complete_episodes, isolated_rng, summarize_episodes


def evaluate_policy(
    task,
    *,
    algorithm: str,
    checkpoint: Path,
    reset_bank: Path,
    device: str = "cuda:0",
    seeds: list[int] = (123,),
    num_envs: int = 1,
    episodes_per_type: int = 0,
    output: Path,
    split_label: str = "development",
    training_env_config: Path | None = None,
    cuda_graph_mode: str = "auto",
) -> dict:
    """Write a complete-episode report to ``output``.

    Each reset category gets ``episodes_per_type`` complete episodes per seed. The
    deterministic actor runs without observation noise; startup randomization and
    every automatic reset stay inside the isolated evaluation random stream.
    """
    seeds = list(seeds)
    if episodes_per_type < 1:
        raise ValueError("Evaluation needs --episodes-per-type of at least one")
    if num_envs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation needs a positive environment count and distinct seeds")
    if output.exists() or output.with_suffix(".env.yaml").exists():
        raise FileExistsError(f"Refusing to replace report: {output}")
    checkpoint = checkpoint.resolve(strict=True)
    reset_bank = reset_bank.resolve(strict=True)
    saved, training_env_config = checkpoint_environment(checkpoint, training_env_config)
    if saved.get("is_finite_horizon") not in ("true", "false"):
        raise ValueError("Saved environment must declare is_finite_horizon explicitly")
    records, dataset = [], None
    for seed in seeds:
        cfg = task.make_eval_cfg(saved, reset_bank)
        cfg.observations["actor"].enable_corruption = False
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        with isolated_rng(seed):
            env = ManagerBasedRlEnv(cfg=cfg, device=device)
            try:
                reset_term = task.reset_term(env)
                dataset = reset_term.metadata
                task.robot.validate_env(env)

                policy = load_policy(
                    env,
                    checkpoint,
                    algorithm=algorithm,
                    task=task,
                    cuda_graph_mode=cuda_graph_mode,
                    training_env_config=training_env_config,
                )
                for type_id in range(len(task.reset_types)):
                    reset_term.type_override = type_id
                    remaining, batch = episodes_per_type, 0
                    while remaining:
                        count = min(num_envs, remaining)
                        batch_seed = seed + 100_000 * type_id + batch
                        rows = complete_episodes(
                            env,
                            policy,
                            seed=batch_seed,
                            group_ids=task.reset_groups,
                            success_signal=task.episode_success,
                            episode_limit=count,
                        )
                        records.extend(
                            dict(row, seed=seed, reset_seed=batch_seed, world=world)
                            for world, row in enumerate(rows)
                        )
                        remaining -= count
                        batch += 1
            finally:
                env.close()
    actor = cfg.observations["actor"]
    control_dt = cfg.sim.mujoco.timestep * cfg.decimation
    report = {
        "task": task.name,
        "checkpoint": str(checkpoint),
        "policy_kind": algorithm,
        "reset_bank": str(reset_bank),
        "dataset": dataset,
        "protocol": {
            "seeds": seeds,
            "num_envs": num_envs,
            "episodes_per_type_per_seed": episodes_per_type,
            "split_label": split_label,
            "policy_mode": "deterministic",
            "observation_noise": False,
            "startup_randomization": True,
            "training_env_config": str(training_env_config),
            "control_dt": control_dt,
            "actor_history_frames": actor.history_length,
            "actor_history_oldest_to_current_seconds": (actor.history_length - 1) * control_dt,
            "finite_horizon": cfg.is_finite_horizon,
        },
        "metrics": summarize_episodes(records),
        "episodes": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    dump_yaml(output.with_suffix(".env.yaml"), asdict(cfg))
    print(json.dumps(report["metrics"], indent=2))
    return report
