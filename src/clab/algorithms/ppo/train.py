"""Launch mjlab's PPO training loop for a task with explicit settings."""

from __future__ import annotations

import os
import re
from pathlib import Path


def train(
    task,
    *,
    reset_bank: Path,
    num_envs: int = 24576,
    iterations: int = 40000,
    save_interval: int = 100,
    seed: int = 42,
    nconmax: int | None = None,
    njmax: int | None = None,
    run_name: str | None = None,
    log_root: Path = Path("logs/rsl_rl"),
    resume: Path | None = None,
    cuda_graph_mode: str = "auto",
    wandb_entity: str | None = None,
    gpu_ids: tuple[int, ...] = (0,),
    reward_source: str = "native",
) -> None:
    """Train in ``LOG_ROOT/<experiment>/<timestamp>[_RUN_NAME]``; checkpoints stay local.

    ``resume`` continues an exact checkpoint of the same experiment, restoring
    networks, normalizers and Adam (not simulation or RNG state); ``iterations``
    then counts additional updates. TensorBoard is always written; W&B metrics are
    added with ``wandb_entity``. ``reward_source="imagined"`` trains on the task's
    SuperMimic MDP reward of each real transition instead of the native reward.
    """
    from mjlab.scripts.train import TrainConfig, launch_training

    from clab.tasks.registry import register_mjlab_tasks

    if min(num_envs, iterations, save_interval) < 1:
        raise ValueError("Environment count, iterations and save interval must be positive")
    if reward_source not in ("native", "imagined"):
        raise ValueError(f"Unknown PPO reward source: {reward_source}")
    if run_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("--run-name may contain only letters, digits, '.', '_' and '-'")
    env = task.make_env_cfg(reset_bank=Path(reset_bank).resolve(strict=True))
    env.scene.num_envs = num_envs
    if reward_source == "imagined":
        # Read-only POST-step capture, as in SuperMimic training.
        task.configure_supermimic(env)
    default_nconmax, default_njmax = task.ppo_capacity or (env.sim.nconmax, env.sim.njmax)
    env.sim.nconmax = default_nconmax if nconmax is None else nconmax
    env.sim.njmax = default_njmax if njmax is None else njmax
    agent = task.make_ppo_cfg()
    agent.seed = seed
    agent.max_iterations = iterations
    agent.save_interval = save_interval
    agent.run_name = run_name or ""
    agent.cuda_graph_mode = cuda_graph_mode
    agent.upload_model = False
    agent.reward_source = reward_source
    agent.task_name = task.name
    agent.logging_backend = "both" if wandb_entity else "tensorboard"
    if wandb_entity:
        os.environ.update(WANDB_ENTITY=wandb_entity, WANDB_MODE="online")
    log_root = Path(log_root).resolve()
    if resume is not None:
        resume = Path(resume).resolve(strict=True)
        experiment = log_root / agent.experiment_name
        if resume.parent.parent != experiment:
            raise ValueError(f"--resume must be a checkpoint of a run in {experiment}")
        # mjlab selects runs and checkpoints by regex; anchor the exact names.
        agent.resume = True
        agent.load_run = f"^{re.escape(resume.parent.name)}$"
        agent.load_checkpoint = f"^{re.escape(resume.name)}$"
    register_mjlab_tasks()
    launch_training(
        task.task_id,
        TrainConfig(env=env, agent=agent, log_root=str(log_root), gpu_ids=list(gpu_ids)),
    )
