"""Deterministic policy loading for learned checkpoints; no reset or success rules."""

from pathlib import Path

ALGORITHMS = ("ppo", "supermimic")


def load_policy(
    env,
    checkpoint: Path,
    *,
    algorithm: str,
    task,
    cuda_graph_mode: str = "auto",
    training_env_config: Path | None = None,
):
    """Return ``policy(observations) -> actions`` for one of :data:`ALGORITHMS`."""
    if algorithm == "ppo":
        from .ppo.inference import load_policy as load_ppo

        config = task.make_ppo_cfg()
        config.cuda_graph_mode = cuda_graph_mode
        return load_ppo(env, checkpoint, config)
    if algorithm == "supermimic":
        from .supermimic.inference import load_policy as load_supermimic

        if cuda_graph_mode != "auto":
            raise ValueError("SuperMimic evaluation supports only automatic CUDA Graph mode")
        return load_supermimic(
            env,
            checkpoint,
            mdp_contract=task.supermimic_contract(env),
            task_name=task.name,
            training_env_config=training_env_config,
        )
    raise ValueError(f"Unsupported algorithm {algorithm!r}; choose one of {ALGORITHMS}")
