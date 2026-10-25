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
        from clab.simulation import configure_cuda_graph

        from .supermimic.inference import load_policy as load_supermimic

        policy = load_supermimic(
            env,
            checkpoint,
            mdp_contract=task.supermimic_contract(env),
            task_name=task.name,
            training_env_config=training_env_config,
        )
        # Physics graphs only; every evaluation batch resets the environment itself.
        configure_cuda_graph(env, cuda_graph_mode)
        return policy
    raise ValueError(f"Unsupported algorithm {algorithm!r}; choose one of {ALGORITHMS}")
