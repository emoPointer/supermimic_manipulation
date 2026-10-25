"""Complete-episode rollouts with isolated RNG and count-based success summaries."""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    """Restore Python, NumPy, CPU and all initialized CUDA random streams on exit."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        # The environment's seed helper also changes this process setting.
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed


@torch.inference_mode()
def complete_episodes(
    env,
    policy: Callable,
    *,
    seed: int,
    group_ids: Callable,
    success_signal: Callable,
    episode_limit: int | None = None,
    episode_details: Callable | None = None,
    action_limit: int | None = None,
    observer=None,
) -> list[dict]:
    """Record the first full episode per world, ignoring subsequent auto-resets.

    Callers own a separate evaluation environment and a frozen inference policy.
    Success is task-defined, never inferred from termination. Optional episode
    details are distinct named finite scalar tensors of shape [num_envs].
    ``episode_limit``
    deterministically selects the first worlds of a final partial batch.
    ``action_limit`` bounds recorded-action replay without inventing a terminal
    event. Such rows additionally report ``complete``; exhausted recordings may
    be incomplete. The ordinary evaluation path still requires full episodes.
    """
    count = env.num_envs if episode_limit is None else episode_limit
    if not 1 <= count <= env.num_envs:
        raise ValueError("episode_limit must be in [1, num_envs]")
    if action_limit is not None and not 1 <= action_limit <= env.max_episode_length:
        raise ValueError("action_limit must be in [1, max_episode_length]")
    with isolated_rng(seed):
        observations, _ = env.reset(seed=seed)
        if observer is not None:
            observer.on_reset(env, count=count, seed=seed)
        groups = group_ids(env).clone()
        seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        successes = torch.zeros_like(seen)
        returns = torch.zeros(env.num_envs, device=env.device)
        lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        details = [{} for _ in range(count)]
        for _ in range(env.max_episode_length if action_limit is None else action_limit):
            if not all(torch.isfinite(value).all() for value in observations.values()):
                raise RuntimeError("Non-finite evaluation observation")
            action = policy(observations)
            if not torch.isfinite(action).all():
                raise RuntimeError("Non-finite evaluation action")
            observations, reward, terminated, truncated, extras = env.step(action)
            if not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite evaluation reward")
            active = ~seen
            returns += torch.where(active, reward, 0.0)
            lengths += active.long()
            done = active & (terminated | truncated)
            if bool(done.any()):
                successes[done] = success_signal(env, extras)[done]
                if episode_details is not None:
                    terminal_details = episode_details(env, extras)
                    if not isinstance(terminal_details, Mapping):
                        raise TypeError("Episode details must be a mapping of named scalar tensors")
                    reserved = {"group", "success", "return", "length", "complete"}
                    if reserved.intersection(terminal_details):
                        raise ValueError("Episode details cannot replace core outcome fields")
                    for key, value in terminal_details.items():
                        if (
                            not isinstance(key, str)
                            or not isinstance(value, torch.Tensor)
                            or tuple(value.shape) != (env.num_envs,)
                        ):
                            raise ValueError(
                                f"Episode detail {key!r} must be a scalar tensor [num_envs]"
                            )
                        if value.is_complex() or not bool(torch.isfinite(value).all()):
                            raise ValueError(f"Episode detail {key!r} must be finite and real")
                    for i in done[:count].nonzero().flatten().tolist():
                        details[i] = {
                            key: value[i].item() for key, value in terminal_details.items()
                        }
            if observer is not None:
                observer.after_step(
                    env,
                    active=active,
                    done=done,
                    success=successes,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            seen |= done
            if bool(seen[:count].all()):
                break
        if action_limit is None and not bool(seen[:count].all()):
            raise RuntimeError("Evaluation failed to complete every requested episode")
        return [
            {
                "group": int(groups[i]),
                "success": bool(successes[i]),
                "return": float(returns[i]),
                "length": int(lengths[i]),
                **({"complete": bool(seen[i])} if action_limit is not None else {}),
                **details[i],
            }
            for i in range(count)
        ]


def summarize_episodes(episodes: list[dict], *, prefix: str = "eval/") -> dict[str, float]:
    """Aggregate counts, not averages of batch success fractions."""
    if not episodes:
        raise ValueError("No complete episodes to summarize")
    success = sum(row["success"] for row in episodes)
    metrics = {
        f"{prefix}successes": success,
        f"{prefix}episodes": len(episodes),
        f"{prefix}success_rate": success / len(episodes),
        f"{prefix}return_mean": sum(row["return"] for row in episodes) / len(episodes),
        f"{prefix}episode_length_mean": sum(row["length"] for row in episodes) / len(episodes),
    }
    for group in sorted({row["group"] for row in episodes}):
        selected = [row for row in episodes if row["group"] == group]
        successes = sum(row["success"] for row in selected)
        metrics.update(
            {
                f"{prefix}type_{group}_successes": successes,
                f"{prefix}type_{group}_count": len(selected),
                f"{prefix}type_{group}_success_rate": successes / len(selected),
            }
        )
    return metrics
