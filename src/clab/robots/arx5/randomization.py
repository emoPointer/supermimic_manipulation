"""Robot physical-parameter randomization mechanisms."""

from __future__ import annotations

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from .description import RIGID_BODIES


def _env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def robot_mass(env, env_ids: torch.Tensor | None, scale_range: tuple[float, float]) -> None:
    """Scale link mass and inertia using the task-selected range."""
    rows = _env_ids(env, env_ids)
    body_ids = torch.tensor(
        [env.sim.mj_model.body(f"robot/{name}").id for name in RIGID_BODIES],
        device=env.device,
        dtype=torch.long,
    )
    scales = torch.empty((len(rows), len(body_ids)), device=env.device).uniform_(*scale_range)
    default_mass = env.sim.get_default_field("body_mass")[body_ids]
    default_inertia = env.sim.get_default_field("body_inertia")[body_ids]
    env.sim.model.body_mass[rows[:, None], body_ids] = scales * default_mass
    env.sim.model.body_inertia[rows[:, None], body_ids] = scales[:, :, None] * default_inertia
