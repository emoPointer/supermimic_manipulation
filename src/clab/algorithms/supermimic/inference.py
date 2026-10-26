"""Strict actor-only loading for task-contracted state-based SuperMimic policies.

Loading is read-only: no world model, optimizer, capture hook or simulator step is
created. Evaluation keeps native task resets, actions and success accounting. Only
the actor's interface must match the current task model; imagination-only settings
(surrogate reward and success, world-model state) may differ, unlike resume.
"""

from copy import deepcopy
from pathlib import Path

import torch

from clab.contracts import checkpoint_environment, validate_environment_contract

from .config import SuperMimicConfig
from .networks import ACTOR_INPUT_NORMALIZATION, Policy

# Task-model contract entries that define the actor's interface.
ACTOR_CONTRACT_KEYS = (
    "task",
    "obs_dim",
    "action_dim",
    "actor_history",
    "coordinates",
    "binding",
    "world_model_action",
)


class SuperMimicPolicy:
    """Frozen deterministic actor with the task's unchanged observation budget."""

    def __init__(self, actor, num_envs, obs_dim, action_dim):
        self.actor = actor.eval().requires_grad_(False)
        self.num_envs = num_envs
        self.obs_dim, self.action_dim = obs_dim, action_dim

    @torch.inference_mode()
    def __call__(self, observations):
        obs = observations.get("actor")
        if not isinstance(obs, torch.Tensor) or obs.shape != (self.num_envs, self.obs_dim):
            raise ValueError(
                f"SuperMimic actor input must have shape ({self.num_envs}, {self.obs_dim})"
            )
        if obs.dtype != torch.float32 or not bool(torch.isfinite(obs).all()):
            raise ValueError("SuperMimic actor input must be finite float32")
        action = self.actor(obs)
        if action.shape != (self.num_envs, self.action_dim) or not bool(
            torch.isfinite(action).all()
        ):
            raise RuntimeError("SuperMimic actor produced actions outside the task tensor contract")
        return action


def load_policy(
    env,
    checkpoint: Path,
    *,
    mdp_contract: dict,
    task_name: str,
    training_env_config: Path | None = None,
):
    """Validate saved environment/model semantics, then load actor weights only."""
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    saved_environment, _ = checkpoint_environment(checkpoint, training_env_config)
    comparison_cfg = deepcopy(env.cfg)
    # The common evaluator validates training settings before disabling actor
    # noise. Preserve that existing evaluation allowance without mutating env.cfg.
    corruption = saved_environment.get("observations", {}).get("actor", {}).get("enable_corruption")
    if corruption not in ("true", "false"):
        raise ValueError("Saved actor corruption setting must be explicit")
    comparison_cfg.observations["actor"].enable_corruption = corruption == "true"
    validate_environment_contract(saved_environment, comparison_cfg)
    if not isinstance(mdp_contract, dict) or mdp_contract.get("task") != task_name:
        raise ValueError("SuperMimic requires the selected task's explicit model contract")
    obs_dim, action_dim = mdp_contract.get("obs_dim"), mdp_contract.get("action_dim")
    if any(type(value) is not int or value < 1 for value in (obs_dim, action_dim)):
        raise ValueError("SuperMimic task-model dimensions must be positive integers")
    if (
        env.observation_manager.group_obs_dim["actor"] != (obs_dim,)
        or env.action_manager.total_action_dim != action_dim
    ):
        raise ValueError(
            "SuperMimic task-model observation/action dimensions differ from the environment"
        )
    state = torch.load(checkpoint, map_location=env.device, weights_only=True)
    if not isinstance(state, dict) or state.get("algorithm") != "supermimic":
        raise ValueError("Expected an explicit SuperMimic checkpoint")
    if state.get("task") != task_name:
        raise ValueError("Checkpoint/current SuperMimic task mismatch")
    if state.get("actor_input_normalization") != ACTOR_INPUT_NORMALIZATION:
        raise ValueError("This actor requires matching PPO observation normalization")
    saved_contract = state.get("mdp_contract")
    if not isinstance(saved_contract, dict) or any(
        saved_contract.get(key) != mdp_contract.get(key) for key in ACTOR_CONTRACT_KEYS
    ):
        raise ValueError("Checkpoint/current SuperMimic actor interface mismatch")
    if not isinstance(state.get("config"), dict):
        raise ValueError("SuperMimic checkpoint requires its algorithm configuration")
    config = SuperMimicConfig(**state["config"])
    agent = state.get("agent")
    if not isinstance(agent, dict) or not isinstance(agent.get("policy"), dict):
        raise ValueError("SuperMimic checkpoint lacks its actor state")
    actor = Policy(obs_dim, action_dim, config.policy_hidden, amp=config.amp).to(env.device)
    actor.load_state_dict(agent["policy"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in actor.state_dict().values()):
        raise ValueError("SuperMimic checkpoint has non-finite actor values")
    actor.validate_normalization()
    return SuperMimicPolicy(actor, env.num_envs, obs_dim, action_dim)
