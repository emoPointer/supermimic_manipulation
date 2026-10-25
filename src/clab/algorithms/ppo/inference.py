"""Load native PPO actor weights and frozen observation normalization."""

from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from .runner import ClabPpoRunner


def load_policy(env, checkpoint: Path, config):
    wrapper = RslRlVecEnvWrapper(env)
    agent_cfg = asdict(config)
    runner = ClabPpoRunner(wrapper, agent_cfg, device=env.device)
    state = torch.load(checkpoint, map_location=env.device, weights_only=True)
    if "actor_state_dict" not in state:
        raise ValueError("Expected a native PPO actor checkpoint")
    # Load only the actor, without restoring optimizer, curriculum or counters.
    runner.alg.load(state, load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy(device=env.device)

    def act(observations):
        return policy(TensorDict(observations, batch_size=[env.num_envs]))

    return act
