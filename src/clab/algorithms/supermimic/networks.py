"""SuperMimic-style networks; task-owned adapters define dynamics coordinates.

Adapted from zita-ch/supermimic_vibed. See REFERENCE_LICENSE for attribution.
Parameters, normalization, integration and analytic rewards remain FP32. Explicit
BF16 autocast is confined to each neural MLP and returns FP32 outputs.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.modules import EmpiricalNormalization
from torch import nn

from .config import SuperMimicConfig

ACTOR_INPUT_NORMALIZATION = "rsl_rl_empirical_real_next_obs"


def mlp(inp, hidden, out, *, activation=nn.ELU, layer_norm=False):
    layers = []
    for width in hidden:
        layers.append(nn.Linear(inp, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        inp = width
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


def run_mlp(net, x: torch.Tensor, amp: str) -> torch.Tensor:
    """Reference-style MLP-only autocast, without mutable process-wide precision."""
    if amp == "none":
        return net(x)
    if amp != "bf16":
        raise ValueError(f"Unsupported MLP autocast mode: {amp!r}")
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
        return net(x).float()


class RunningNorm(nn.Module):
    """Parallel running moments with explicit updates and checkpointed buffers."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        if dim < 1 or not math.isfinite(max_count) or max_count <= 0:
            raise ValueError("Normalizer dimension and max_count must be positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Normalizer eps must be positive and finite")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.ndim < 2 or x.shape[-1] != self.mean.numel():
            raise ValueError("Normalizer input must end in its declared feature dimension")
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if x.shape[0] == 0 or not torch.isfinite(x).all():
            raise ValueError("Normalizer updates require nonempty finite samples")
        if self.count >= self.max_count:
            return
        count = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = self.count + count
        delta = mean - self.mean
        combined_var = (
            self.var * self.count + var * count + delta.square() * self.count * count / total
        ) / total
        combined_mean = self.mean + delta * count / total
        if not torch.isfinite(combined_var).all() or not torch.isfinite(combined_mean).all():
            raise FloatingPointError("Normalizer moments overflowed; previous statistics preserved")
        self.mean.copy_(combined_mean)
        self.var.copy_(combined_var.clamp_min(0))
        self.count.copy_(total)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()


class WorldModel(nn.Module):
    """Normalized increment prediction followed by task-owned state integration."""

    def __init__(
        self, mdp, hidden=SuperMimicConfig.wm_hidden, normalizer_max_count=2e8, *, amp="none"
    ):
        super().__init__()
        self.mdp = mdp
        self.amp = amp
        self.net = mlp(mdp.input_dim, hidden, mdp.delta_dim, activation=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(mdp.input_dim, normalizer_max_count)
        self.out_norm = RunningNorm(mdp.delta_dim, normalizer_max_count, eps=1e-4)

    def step(self, state, action, context):
        features = self.mdp.inputs(state, action, context)
        delta = run_mlp(self.net, self.in_norm(features), self.amp)
        delta = delta * self.out_norm.std() + self.out_norm.mean
        return self.mdp.integrate(state, delta, action, context)


class WorldModelEnsemble(nn.Module):
    """K independently initialized increment MLPs sharing normalizer statistics.

    Members never share parameters. ``step(member=k)`` uses one member; without
    ``member`` it integrates the mean normalized prediction. ``imagine`` advances
    every sample with its own member index and returns the members'
    disagreement: the std over members of normalized increments, averaged over
    increment dimensions (dimensionless, in units of the target std).
    """

    def __init__(
        self,
        mdp,
        hidden=SuperMimicConfig.wm_hidden,
        normalizer_max_count=2e8,
        *,
        amp="none",
        members=2,
    ):
        super().__init__()
        if type(members) is not int or members < 2:
            raise ValueError("A world-model ensemble needs at least two members")
        self.mdp, self.amp, self.members = mdp, amp, members
        self.nets = nn.ModuleList()
        for _ in range(members):
            net = mlp(mdp.input_dim, hidden, mdp.delta_dim, activation=nn.SiLU, layer_norm=True)
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets.append(net)
        self.in_norm = RunningNorm(mdp.input_dim, normalizer_max_count)
        self.out_norm = RunningNorm(mdp.delta_dim, normalizer_max_count, eps=1e-4)

    def _predict(self, state, action, context):
        features = self.in_norm(self.mdp.inputs(state, action, context))
        return torch.stack([run_mlp(net, features, self.amp) for net in self.nets])

    def _integrate(self, state, normalized, action, context):
        delta = normalized * self.out_norm.std() + self.out_norm.mean
        return self.mdp.integrate(state, delta, action, context)

    def step(self, state, action, context, member=None):
        if member is None:
            normalized = self._predict(state, action, context).mean(0)
        else:
            features = self.in_norm(self.mdp.inputs(state, action, context))
            normalized = run_mlp(self.nets[member], features, self.amp)
        return self._integrate(state, normalized, action, context)

    def imagine(self, state, action, context, index):
        predictions = self._predict(state, action, context)
        disagreement = predictions.std(0).mean(-1)
        chosen = predictions.gather(0, index[None, :, None].expand(1, *predictions.shape[1:]))[0]
        return self._integrate(state, chosen, action, context), disagreement


class Policy(nn.Module):
    """Deterministic decoder over the task's existing observation/history vector.

    Observation normalization reuses the original PPO EmpiricalNormalization.
    Statistics update explicitly from real returned observations, not forward.
    No output squashing is added: native action processing, including saturation
    and the binary gripper, remains the controller's contract.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden=(2048, 1024, 512, 256),
        *,
        amp="none",
    ):
        super().__init__()
        self.amp = amp
        self.obs_norm = EmpiricalNormalization(obs_dim)
        self.net = mlp(obs_dim, hidden, action_dim)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, obs):
        return run_mlp(self.net, self.obs_norm(obs), self.amp)

    def validate_normalization(self) -> None:
        """Reject corrupt saved moments without repairing or migrating them."""
        norm = self.obs_norm
        if any(not bool(torch.isfinite(value).all()) for value in norm.state_dict().values()):
            raise ValueError("Actor normalization statistics must be finite")
        if bool((norm._var < 0).any()) or bool((norm._std < 0).any()) or norm.count.item() < 0:
            raise ValueError("Actor normalization statistics must be nonnegative")
        if not torch.allclose(norm._std.square(), norm._var):
            raise ValueError("Actor normalization standard deviation and variance are inconsistent")


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 128), normalizer_max_count=2e8, *, amp="none"):
        super().__init__()
        self.amp = amp
        self.net = mlp(input_dim, hidden, 1)
        self.norm = RunningNorm(input_dim, normalizer_max_count)

    def forward(self, features):
        return run_mlp(self.net, self.norm(features), self.amp).squeeze(-1)
