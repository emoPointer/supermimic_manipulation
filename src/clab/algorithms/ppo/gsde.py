"""RSL-RL PPO actor with state-dependent action variance.

The source PPO draws from a diagonal Normal with variance
`features.square() @ exp(log_std).square() + epsilon`; its exploration-matrix
samples do not enter the action draw. Keep that effective distribution while
using mjlab's current RSL-RL model interface.
"""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal


class GsdeDistribution(Distribution):
    """Diagonal Gaussian whose variance depends on the penultimate actor features."""

    def __init__(self, output_dim: int, feature_dim: int, init_std: float = 1.0) -> None:
        super().__init__(output_dim)
        if init_std <= 0:
            raise ValueError("init_std must be positive")
        self.log_std = nn.Parameter(torch.full((feature_dim, output_dim), float(init_std)).log())
        self._distribution: Normal | None = None

    def update(self, mlp_output: tuple[torch.Tensor, torch.Tensor]) -> None:
        mean, features = mlp_output
        variance = features.square() @ self.log_std.exp().square()
        self._distribution = Normal(mean, torch.sqrt(variance + 1e-6), validate_args=False)

    def _current(self) -> Normal:
        if self._distribution is None:
            raise RuntimeError("gSDE distribution has not been updated")
        return self._distribution

    def sample(self) -> torch.Tensor:
        return self._current().sample()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        return nn.Identity()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._current().mean

    @property
    def std(self) -> torch.Tensor:
        return self._current().stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._current().entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return self.mean, self.std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._current().log_prob(outputs).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old = Normal(old_mean, old_std, validate_args=False)
        new = Normal(new_mean, new_std, validate_args=False)
        return torch.distributions.kl_divergence(old, new).sum(dim=-1)


class GsdeMLPModel(MLPModel):
    """RSL-RL MLP actor with the original state-dependent exploration law."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128, 64),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        if distribution_cfg is None:
            raise ValueError("gSDE actor requires a distribution configuration")
        dist_cfg = distribution_cfg.copy()
        if dist_cfg.pop("class_name", None) != "GsdeDistribution":
            raise ValueError("gSDE actor requires GsdeDistribution")
        init_std = dist_cfg.pop("init_std", 1.0)
        if dist_cfg:
            raise ValueError(f"Unknown gSDE distribution settings: {sorted(dist_cfg)}")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )
        final_layer = self.mlp[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("gSDE requires a linear final actor layer")
        self.distribution = GsdeDistribution(output_dim, final_layer.in_features, init_std)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs, masks, hidden_state)
        features = latent
        for layer_index in range(len(self.mlp) - 1):
            features = self.mlp[layer_index](features)
        mean = self.mlp[-1](features)
        if not stochastic_output:
            return mean
        self.distribution.update((mean, features))
        return self.distribution.sample()
