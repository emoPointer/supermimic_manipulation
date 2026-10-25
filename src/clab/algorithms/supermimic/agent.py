"""World-model fitting, analytic BPTT policy updates and imagined TD(lambda).

The update equations follow zita-ch/supermimic_vibed (REFERENCE_LICENSE).
Unlike the humanoid implementation, the task owns state coordinates, observation
history and analytic reward/termination. No learned reward head is introduced.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn

from .config import SuperMimicConfig
from .networks import Critic, Policy, WorldModel, WorldModelEnsemble


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"SuperMimic non-finite {name}")


@contextmanager
def _fixed_parameters(*modules):
    """Freeze weights, not input derivatives, and restore flags even on failure."""
    parameters = [
        (parameter, parameter.requires_grad) for m in modules for parameter in m.parameters()
    ]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in parameters:
            parameter.requires_grad_(flag)


def lambda_returns(rewards, dones, values, gamma: float, lam: float):
    """Detached-target arithmetic; shapes (H,B), (H,B), (H+1,B).

    ``dones`` includes finite task timeouts. Values must never be taken from an
    automatically reset successor. The caller controls gradient detachment.
    """
    if rewards.shape != dones.shape or values.shape != (rewards.shape[0] + 1, *rewards.shape[1:]):
        raise ValueError("TD(lambda) rewards, dones and values have incompatible shapes")
    result = values[-1]
    targets = []
    for step in range(rewards.shape[0] - 1, -1, -1):
        result = rewards[step] + gamma * (1 - dones[step]) * (
            (1 - lam) * values[step + 1] + lam * result
        )
        targets.append(result)
    return torch.stack(targets[::-1])


class SuperMimicAgent:
    """Algorithm core. The runner owns collection, replay, paths and run metadata."""

    def __init__(self, mdp, cfg: SuperMimicConfig, device):
        cfg.validate()
        self.mdp, self.cfg = mdp, cfg
        self.device = torch.device(device)
        for name in ("state_dim", "delta_dim", "obs_dim", "action_dim", "input_dim", "critic_dim"):
            value = getattr(mdp, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"MDP {name} must be a positive integer")
        if type(mdp.context_dim) is not int or mdp.context_dim < 0:
            raise ValueError("MDP context_dim must be a nonnegative integer")
        self.error_floor = torch.as_tensor(mdp.error_floor, device=self.device, dtype=torch.float32)
        if self.error_floor.shape != (mdp.delta_dim,):
            raise ValueError("MDP error_floor must match delta_dim")
        _finite("error floor", self.error_floor)
        if (self.error_floor <= 0).any():
            raise ValueError("MDP error_floor entries must be strictly positive")
        if cfg.wm_ensemble_size == 1:
            self.wm = WorldModel(mdp, cfg.wm_hidden, cfg.normalizer_max_count, amp=cfg.amp)
        else:
            self.wm = WorldModelEnsemble(
                mdp,
                cfg.wm_hidden,
                cfg.normalizer_max_count,
                amp=cfg.amp,
                members=cfg.wm_ensemble_size,
            )
        self.wm = self.wm.to(self.device)
        self.policy = Policy(mdp.obs_dim, mdp.action_dim, cfg.policy_hidden, amp=cfg.amp).to(
            self.device
        )
        self.critic = Critic(
            mdp.critic_dim, cfg.critic_hidden, cfg.normalizer_max_count, amp=cfg.amp
        ).to(self.device)
        self.wm_optimizer = torch.optim.Adam(self.wm.parameters(), lr=cfg.learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.world_model_updates = self.policy_updates = 0
        # Device-side validity of every world-model update since construction.
        self._world_model_ok = torch.ones((), dtype=torch.bool, device=self.device)
        self._world_model_loss_call = self._world_model_loss
        self._imagine_step_call = self._imagine_step

    def enable_world_model_compile(self, enabled: bool = True) -> None:
        """Select the fixed-shape loss execution path without wrapping model state.

        Compilation is lazy and fail-closed: an unsupported full graph raises on
        its first use rather than silently falling back to eager execution.
        This runtime choice is not an algorithm/checkpoint configuration field.
        """
        if type(enabled) is not bool:
            raise ValueError("World-model compilation must be explicitly enabled or disabled")
        self._world_model_loss_call = (
            torch.compile(self._world_model_loss, fullgraph=True, dynamic=False, mode="default")
            if enabled
            else self._world_model_loss
        )

    def enable_policy_update_compile(self, enabled: bool = True) -> None:
        """Compile one imagined transition (policy, WM step, reward, observation).

        Lazy and fail-closed like the WM path: an unsupported full graph raises
        on first use instead of silently running eager. Not a checkpoint field.
        """
        if type(enabled) is not bool:
            raise ValueError("Policy-update compilation must be explicitly enabled or disabled")
        self._imagine_step_call = (
            torch.compile(self._imagine_step, fullgraph=True, dynamic=False, mode="default")
            if enabled
            else self._imagine_step
        )

    @torch.no_grad()
    def observe_actor_observation(self, obs: torch.Tensor) -> None:
        """Match PPO: update from env.step's returned NEXT observation, including resets."""
        if obs.ndim != 2 or obs.shape[-1] != self.mdp.obs_dim or obs.shape[0] == 0:
            raise ValueError(
                f"Policy observation must have shape (nonempty batch, {self.mdp.obs_dim})"
            )
        if obs.dtype != torch.float32:
            raise ValueError("Policy observation must be float32")
        _finite("policy observation", obs)
        self.policy.obs_norm.update(obs)

    @torch.no_grad()
    def observe_transition(self, transition: dict) -> None:
        """Update WM moments from real PRE/terminal POST, never resets.

        The runner updates actor moments separately from returned NEXT observations,
        including autoresets, matching the original PPO collection order.
        The critic follows the reference implementation and updates its moments
        separately from detached imagined inputs before critic regression.
        """
        state, next_state = transition["s"], transition["s1"]
        action, context = transition["a"], transition["context"]
        features = self.mdp.inputs(state, action, context)
        targets = self.mdp.targets(state, next_state)
        obs = transition["obs"]
        # Check all three before updating any moments.
        for name, value, dim in (
            ("world model features", features, self.mdp.input_dim),
            ("world model targets", targets, self.mdp.delta_dim),
            ("policy observation", obs, self.mdp.obs_dim),
        ):
            if value.ndim != 2 or value.shape[-1] != dim:
                raise ValueError(f"{name} must have shape (batch, {dim})")
            _finite(name, value)
        self.wm.in_norm.update(features)
        self.wm.out_norm.update(targets)

    def _world_model_loss(
        self, windows, member=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed-shape open-loop loss, detached diagnostics and one finite flag.

        Replay rows are always finite (a diverged world's row holds its PRE state
        and is masked), so the first sampled transition is safe padding before any
        MLP or task arithmetic, never an invalid post-reset tail. LayerNorm and the task operate independently
        per batch row; masked padding therefore contributes no loss or gradient.
        """
        state = windows["s"][:, 0]
        padding_state = windows["s"][0, 0]
        padding_action = windows["a"][0, 0]
        padding_context = windows["context"][0, 0]
        padding_target = windows["s1"][0, 0]
        scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
        total = torch.zeros((), device=self.device)
        valid_count = torch.zeros((), device=self.device)
        last_error = torch.zeros((), device=self.device)
        last_count = torch.zeros((), device=self.device)
        all_finite = torch.ones((), device=self.device, dtype=torch.bool)
        for step in range(self.cfg.wm_window):
            active = windows["mask"][:, step] > 0
            rows = active[:, None]
            state = torch.where(rows, state, padding_state)
            action = torch.where(rows, windows["a"][:, step], padding_action)
            context = torch.where(rows, windows["context"][:, step], padding_context)
            target = torch.where(rows, windows["s1"][:, step], padding_target)
            if member is None:  # Single model, or the ensemble-mean prediction.
                predicted = self.wm.step(state, action, context)
            else:
                predicted = self.wm.step(state, action, context, member)
            all_finite = all_finite & (torch.isfinite(predicted) | ~rows).all()
            error = self.mdp.state_error(predicted, target)
            if error.shape != (self.cfg.wm_batch_size, self.mdp.delta_dim):
                raise ValueError("MDP state_error must have shape (batch, delta_dim)")
            all_finite = all_finite & (torch.isfinite(error) | ~rows).all()
            # Mask before squaring, so invalid rows never form NaN * 0 losses.
            error = torch.where(rows, error, torch.zeros_like(error))
            per_sample = (error / scale).square().mean(-1)
            total = total + per_sample.sum()
            count = active.sum()
            valid_count = valid_count + count
            # Keep active open-loop gradients; never feed padded predictions back.
            state = torch.where(rows, predicted, padding_state)
            if step == self.cfg.wm_window - 1:
                last_error = per_sample.sum() / count.clamp_min(1)
                last_count = count
        loss = total / self.cfg.wm_batch_size  # reference: sum steps / batch, not / valid steps
        all_finite = all_finite & torch.isfinite(loss) & (valid_count > 0)
        stats = torch.stack(
            (total / valid_count.clamp_min(1), last_error, last_count, valid_count)
        ).detach()
        return loss, stats, all_finite

    def update_world_model(self, buffer, env_range=None) -> dict[str, torch.Tensor]:
        """One WM step; ``env_range`` keeps held-out validation worlds out of fitting.

        An ensemble fits every member on its own minibatch (members share no
        parameters, so one Adam step equals independent steps) and clips each
        member's gradient separately. Validity (finite predictions, errors, loss
        and gradient norm, some valid target) accumulates on the device, so
        successive updates queue without host synchronization; the caller runs
        check_world_model_updates once per iteration, before any checkpoint.
        """
        kwargs = {} if env_range is None else {"env_range": env_range}
        size = self.cfg.wm_ensemble_size
        members = [None] if size == 1 else list(range(size))
        results = []
        for member in members:
            windows = buffer.sample_windows(self.cfg.wm_batch_size, self.cfg.wm_window, **kwargs)
            if member is None:  # Unchanged single-model call.
                results.append(self._world_model_loss_call(windows))
            else:
                results.append(self._world_model_loss_call(windows, member))
        valid = torch.stack([finite for _, _, finite in results]).all()
        self.wm_optimizer.zero_grad(set_to_none=True)
        for loss, _, _ in results:
            loss.backward()
        groups = [self.wm.parameters()] if size == 1 else [net.parameters() for net in self.wm.nets]
        grad = torch.stack(
            [nn.utils.clip_grad_norm_(group, self.cfg.max_grad_norm) for group in groups]
        ).mean()
        self._world_model_ok &= valid & torch.isfinite(grad)
        self.wm_optimizer.step()
        self.world_model_updates += 1
        loss = torch.stack([loss.detach() for loss, _, _ in results]).mean()
        stats = torch.stack([stats for _, stats, _ in results]).mean(0)
        return {
            "wm_loss": loss,
            "wm_step_loss": stats[0],
            "wm_last_error": stats[1],
            "wm_last_count": stats[2],
            "wm_valid_steps": stats[3],
            "wm_grad_norm": grad.detach(),
        }

    def check_world_model_updates(self) -> None:
        """Raise if any world-model update so far was invalid (one host synchronization)."""
        if not bool(self._world_model_ok):
            raise FloatingPointError(
                "SuperMimic non-finite world model prediction, error, loss or gradient, "
                "or no valid targets"
            )

    @torch.no_grad()
    def world_model_step_loss(self, windows) -> torch.Tensor:
        """Detached per-valid-step WM loss on given windows; no optimizer state changes."""
        _, stats, all_finite = self._world_model_loss(windows)
        if not bool(all_finite):
            raise FloatingPointError("SuperMimic non-finite world model diagnostic loss")
        return stats[0]

    def _critic_features(self, state, context, obs):
        features = self.mdp.critic_features(state, context, obs)
        if features.shape != (state.shape[0], self.mdp.critic_dim):
            raise ValueError("MDP critic_features must have shape (batch, critic_dim)")
        _finite("critic features", features)
        return features

    # Order of the per-category validity flags returned by _imagine_step.
    _IMAGINATION_CHECKS = (
        "critic features",
        "imagined action",
        "imagined state",
        "imagined reward",
        "imagined termination",
        "imagined observation",
    )

    def _imagine_step(self, state, obs, context):
        """One imagined transition as a pure tensor function (optionally compiled).

        Validity is returned as device flags rather than checked here, so the
        rollout queues without host synchronization; update_policy checks all
        flags once, before any backward pass or optimizer step.
        """
        features = self.mdp.critic_features(state, context, obs)
        action = self.policy(obs)
        if self.cfg.wm_ensemble_size == 1:
            next_state = self.wm.step(state, action, context)
            disagreement = torch.zeros(state.shape[0], device=state.device)
        else:
            # A random member per sample and step (as in MBPO/PETS); the
            # penalty's gradient reaches the action through every member.
            index = torch.randint(self.cfg.wm_ensemble_size, (state.shape[0],), device=state.device)
            next_state, disagreement = self.wm.imagine(state, action, context, index)
        reward, done = self.mdp.reward(state, action, next_state, context)
        reward = reward - self.cfg.uncertainty_penalty * disagreement
        next_obs = self.mdp.advance_obs(obs, next_state, action, context)
        flags = torch.stack(
            (
                torch.isfinite(features).all(),
                torch.isfinite(action).all(),
                torch.isfinite(next_state).all() & torch.isfinite(disagreement).all(),
                torch.isfinite(reward).all(),
                torch.isfinite(done).all(),
                torch.isfinite(next_obs).all(),
            )
        )
        in_range = ((done >= 0) & (done <= 1)).all()
        return (
            features,
            next_state,
            reward,
            done.float(),
            next_obs,
            flags,
            in_range,
            disagreement.detach(),
        )

    def update_policy(self, buffer) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.policy_batch_size)
        state, obs, context = start["s"], start["obs"], start["context"]
        rewards, dones, features, disagreements = [], [], [], []
        valid = torch.ones(len(self._IMAGINATION_CHECKS), dtype=torch.bool, device=self.device)
        in_range = torch.ones((), dtype=torch.bool, device=self.device)
        with _fixed_parameters(self.wm, self.critic):
            for step in range(cfg.horizon):
                feature, next_state, reward, done, next_obs, flags, ok_range, disagreement = (
                    self._imagine_step_call(state, obs, context)
                )
                disagreements.append(disagreement)
                if step == 0:  # Static shapes; checking once needs no synchronization.
                    if feature.shape != (state.shape[0], self.mdp.critic_dim):
                        raise ValueError("MDP critic_features must have shape (batch, critic_dim)")
                    if reward.shape != (cfg.policy_batch_size,) or done.shape != reward.shape:
                        raise ValueError("MDP reward/done must have shape (batch,)")
                    if next_obs.shape != (cfg.policy_batch_size, self.mdp.obs_dim):
                        raise ValueError("MDP advance_obs must preserve the observation contract")
                valid = valid & flags
                in_range = in_range & ok_range
                features.append(feature)
                rewards.append(reward)
                dones.append(done)
                state, obs = next_state, next_obs
            final = self.mdp.critic_features(state, context, obs)
            final_ok = torch.isfinite(final).all()[None]
            valid = valid & torch.cat((final_ok, valid.new_ones(len(valid) - 1)))
            features.append(final)
            reward_tensor, done_tensor = torch.stack(rewards), torch.stack(dones)
            alive = torch.cumprod(1 - done_tensor, dim=0)
            before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), dim=0)
            discount = cfg.gamma ** torch.arange(cfg.horizon, device=self.device)[:, None]
            objective = (discount * before * reward_tensor).sum(0)
            objective = objective + cfg.gamma**cfg.horizon * alive[-1] * self.critic(features[-1])
            policy_loss = -objective.mean()
            # One host synchronization for every imagined-step and loss check.
            if not bool(valid.all() & in_range & torch.isfinite(policy_loss)):
                if bool(valid.all()) and not bool(in_range):
                    raise ValueError("MDP imagined termination must be in [0, 1]")
                failed = [
                    name
                    for name, ok in zip(self._IMAGINATION_CHECKS, valid.tolist(), strict=True)
                    if not ok
                ]
                raise FloatingPointError(
                    "SuperMimic non-finite " + (", ".join(failed) if failed else "policy loss")
                )
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            grad = nn.utils.clip_grad_norm_(
                self.policy.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
            )
            nn.utils.clip_grad_value_(self.policy.parameters(), cfg.policy_grad_value_clip)
            self.policy_optimizer.step()
        self.policy_updates += 1

        # Targets and survival weights are fixed throughout critic optimization.
        with torch.no_grad():
            x = torch.stack([feature.detach() for feature in features])
            weights = before.detach().reshape(-1)
            valid_rows = weights > 0
            self.critic.norm.update(x[:-1].flatten(0, 1)[valid_rows])
            values = self.critic(x)
            targets = lambda_returns(
                reward_tensor.detach(), done_tensor.detach(), values, cfg.gamma, cfg.lambda_return
            )
            _finite("critic targets", targets)
            xf = x[:-1].flatten(0, 1)[valid_rows]
            yf = targets.flatten()[valid_rows]
            weights = weights[valid_rows]
        # Native hard terminals create unreachable imagined tails. Exclude them
        # rather than fitting critic values on states the task cannot visit.
        critic_loss_sum = torch.zeros((), device=self.device)
        critic_updates = 0
        for _ in range(cfg.critic_epochs):
            permutation = torch.randperm(xf.shape[0], device=self.device)
            chunks = permutation.tensor_split(min(cfg.critic_minibatches, xf.shape[0]))
            for indices in chunks:
                error = (self.critic(xf[indices]) - yf[indices]).square()
                critic_loss = (error * weights[indices]).sum() / weights[indices].sum()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                # Non-finite loss or gradients raise here, before the optimizer step.
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
                )
                self.critic_optimizer.step()
                critic_loss_sum = critic_loss_sum + critic_loss.detach()
                critic_updates += 1
        metrics = {
            "policy_loss": float(policy_loss.detach()),
            "critic_loss": float(critic_loss_sum) / critic_updates,
            "imagined_reward": float((before.detach() * reward_tensor.detach()).sum(0).mean()),
            "imagined_survival": float(alive[-1].detach().mean()),
            "policy_grad_norm": float(grad),
        }
        if cfg.wm_ensemble_size > 1:
            # Survival-weighted mean over reachable imagined steps (as the reward).
            disagreement = (torch.stack(disagreements) * before.detach()).sum()
            disagreement = float(disagreement / before.detach().sum().clamp_min(1))
            metrics["imagined_disagreement"] = disagreement
            metrics["imagined_uncertainty_penalty"] = cfg.uncertainty_penalty * disagreement
        return metrics

    def set_learning_rate(self, learning_rate: float) -> None:
        if (
            isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        for optimizer in (self.wm_optimizer, self.policy_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

    def state_dict(self) -> dict:
        """Optimizer/model state only; this does not snapshot simulator or replay."""
        return {
            "world_model": self.wm.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "world_model_optimizer": self.wm_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "world_model_updates": self.world_model_updates,
            "policy_updates": self.policy_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore parameter/Adam state; caller must validate task/config contracts."""
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError(f"Agent checkpoint keys differ: {set(state) ^ expected}")
        for name in ("world_model_updates", "policy_updates"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"Checkpoint {name} must be a nonnegative integer")
        self.wm.load_state_dict(state["world_model"], strict=True)
        self.policy.load_state_dict(state["policy"], strict=True)
        self.policy.validate_normalization()
        self.critic.load_state_dict(state["critic"], strict=True)
        for name, module in (
            ("world_model", self.wm),
            ("policy", self.policy),
            ("critic", self.critic),
        ):
            for key, value in module.state_dict().items():
                _finite(f"checkpoint {name}.{key}", value)
        self.wm_optimizer.load_state_dict(state["world_model_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.world_model_updates = state["world_model_updates"]
        self.policy_updates = state["policy_updates"]
