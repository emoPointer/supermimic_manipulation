"""PPO runner with an opt-in plain CUDA Graph mode for older drivers."""

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from clab.contracts import (
    action_contract,
    checkpoint_environment,
    validate_environment_contract,
)
from clab.logging.episode_metrics import EpisodeLogger
from clab.simulation import CONTACT_MATCH_OVERFLOW, configure_cuda_graph


def validate_resume_control(path: str, cfg) -> None:
    """Reject incompatible sensor/action/dynamics contracts before loading weights.

    Allocation and bank selection may differ; this is not an exact-resume claim.
    """
    saved, _ = checkpoint_environment(Path(path))
    validate_environment_contract(saved, cfg, for_resume=True)


class ClabPpoRunner(MjlabOnPolicyRunner):
    """Keep mjlab's PPO loop and optionally capture fixed-iteration physics graphs."""

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        train_cfg = train_cfg.copy()
        backend = train_cfg.pop("logging_backend", "both")
        if backend == "tensorboard":
            train_cfg["logger"] = "tensorboard"
        elif backend != "both":
            raise ValueError(f"Unsupported PPO logger: {backend}")
        graph_mode = train_cfg.pop("cuda_graph_mode", "auto")
        reward_source = train_cfg.pop("reward_source", "native")
        task_name = train_cfg.pop("task_name", "")
        adapter = None
        if reward_source == "imagined":
            from clab.tasks.registry import load_task

            adapter = load_task(task_name).make_supermimic_adapter(env.unwrapped)
        elif reward_source != "native":
            raise ValueError(f"Unsupported PPO reward source: {reward_source}")
        configure_cuda_graph(env.unwrapped, graph_mode)
        if graph_mode in ("plain", "substeps"):
            env.reset()  # Discard the warmup state before PPO starts collecting.
        super().__init__(env, train_cfg, log_dir, device)
        # A world whose physics diverged is terminated and reset by the task;
        # its final reward is NaN and must not reach the PPO returns.
        step = self.env.step
        clip = self.env.clip_actions
        sim = env.unwrapped.sim
        overflow_counts = {"physics": 0, "contact_match": 0}

        def finite_step(actions):
            if adapter is not None:
                state, context = adapter.read()
            obs, rewards, dones, extras = step(actions)
            # As in SuperMimic: PPO's per-task capacities are smaller than the
            # environment's, so count physics overflows (flags persist until a world
            # resets) and stop when many worlds overflow at once.
            overflow = sim.data.overflow
            if bool((overflow != 0).any()):
                physics = torch.bitwise_and(overflow, ~CONTACT_MATCH_OVERFLOW) != 0
                overflowed = int(physics.sum())
                if overflowed > max(1, overflow.numel() // 100):
                    codes = sorted(set(overflow[physics].tolist()))
                    raise RuntimeError(
                        f"Physics capacity overflow in {overflowed} worlds (mujoco_warp "
                        f"OverflowType bitmasks {codes}); raise the task's ppo_capacity"
                    )
                overflow_counts["physics"] += overflowed
                overflow_counts["contact_match"] += int(((overflow != 0) & ~physics).sum())
            if adapter is not None:
                # The same equation SuperMimic optimizes, on the actual PRE state,
                # issued action and pre-autoreset POST state of this step.
                issued = actions if clip is None else torch.clamp(actions, -clip, clip)
                native = rewards
                rewards, _ = adapter.mdp.reward(state, issued, adapter.post_step(), context)
                # Only steps that already carry episode logs: a record without the
                # episode keys first would hide them from the upstream reduction.
                if "log" in extras:
                    extras["log"]["Reward/native_step_mean"] = torch.nan_to_num(native).mean()
                    extras["log"]["Reward/imagined_step_mean"] = torch.nan_to_num(rewards).mean()
            if "log" in extras:
                for name, count in overflow_counts.items():
                    extras["log"][f"Train/{name}_overflow_world_steps"] = torch.tensor(
                        float(count), device=rewards.device
                    )
            return obs, torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0), dones, extras

        self.env.step = finite_step
        # Upstream constructs its default logger without a factory hook. Replace
        # the still-uninitialized logger; retain the PPO loop and writer config.
        self.logger = EpisodeLogger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        if log_dir is not None:
            datasets = {}
            for name in env.unwrapped.cfg.events:
                term = env.unwrapped.event_manager.get_term_cfg(name).func
                if hasattr(term, "metadata"):
                    datasets[name] = term.metadata
            action_configs = [
                action_contract(asdict(action)) for action in env.cfg.actions.values()
            ]
            if any(
                action_contract(dataset.get("provenance", {}).get("controller", {}))
                not in action_configs
                for dataset in datasets.values()
            ):
                warnings.warn(
                    "Reset-bank controller does not match the current hold contract; "
                    "pass per-task collection/update/reload gates before formal training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            metadata = {
                "datasets": datasets,
                "device": device,
                "device_name": torch.cuda.get_device_name(device)
                if device.startswith("cuda")
                else "cpu",
                "logging_backend": backend,
                "cuda_graph_mode": graph_mode,
                "reward_source": reward_source,
                "reward_function": (
                    "task SuperMimic MDP reward(s, a, s1, c) on real transitions"
                    if adapter is not None
                    else "native reward manager"
                ),
                "finite_horizon": env.cfg.is_finite_horizon,
                "timeout_bootstrap": not env.cfg.is_finite_horizon,
                "resume_is_exact": False,
                "success_metric_aggregation": "completed_episode_counts_per_iteration",
                "success_metric_scope": "logging_process_only",
            }
            path = Path(log_dir) / "params" / "runtime.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump(metadata, stream, indent=2)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Restore upstream state and the adaptive scheduler's current rate.

        Upstream restores Adam param groups but retains the constructor's
        learning_rate attribute. Its next adaptive update would otherwise
        replace the restored optimizer rate with the fresh-run initial rate.
        Simulation and random streams remain approximate continuation.
        """
        validate_resume_control(path, self.env.cfg)
        infos = super().load(path, load_cfg, strict, map_location)
        if load_cfg is None or load_cfg.get("optimizer", False):
            rates = {float(group["lr"]) for group in self.alg.optimizer.param_groups}
            if len(rates) != 1 or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
                raise ValueError(f"PPO resume requires one finite positive learning rate: {rates}")
            self.alg.learning_rate = rates.pop()
        return infos
