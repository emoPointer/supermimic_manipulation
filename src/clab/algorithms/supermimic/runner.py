"""Task-independent real collection / world-model / imagined BPTT workflow.

The task adapter captures native POST states before autoreset. Real task rewards,
controller execution and observation history are never replaced by predictions.
Checkpoints exclude simulator/replay state: resume always refills replay first.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from clab.contracts import checkpoint_environment, validate_environment_contract
from clab.logging.episode_metrics import aggregate_success_rates
from clab.logging.metrics import MetricsLogger

from .agent import SuperMimicAgent
from .buffer import ReplayBuffer
from .config import SuperMimicConfig
from .networks import ACTOR_INPUT_NORMALIZATION


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    iterations: int = 10_000
    buffer_steps: int = 1_000
    warmup_steps: int = 64
    world_model_batches: int = 30
    world_model_only_iterations: int = 1_000
    updates_per_sim_step: float = 0.4
    learning_rate_decay_iterations: int = 10_000
    save_interval: int = 1_000
    log_interval: int = 50
    action_noise: float = 0.1
    compile_world_model: bool = True
    compile_policy_update: bool = True
    # At the first policy-update iteration keep the replay as a fixed segment
    # (WM-only data from the unchanged initial policy) and continue collection
    # in a new ring of streaming_buffer_steps; sampling is uniform over both.
    pin_replay: bool = False
    streaming_buffer_steps: int = 1_000
    # "batches": world_model_batches updates per iteration. "epochs": updates
    # such that sampled windows ~= world_model_epochs x newly collected
    # training-world transitions, i.e. expected window-start passes per
    # transition over its replay lifetime in steady state.
    world_model_schedule: str = "epochs"
    world_model_epochs: float = 4.0
    # Held-out worlds (the last fraction of indices) never enter WM fitting;
    # their data flows through the replay and is evaluated at log iterations.
    validation_fraction: float = 0.05
    validation_batches: int = 4
    validation_recent_steps: int = 100
    # Exploration worlds (the first fraction of indices, disjoint from held-out
    # validation worlds) follow the policy, but at random times flip its binary
    # gripper decision and hold the flipped command for exploration_steps steps.
    exploration_fraction: float = 0.0
    exploration_start_probability: float = 0.05
    exploration_steps: int = 10
    # Actor/critic update every policy_update_interval iterations after WM-only.
    policy_update_interval: int = 1

    def validate(self, algorithm: SuperMimicConfig) -> None:
        for name in ("compile_world_model", "compile_policy_update", "pin_replay"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit boolean")
        streaming = self.streaming_buffer_steps
        if type(streaming) is not int or streaming < algorithm.wm_window:
            raise ValueError("streaming_buffer_steps must be an integer of at least wm_window")
        if self.world_model_schedule not in ("batches", "epochs"):
            raise ValueError("world_model_schedule must be 'batches' or 'epochs'")
        for name in ("world_model_epochs", "validation_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number")
        if not math.isfinite(self.world_model_epochs) or self.world_model_epochs <= 0:
            raise ValueError("world_model_epochs must be finite and positive")
        if not math.isfinite(self.validation_fraction) or not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if type(self.validation_batches) is not int or self.validation_batches < 1:
            raise ValueError("validation_batches must be a positive integer")
        recent = self.validation_recent_steps
        if type(recent) is not int or recent < algorithm.wm_window:
            raise ValueError("validation_recent_steps must be an integer of at least wm_window")
        for name in ("exploration_fraction", "exploration_start_probability"):
            value = getattr(self, name)
            real = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not real or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 0 <= self.exploration_fraction < 1:
            raise ValueError("exploration_fraction must be in [0, 1)")
        if self.exploration_fraction + self.validation_fraction >= 1:
            raise ValueError("Exploration and validation worlds must leave policy training worlds")
        if not 0 < self.exploration_start_probability <= 1:
            raise ValueError("exploration_start_probability must be in (0, 1]")
        for name in ("exploration_steps", "policy_update_interval"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "iterations",
            "buffer_steps",
            "warmup_steps",
            "world_model_batches",
            "learning_rate_decay_iterations",
            "save_interval",
            "log_interval",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.world_model_only_iterations) is not int
            or self.world_model_only_iterations < 0
        ):
            raise ValueError("world_model_only_iterations must be a nonnegative integer")
        if (
            isinstance(self.updates_per_sim_step, bool)
            or not math.isfinite(self.updates_per_sim_step)
            or self.updates_per_sim_step <= 0
        ):
            raise ValueError("updates_per_sim_step must be finite and positive")
        if not self.buffer_steps >= self.warmup_steps >= algorithm.wm_window:
            raise ValueError("Require buffer_steps >= warmup_steps >= wm_window")
        if not math.isfinite(self.action_noise) or self.action_noise < 0:
            raise ValueError("action_noise must be finite and nonnegative")


def finite_tensors(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tensors(item) for item in value)
    return True


class SuperMimicRunner:
    def __init__(self, *, env, adapter, config, run_config, output, task, logger="tensorboard"):
        run_config.validate(config)
        if not env.cfg.auto_reset or not env.cfg.is_finite_horizon:
            raise ValueError("SuperMimic currently requires native autoreset and finite horizon")
        self.env, self.adapter, self.mdp = env, adapter, adapter.mdp
        self.reward_matches_native = getattr(self.mdp, "reward_matches_native", True)
        if type(self.reward_matches_native) is not bool:
            raise ValueError("MDP reward_matches_native must be an explicit boolean")
        self.config, self.run_config = config, run_config
        self.output, self.task = Path(output), task
        self.device = torch.device(env.device)
        self.agent = SuperMimicAgent(self.mdp, config, self.device)
        if run_config.compile_world_model:
            self.agent.enable_world_model_compile()
        if run_config.compile_policy_update:
            self.agent.enable_policy_update_compile()
        self.buffer = ReplayBuffer(
            run_config.buffer_steps,
            env.num_envs,
            self.mdp.state_dim,
            self.mdp.action_dim,
            self.mdp.context_dim,
            self.mdp.obs_dim,
            self.device,
        )
        self.logger = MetricsLogger(self.output / "tensorboard", logger)
        self.observations, _ = env.reset()
        self.iteration = self.sim_steps = self.real_transitions = 0
        self.collection_fraction = 0.0
        self.episode_records: list[dict] = []
        self.max_reward_error = 0.0
        self.reward_comparison_count = 0
        self.surrogate_native_reward_abs_sum = 0.0
        self.surrogate_native_reward_max_abs_gap = 0.0
        self.surrogate_native_done_mismatch_count = 0
        # Optional task hook: per-step success predicates, independent of counters.
        self.candidate_counts = {"surrogate": 0, "native": 0, "surrogate_only": 0, "native_only": 0}
        self.max_observation_error = 0.0
        self.terminal_transitions = 0
        self.diverged_transitions = 0

    def _record_reward_comparison(self, model_reward, model_done, native_reward, native_done):
        expected_shape = (self.env.num_envs,)
        for name, value in (("reward", model_reward), ("done", model_done)):
            if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
                raise RuntimeError(f"Model {name} must be a tensor of shape {expected_shape}")
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Non-finite model {name} at vector step {self.sim_steps}")
        if bool(((model_done < 0) | (model_done > 1)).any()):
            raise RuntimeError("Model done must be in [0, 1]")
        gap = (model_reward - native_reward).abs()
        if not bool(torch.isfinite(gap).all()):
            raise RuntimeError("Non-finite surrogate/native reward gap")
        error = gap.max().item()
        mismatch = model_done.bool() != native_done
        self.reward_comparison_count += gap.numel()
        self.surrogate_native_reward_abs_sum += gap.sum(dtype=torch.float64).item()
        self.surrogate_native_reward_max_abs_gap = max(
            self.surrogate_native_reward_max_abs_gap, error
        )
        self.surrogate_native_done_mismatch_count += int(mismatch.sum().item())
        if self.reward_matches_native:
            self.max_reward_error = max(self.max_reward_error, error)
            if not torch.allclose(model_reward, native_reward, atol=2e-5, rtol=2e-5):
                raise RuntimeError(f"Native/model reward contract mismatch: max error {error}")
            if bool(mismatch.any()):
                raise RuntimeError("Native/model terminal contract mismatch")

    def reward_comparison_metrics(self) -> dict[str, float | int]:
        """Real-transition diagnostics since startup, including replay warmup."""
        count = self.reward_comparison_count
        metrics = {
            "surrogate_native_reward_mean_abs_gap": (
                self.surrogate_native_reward_abs_sum / count if count else 0.0
            ),
            "surrogate_native_reward_max_abs_gap": self.surrogate_native_reward_max_abs_gap,
            "surrogate_native_done_mismatch_rate": (
                self.surrogate_native_done_mismatch_count / count if count else 0.0
            ),
            "surrogate_native_comparison_count": count,
        }
        if hasattr(self.adapter, "candidate_comparison"):
            for key, value in self.candidate_counts.items():
                metrics[f"candidate_{key}_rate"] = value / count if count else 0.0
        if self.reward_matches_native:
            metrics["native_reward_max_error"] = self.max_reward_error
        return metrics

    @torch.no_grad()
    def collect(self, steps: int, *, warmup: bool = False) -> None:
        def report_warmup(completed: int) -> None:
            metrics = {
                "Warmup/vector_steps_completed": completed,
                "Warmup/vector_steps_target": steps,
                "Warmup/sim_steps": self.sim_steps,
                "Warmup/real_transitions": self.real_transitions,
            }
            # Warmup has its own local step axis; do not advance W&B's shared
            # outer-iteration clock or invent optimizer/episode measurements.
            if self.logger.writer is not None:
                for name, value in metrics.items():
                    self.logger.writer.add_scalar(name, value, completed)
                self.logger.writer.flush()
            print(
                f"SuperMimic warmup={completed}/{steps} "
                f"sim_steps={self.sim_steps} real_transitions={self.real_transitions}",
                flush=True,
            )

        if warmup:
            report_warmup(0)
        for index in range(steps):
            state, context = self.adapter.read()
            obs = self.observations["actor"].clone()
            observation_error = self.adapter.validate_observation(state, context, obs)
            self.max_observation_error = max(self.max_observation_error, observation_error)
            action = self.agent.policy(obs)
            action = action + self.run_config.action_noise * torch.randn_like(action)
            action = self._explore(action)
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError(f"Non-finite real action at vector step {self.sim_steps}")
            observations, reward, terminated, truncated, extras = self.env.step(action)
            self._count_episodes(terminated, truncated)
            post = self.adapter.post_step()
            # A world whose physics diverged was terminated and reset by the task.
            # Its POST row is not a real transition: replace it by the finite PRE
            # state with zero reward and mark it invalid for world-model windows.
            valid = torch.isfinite(post).all(-1) & torch.isfinite(reward)
            post = torch.where(valid[:, None], post, state)
            reward = torch.where(valid, reward, 0.0)
            self.diverged_transitions += int((~valid).sum().item())
            transition = {
                "s": state,
                "s1": post,
                "a": action,
                "context": context,
                "obs": obs,
                "reward": reward.clone(),
                "terminated": terminated.clone(),
                "truncated": truncated.clone(),
                "valid": valid,
            }
            if not finite_tensors(transition):
                raise RuntimeError(f"Non-finite real transition at vector step {self.sim_steps}")
            # Compare task-owned analytic reward equations on actual simulator
            # POST states before reset. An explicitly declared imagination
            # surrogate may differ; native replay rewards and terminals stay intact.
            reconstructed_reward, reconstructed_done = self.mdp.reward(
                state,
                action,
                post,
                context,
            )
            self._record_reward_comparison(
                reconstructed_reward, reconstructed_done, reward, terminated | truncated
            )
            if hasattr(self.adapter, "candidate_comparison"):
                surrogate, native = self.adapter.candidate_comparison(post, context)
                for key, value in (
                    ("surrogate", surrogate),
                    ("native", native),
                    ("surrogate_only", surrogate & ~native),
                    ("native_only", native & ~surrogate),
                ):
                    self.candidate_counts[key] += int(value.sum().item())
            overflow = self.env.sim.data.overflow
            if bool((overflow != 0).any()):
                codes = sorted(set(overflow[overflow != 0].tolist()))
                raise RuntimeError(
                    f"Physics capacity overflow in {int((overflow != 0).sum())} worlds "
                    f"(mujoco_warp OverflowType bitmasks {codes}); raise the task's "
                    "nconmax, njmax or contact_sensor_maxmatch"
                )
            self.buffer.add(transition)
            self.agent.observe_transition(transition)
            self.agent.observe_actor_observation(observations["actor"])
            if "log" in extras:
                self.episode_records.append(
                    {
                        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                        for key, value in extras["log"].items()
                        if key.startswith("Episode_Metrics/")
                    }
                )
            self.terminal_transitions += int((terminated | truncated).sum().item())
            self.observations = observations
            self.sim_steps += 1
            self.real_transitions += self.env.num_envs
            if warmup:
                report_warmup(index + 1)

    def _explore(self, action: torch.Tensor) -> torch.Tensor:
        """Flip the gripper decision in exploration worlds for held random intervals.

        At each start, the opposite of the policy's current binary decision is
        held for exploration_steps vector steps, so a force-limited gripper can
        actually open (or close). Arm axes and all other worlds are unchanged.
        """
        cfg = self.run_config
        if not cfg.exploration_fraction:
            return action
        index = getattr(self.mdp, "gripper_action_index", None)
        if type(index) is not int:
            raise ValueError("Gripper exploration requires the task MDP's gripper_action_index")
        count = round(cfg.exploration_fraction * action.shape[0])
        if not count:
            return action
        if getattr(self, "_exploration_remaining", None) is None:
            self._exploration_remaining = torch.zeros(count, dtype=torch.long, device=action.device)
            self._exploration_target = torch.zeros(count, device=action.device)
        remaining = self._exploration_remaining
        start = (remaining == 0) & (
            torch.rand(count, device=action.device) < cfg.exploration_start_probability
        )
        flipped = torch.where(action[:count, index] >= 0, -1.0, 1.0)
        self._exploration_target = torch.where(start, flipped, self._exploration_target)
        remaining = torch.where(start, cfg.exploration_steps, remaining)
        active = remaining > 0
        action = action.clone()
        action[:count, index] = torch.where(active, self._exploration_target, action[:count, index])
        self._exploration_remaining = remaining - active.long()
        return action

    def _count_episodes(self, terminated, truncated) -> None:
        """Per world group: terminated and completed episodes since the last log."""
        count = round(self.run_config.exploration_fraction * terminated.shape[0])
        if getattr(self, "episode_group_counts", None) is None:
            self.episode_group_counts = {"policy": [0, 0], "exploration": [0, 0]}
        done = terminated | truncated
        for name, rows in (("exploration", slice(0, count)), ("policy", slice(count, None))):
            self.episode_group_counts[name][0] += int(terminated[rows].sum().item())
            self.episode_group_counts[name][1] += int(done[rows].sum().item())

    def _episode_group_metrics(self) -> dict[str, float]:
        """Terminated-episode rates; for BoxInBin only success terminates (timeouts truncate)."""
        counts = getattr(self, "episode_group_counts", None) or {}
        result = {}
        for name, (terminal, completed) in counts.items():
            if completed:
                result[f"Train/{name}_worlds_terminated_episode_rate"] = terminal / completed
                result[f"Train/{name}_worlds_completed_episodes"] = completed
        self.episode_group_counts = {"policy": [0, 0], "exploration": [0, 0]}
        return result

    def save(self) -> Path:
        path = self.output / f"model_{self.iteration:06d}.pt"
        if path.exists():
            raise FileExistsError(path)
        numpy_state = np.random.get_state()
        checkpoint = {
            "algorithm": "supermimic",
            "actor_input_normalization": ACTOR_INPUT_NORMALIZATION,
            "task": self.task,
            "config": asdict(self.config),
            "run_config": asdict(self.run_config),
            "mdp_contract": self.mdp.contract(),
            "agent": self.agent.state_dict(),
            "iteration": self.iteration,
            "sim_steps": self.sim_steps,
            "real_transitions": self.real_transitions,
            "collection_fraction": self.collection_fraction,
            "world_model_fraction": getattr(self, "world_model_fraction", 0.0),
            "resume_is_exact": False,
            "replay_included": False,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python": random.getstate(),
                "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            },
        }
        if not finite_tensors(checkpoint):
            raise RuntimeError("Refusing to save non-finite checkpoint")
        temporary = path.with_suffix(".pt.tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        return path

    def load(self, path: Path) -> None:
        saved_environment, _ = checkpoint_environment(path)
        validate_environment_contract(saved_environment, self.env.cfg, for_resume=True)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "supermimic" or checkpoint.get("task") != self.task:
            raise ValueError("Checkpoint algorithm/task mismatch")
        if checkpoint.get("actor_input_normalization") != ACTOR_INPUT_NORMALIZATION:
            raise ValueError("Resume requires matching PPO actor observation normalization")
        if checkpoint.get("mdp_contract") != self.mdp.contract():
            raise ValueError("Checkpoint world-model state/context contract mismatch")
        if checkpoint.get("config") != asdict(self.config):
            raise ValueError("Resume requires the saved algorithm configuration")
        if not finite_tensors(checkpoint):
            raise ValueError("Checkpoint contains non-finite tensors")
        self.agent.load_state_dict(checkpoint["agent"])
        self.iteration = checkpoint["iteration"]
        self.sim_steps = checkpoint["sim_steps"]
        self.real_transitions = checkpoint["real_transitions"]
        self.collection_fraction = checkpoint["collection_fraction"]
        self.world_model_fraction = checkpoint.get("world_model_fraction", 0.0)
        rng = checkpoint["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"]:
            if len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("Resume requires the same visible CUDA device count")
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        random.setstate(rng["python"])
        np_state = rng["numpy"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
        # Simulator and ring are intentionally new. run() always collects the
        # configured warmup even when weights/optimizers/RNG have been loaded.

    def _world_model_diagnostics(self, cfg, train_envs, val_envs) -> dict[str, float]:
        """Detached per-valid-step WM loss on replay subsets; no optimizer state changes.

        With held-out worlds: eval-mode loss on training worlds, on held-out
        worlds, on held-out worlds' newest streaming rows, and per segment when
        pinned. Without them, only the per-segment losses over all worlds.
        """
        specs = {}
        if val_envs is not None:
            specs["train_eval"] = {"env_range": train_envs}
            specs["val"] = {"env_range": val_envs}
            specs["val_recent"] = {"env_range": val_envs, "newest": cfg.validation_recent_steps}
        if self.buffer.pinned is not None:
            for segment in ("pinned", "streaming"):
                if val_envs is None:
                    specs[segment] = {"segment": segment}
                else:
                    specs[f"train_{segment}"] = {"segment": segment, "env_range": train_envs}
                    specs[f"val_{segment}"] = {"segment": segment, "env_range": val_envs}
        batches = cfg.validation_batches if val_envs is not None else 1
        result = {}
        for name, kwargs in specs.items():
            losses = []
            for _ in range(batches):
                try:
                    windows = self.buffer.sample_windows(
                        self.config.wm_batch_size, self.config.wm_window, **kwargs
                    )
                except ValueError:
                    break  # Subset not yet one window long (e.g. a fresh streaming ring).
                losses.append(float(self.agent.world_model_step_loss(windows)))
            if losses:
                result[f"Model/wm_{name}_step_loss"] = sum(losses) / len(losses)
        if "Model/wm_val_step_loss" in result and result.get("Model/wm_train_eval_step_loss"):
            result["Model/wm_val_train_ratio"] = (
                result["Model/wm_val_step_loss"] / result["Model/wm_train_eval_step_loss"]
            )
        return result

    def run(self, *, stop_requested=lambda: False) -> tuple[Path, bool]:
        cfg = self.run_config
        if cfg.pin_replay and self.iteration > cfg.world_model_only_iterations:
            # Replay is not checkpointed; refilled warmup would come from the
            # resumed policy, not the initial-policy data the segment must hold.
            raise ValueError(
                "pin_replay cannot be resumed after its pin iteration; replay is not checkpointed"
            )
        worlds = self.buffer.num_envs
        held_out = round(cfg.validation_fraction * worlds)
        if cfg.validation_fraction > 0 and not 0 < held_out < worlds:
            raise ValueError("validation_fraction must hold out at least one world and keep one")
        if round(cfg.exploration_fraction * worlds) + held_out >= worlds:
            raise ValueError("Exploration and validation worlds must leave policy training worlds")
        train_envs = (0, worlds - held_out) if held_out else None
        val_envs = (worlds - held_out, worlds) if held_out else None
        self.world_model_fraction = getattr(self, "world_model_fraction", 0.0)
        log_windows = log_new_transitions = 0
        start = time.monotonic()
        self.collect(cfg.warmup_steps, warmup=True)
        last_checkpoint = None
        end = self.iteration + cfg.iterations
        for iteration in range(self.iteration, end):
            record = (iteration + 1) % cfg.log_interval == 0 or iteration + 1 == end
            fraction = min(iteration / cfg.learning_rate_decay_iterations, 1.0)
            learning_rate = self.config.learning_rate * (
                0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * fraction))
            )
            self.agent.set_learning_rate(learning_rate)
            if cfg.pin_replay and iteration == cfg.world_model_only_iterations:
                self.buffer.pin(cfg.streaming_buffer_steps)
            self.collection_fraction += 1.0 / cfg.updates_per_sim_step
            collect_steps = int(self.collection_fraction)
            self.collection_fraction -= collect_steps
            self.collect(collect_steps)
            new_transitions = collect_steps * (worlds - held_out)
            if cfg.world_model_schedule == "epochs":
                self.world_model_fraction += (
                    cfg.world_model_epochs * new_transitions / self.config.wm_batch_size
                )
                wm_updates = int(self.world_model_fraction)
                self.world_model_fraction -= wm_updates
            else:
                wm_updates = cfg.world_model_batches
            log_windows += wm_updates * self.config.wm_batch_size
            log_new_transitions += new_transitions
            metrics: dict[str, float | torch.Tensor] = {}
            for _ in range(wm_updates):
                if train_envs is None:
                    update = self.agent.update_world_model(self.buffer)
                else:
                    update = self.agent.update_world_model(self.buffer, env_range=train_envs)
                if record:
                    for key, value in update.items():
                        name = "Model/" + key
                        # Keep detached diagnostics on their device until logging.
                        # Non-recorded iterations need no metric reduction/transfer.
                        metrics[name] = metrics.get(name, 0.0) + value / wm_updates
            policy_phase = iteration - cfg.world_model_only_iterations
            if policy_phase >= 0 and policy_phase % cfg.policy_update_interval == 0:
                for key, value in self.agent.update_policy(self.buffer).items():
                    prefix = "Policy/"
                    if key.startswith("critic_"):
                        prefix = "Critic/"
                    elif key.startswith("imagined_"):
                        prefix = "Imagination/"
                    metrics[prefix + key] = float(value)
            self.iteration = iteration + 1
            if record:
                tensor_keys = [
                    key for key, value in metrics.items() if isinstance(value, torch.Tensor)
                ]
                if tensor_keys:
                    values = (
                        torch.stack([metrics[key].detach() for key in tensor_keys]).cpu().tolist()
                    )
                    metrics.update(zip(tensor_keys, values, strict=True))
                # Reference logging divides the same optimized sum by fixed H;
                # wm_step_loss instead divides by the number of valid positions.
                if "Model/wm_loss" in metrics:
                    metrics["Model/wm_loss_per_step"] = (
                        metrics["Model/wm_loss"] / self.config.wm_window
                    )
                # Expected passes per newly collected training transition (epochs).
                metrics["Train/wm_windows_per_new_transition"] = (
                    log_windows / log_new_transitions if log_new_transitions else 0.0
                )
                metrics["Train/world_model_updates"] = getattr(self.agent, "world_model_updates", 0)
                log_windows = log_new_transitions = 0
                metrics.update(self._world_model_diagnostics(cfg, train_envs, val_envs))
                metrics.update(self._episode_group_metrics())
                if self.buffer.pinned is not None:
                    metrics["Train/pinned_replay_steps"] = self.buffer.pinned_size
                    metrics["Train/streaming_replay_steps"] = self.buffer.size
                metrics.update(aggregate_success_rates(self.episode_records))
                self.episode_records.clear()
                metrics.update(
                    {
                        "Model/" + key: value
                        for key, value in self.reward_comparison_metrics().items()
                    }
                )
                metrics.update(
                    {
                        "Train/learning_rate": learning_rate,
                        "Train/real_transitions": self.real_transitions,
                        "Train/wall_seconds": time.monotonic() - start,
                        "Model/native_observation_max_error": self.max_observation_error,
                        "Model/native_observation_rotation_coordinate_max_error": getattr(
                            self.adapter, "observation_rotation_coordinate_max_error", 0.0
                        ),
                    }
                )
                self.logger.log(metrics, step=self.iteration)
                print(f"SuperMimic iteration={self.iteration} metrics={metrics}", flush=True)
            stopped = stop_requested()
            if self.iteration % cfg.save_interval == 0 or self.iteration == end or stopped:
                last_checkpoint = self.save()
            if stopped:
                return last_checkpoint, False
        return last_checkpoint, True

    def close(self) -> None:
        self.logger.close()
