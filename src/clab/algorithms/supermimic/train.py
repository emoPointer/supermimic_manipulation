"""SuperMimic training: real collection, world-model fitting and BPTT policy updates.

The task supplies the environment, reset bank and SuperMimic adapter (its reward,
success and imagination surrogate). Outputs are the resolved configuration,
TensorBoard metrics and local checkpoints. Resume is approximate: networks and
optimizers are restored, the environment and replay are rebuilt with fresh warmup.
"""

from __future__ import annotations

import json
import random
import signal
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import dump_yaml

from clab.contracts import action_contract
from clab.simulation import configure_cuda_graph

from .agent import SuperMimicAgent
from .config import SuperMimicConfig
from .networks import ACTOR_INPUT_NORMALIZATION
from .runner import RunConfig, SuperMimicRunner, finite_tensors


def smoke_profile(algorithm: SuperMimicConfig, run: RunConfig):
    """Tiny networks and budgets for a quick update, save and reload check."""
    algorithm = replace(
        algorithm,
        amp="none",
        tf32=False,
        wm_hidden=(64, 64),
        policy_hidden=(64, 64),
        critic_hidden=(64, 32),
        wm_batch_size=16,
        policy_batch_size=16,
        critic_epochs=1,
        critic_minibatches=2,
    )
    run = replace(
        run,
        iterations=3,
        buffer_steps=64,
        warmup_steps=16,
        world_model_batches=1,
        world_model_only_iterations=1,
        save_interval=3,
        log_interval=1,
        compile_world_model=False,
        compile_policy_update=False,
    )
    return algorithm, run


def bank_control_matches(metadata: dict, cfg) -> bool:
    """Compare the bank's controller and timing provenance with the current config."""

    def normalize(value):
        return json.loads(json.dumps(value))

    provenance = metadata["provenance"]
    expected = [normalize(action_contract(asdict(action))) for action in cfg.actions.values()]
    return (
        normalize(action_contract(provenance.get("controller", {}))) in expected
        and provenance.get("physics_dt") == cfg.sim.mujoco.timestep
        and provenance.get("decimation") == cfg.decimation
    )


def train(
    task,
    *,
    reset_bank: Path,
    output: Path,
    algorithm: SuperMimicConfig | None = None,
    run: RunConfig | None = None,
    num_envs: int = 4096,
    device: str = "cuda:0",
    seed: int = 0,
    nconmax: int | None = None,
    njmax: int | None = None,
    resume: Path | None = None,
    logger: str = "tensorboard",
    cuda_graph_mode: str = "auto",
    smoke: bool = False,
    adapter_options: dict | None = None,
    full_task_resets: bool = False,
) -> dict:
    """Train until the iteration budget or SIGINT/SIGTERM; returns the completion report.

    ``adapter_options`` override the task adapter's imagination settings and are
    recorded in ``params/runtime.json``. ``smoke`` additionally reloads the final
    checkpoint and runs one more update.
    """
    algorithm = algorithm or SuperMimicConfig()
    run = run or RunConfig()
    adapter_options = dict(adapter_options or {})
    if num_envs < 1 or any(value is not None and value < 1 for value in (nconmax, njmax)):
        raise ValueError("Environment count and physics capacities must be positive")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("SuperMimic training requires an available CUDA device")
    bank = Path(reset_bank).resolve(strict=True)
    cfg = task.make_env_cfg(reset_bank=bank)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if nconmax is not None:
        cfg.sim.nconmax = nconmax
    if njmax is not None:
        cfg.sim.njmax = njmax
    if full_task_resets:
        # Ablation of the reset categories: every episode starts from category 0.
        params = cfg.events["reset_from_bank"].params
        if "reaching_only" not in params:
            raise ValueError(f"{task.name} reset event has no full-task-only mode")
        params["reaching_only"] = True
    dataset = task.bank_metadata(torch.load(bank, map_location="cpu", weights_only=True))
    if not bank_control_matches(dataset, cfg):
        raise ValueError(
            "Reset bank controller/timing provenance differs from the current task; "
            "collect a bank with the current controller first"
        )
    run.validate(algorithm)
    if smoke and (run.iterations > 20 or num_envs > 64):
        raise ValueError("Smoke runs are bounded to at most 20 iterations and 64 environments")
    if algorithm.amp == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Requested BF16 MLP autocast is unsupported by this CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = algorithm.tf32
    torch.backends.cudnn.allow_tf32 = algorithm.tf32
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    task.configure_supermimic(cfg)
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", asdict(algorithm))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    metadata = {
        "algorithm": "supermimic",
        "actor_input_normalization": ACTOR_INPUT_NORMALIZATION,
        "task": task.name,
        "seed": seed,
        "smoke": smoke,
        "reset_bank": str(bank),
        "dataset": dataset,
        "run": asdict(run),
        "imagination_overrides": adapter_options,
        "full_task_resets": full_task_resets,
        # The last worlds are held out of world-model fitting; they still seed imagination.
        "world_model_validation_envs": round(run.validation_fraction * num_envs),
        "agent": asdict(algorithm),
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "precision": {
            "parameters_normalizers_state_reward": "float32",
            "mlp_autocast": algorithm.amp,
            "controller_ik": "float64",
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "torch_compile": run.compile_world_model,
        "torch_compile_scope": "world_model_rollout_loss" if run.compile_world_model else None,
        "torch_compile_options": (
            {"fullgraph": True, "dynamic": False, "mode": "default"}
            if run.compile_world_model
            else None
        ),
        # Separate request: one imagined policy-update transition, same options.
        "torch_compile_policy_update": run.compile_policy_update,
        "logger": logger,
        "cuda_graph_mode": cuda_graph_mode,
        "timeout_bootstrap": False,
        "resume_is_exact": False,
        "resume_checkpoint": str(Path(resume).resolve()) if resume else None,
        "evaluation": "separate complete-episode workflow (clab eval policy)",
        "limitations": [
            "Bank controller match is not a physical hold/capacity certificate",
            "World model is a reduced stochastic surrogate, not an exact MuJoCo state",
            "Imagination reward/success is analytic from endpoint states, not a learned reward head",
            "Endpoint imagination reward/success is approximate and need not match native contacts",
            "Real collection and evaluation retain native rewards and terminal conditions",
        ],
    }
    env = runner = None
    started = perf_counter()
    stop = False

    def request_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"{signal.Signals(signum).name}: checkpoint after the current iteration", flush=True)

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        env = ManagerBasedRlEnv(cfg, device=device)
        adapter = task.make_supermimic_adapter(env, **adapter_options)
        mdp_contract = adapter.mdp.contract()
        metadata["reward_mode"] = mdp_contract["reward_mode"]
        metadata["reward_matches_native"] = adapter.mdp.reward_matches_native
        metadata["cuda_graph_runtime"] = configure_cuda_graph(env, cuda_graph_mode)
        with (output / "params/runtime.json").open("x") as stream:
            json.dump(metadata, stream, indent=2)
        with (output / "params/mdp.json").open("x") as stream:
            json.dump(mdp_contract, stream, indent=2)
        runner = SuperMimicRunner(
            env=env,
            adapter=adapter,
            config=algorithm,
            run_config=run,
            output=output,
            task=task.name,
            logger=logger,
        )
        if resume:
            runner.load(Path(resume).resolve(strict=True))
        checkpoint, completed = runner.run(stop_requested=lambda: stop)
        report = {
            "completed_budget": completed,
            "checkpoint": str(checkpoint),
            "iteration": runner.iteration,
            "vector_sim_steps": runner.sim_steps,
            "real_transitions": runner.real_transitions,
            "reward_mode": mdp_contract["reward_mode"],
            "reward_matches_native": runner.reward_matches_native,
            **runner.reward_comparison_metrics(),
            "native_observation_max_error": runner.max_observation_error,
            "terminal_transitions": int(runner.terminal_transitions),
            "diverged_transitions": int(runner.diverged_transitions),
            "contact_match_overflow_world_steps": runner.contact_match_overflow_world_steps,
            "physics_overflow_world_steps": runner.physics_overflow_world_steps,
            "replay_allocated_bytes": runner.buffer.allocated_nbytes,
            "pinned_replay_steps": runner.buffer.pinned_size,
            "world_model_updates": runner.agent.world_model_updates,
            "wall_seconds": perf_counter() - started,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "limitation": "No convergence, held-out evaluation, contact-hold or full-capacity claim",
        }
        if smoke and completed:
            loaded = SuperMimicAgent(adapter.mdp, algorithm, device)
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            loaded.load_state_dict(saved["agent"])
            for name in ("wm", "policy", "critic"):
                for original, restored in zip(
                    getattr(runner.agent, name).state_dict().values(),
                    getattr(loaded, name).state_dict().values(),
                    strict=True,
                ):
                    torch.testing.assert_close(original, restored, atol=0, rtol=0)
            loaded.update_world_model(runner.buffer)
            loaded.update_policy(runner.buffer)
            if not finite_tensors(loaded.state_dict()):
                raise RuntimeError("Non-finite model/optimizer after checkpoint reload and update")
            report["reload_and_update"] = True
        with (output / ("completion.json" if completed else "interrupted.json")).open(
            "x"
        ) as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if runner is not None:
            runner.close()
        if env is not None:
            env.close()
