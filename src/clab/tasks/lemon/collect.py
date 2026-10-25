"""Collect a Lemon reset bank from CPU IK proposals and GPU physics holds.

Run through ``clab resets collect --task lemon``. Holds command the initial
end-effector pose through ordinary OSC actions, with no extra force, fixation or gains.
Accepted candidates are finger-projected in a copy, reloaded through a normal reset and
must pass the per-step predicates for one more full second; the exported state is that
tested input, not its end state. ``--proposal-workers`` scales CPU proposals and
``--num-envs`` the GPU holds; both change the recorded sampling streams. Per-batch timings
are printed and saved. A failed run writes no bank.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from clab.controllers.osc import OscActionCfg
from clab.controllers.pose import pose_target_action
from clab.logging.model import joint_damping_metadata
from clab.observations.state import observation_pose
from clab.robots.parallel_kinematics import DEFAULT_PROPOSAL_WORKERS
from clab.scenes.workspace import workspace_metadata
from clab.tasks.rigid_resets import reloaded_hold_mask

from .config import make_env_cfg
from .mdp import narrow_grasp_alignment
from .proposals import Sampler
from .props import within_workspace
from .reset_data import FIELDS, SCHEMA, SPEC, bank_contract, validate_bank
from .resets import snapshot, write_states
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


@dataclass(frozen=True)
class AcceptanceConfig:
    """Reset quality tolerances, not additional task success conditions."""

    linear_speed_max: float = 0.03
    angular_speed_max: float = 0.3
    resting_linear_speed_max: float = 0.003
    resting_angular_speed_max: float = 0.05
    joint_speed_max: float = 2.0
    finger_upper_tolerance: float = DEFAULT_ROBOT.reset_finger_upper_tolerance
    object_drift_max: float = 0.02
    stable_seconds: float = 0.5
    grasp_alignment_max_degrees: float = 10.0


def hold_action(
    env,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    *,
    gripper_command: float,
    action_cfg: OscActionCfg,
) -> torch.Tensor:
    """Track one fixed world pose using the unchanged relative-action interface.

    Zero relative actions would re-anchor the target to a drifting current pose
    every control step. This instead recomputes the corrective command; it does
    not change the controller, state, forces or target between steps.
    """
    return pose_target_action(
        *observation_pose(env, "robot", DEFAULT_ROBOT.ee_body),
        target_pos,
        target_quat,
        action_cfg=action_cfg,
        gripper_command=gripper_command,
    )


def _touch(env, name: str) -> torch.Tensor:
    return (env.scene[name].data.found > 0).any(dim=-1)


def _safe_contacts(env, max_penetration: float) -> torch.Tensor:
    good = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in (
        "robot_contact",
        "lemon_table_contact",
        "lemon_plate_contact",
    ):
        sensor = env.scene[name].data
        good &= ~((sensor.found > 0) & (sensor.dist < -max_penetration)).any(dim=-1)
    return good


def accepted(
    env,
    category: int,
    start: dict,
    *,
    max_penetration: float,
    criteria: AcceptanceConfig,
    joint_limits: torch.Tensor,
) -> torch.Tensor:
    """Reset quality filters, intentionally distinct from the task's success rule."""
    rows = snapshot(env)
    good = _safe_contacts(env, max_penetration)
    for value in rows.values():
        good &= torch.isfinite(value).all(dim=-1)
    speed = torch.linalg.vector_norm(rows["lemon_vel"][:, :3], dim=-1)
    spin = torch.linalg.vector_norm(rows["lemon_vel"][:, 3:], dim=-1)
    good &= (speed < criteria.linear_speed_max) & (spin < criteria.angular_speed_max)
    good &= rows["joint_vel"].abs().amax(-1) < criteria.joint_speed_max
    good &= DEFAULT_ROBOT.reset_joint_position_mask(
        rows["joint_pos"], joint_limits, criteria.finger_upper_tolerance
    )
    good &= (
        torch.linalg.vector_norm(rows["lemon_pose"][:, :3] - start["lemon_pose"][:, :3], dim=-1)
        < criteria.object_drift_max
    )
    finger = _touch(env, "left_finger_contact") & _touch(env, "right_finger_contact")
    table = _touch(env, "lemon_table_contact")
    plate = _touch(env, "lemon_plate_contact")
    if category == 0:
        return (
            good
            & table
            & ~plate
            & ~_touch(env, "lemon_robot_contact")
            & (speed < criteria.resting_linear_speed_max)
            & (spin < criteria.resting_angular_speed_max)
        )
    good &= narrow_grasp_alignment(env) >= math.cos(
        math.radians(criteria.grasp_alignment_max_degrees)
    )
    if category == 1:
        return good & table & ~plate & finger
    if category == 2:
        return good & ~table & ~plate & finger & (rows["lemon_pose"][:, 2] > 0.05)
    relative = rows["lemon_pose"][:, :3] - rows["plate_pose"][:, :3]
    return (
        good
        & finger
        & ~table
        & (relative[:, :2].norm(dim=-1) < 0.035)
        & (relative[:, 2] > 0.025)
        & (relative[:, 2] < 0.10)
    )


def collect(
    output: Path,
    *,
    per_type: int = 100,
    num_envs: int = 64,
    max_batches: int = 50,
    hold_steps: int = 20,
    settle_steps: int = 200,
    seed: int = 42,
    device: str = "cuda:0",
    cuda_graph_mode: str = "auto",
    proposal_workers: int = DEFAULT_PROPOSAL_WORKERS,
    nconmax: int | None = None,
    njmax: int | None = None,
    split: str = "development",
) -> None:
    if output.exists():
        raise FileExistsError(output)
    if min(per_type, num_envs, max_batches, hold_steps, settle_steps, proposal_workers) < 1:
        raise ValueError("Positive budgets are required")
    if split not in ("development", "train", "validation", "test"):
        raise ValueError(f"Unknown dataset split: {split}")
    criteria = AcceptanceConfig()
    cfg = make_env_cfg()
    action_cfg = cfg.actions[DEFAULT_ROBOT.action_name]
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    required_steps = math.ceil(criteria.stable_seconds / control_dt - 1e-9)
    if min(hold_steps, settle_steps) < required_steps:
        raise ValueError("Reset acceptance needs at least 0.5 seconds of physical hold")
    cfg.scene.num_envs = num_envs
    for name, value in (("nconmax", nconmax), ("njmax", njmax)):
        if value is not None:
            if value < 1:
                raise ValueError(f"{name} must be positive")
            setattr(cfg.sim, name, value)
    cfg.events.pop("reset_from_bank")
    cfg.events = {name: event for name, event in cfg.events.items() if event.mode != "startup"}
    cfg.terminations.pop("success")
    cfg.rewards = {}
    cfg.auto_reset = False
    cfg.episode_length_s = max(
        cfg.episode_length_s, (max(hold_steps, settle_steps) + 1) * control_dt
    )
    # Nominal physics during collection; training randomization, if introduced,
    # must be separately validated rather than hidden in the proposal bank.
    collected = {kind: {key: [] for key in FIELDS} for kind in RESET_TYPES}
    counts = dict.fromkeys(RESET_TYPES, 0)
    attempts = dict.fromkeys(RESET_TYPES, 0)
    executed_steps = {kind: [] for kind in RESET_TYPES}
    stage_seconds = {kind: [] for kind in RESET_TYPES}
    replay_counts = {kind: {"tested": 0, "passed": 0} for kind in RESET_TYPES}
    sampler = Sampler(seed, workers=proposal_workers)
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        free_joint_damping = joint_damping_metadata(env.sim.mj_model, "lemon/lemon_free")
        lemon_body = env.sim.mj_model.body("lemon/lemon")
        object_physics = {
            "mass_kg": float(lemon_body.mass[0]),
            "diagonal_inertia_kg_m2": lemon_body.inertia.tolist(),
        }
        DEFAULT_ROBOT.validate_model(env.sim.mj_model)
        joint_ids = [env.sim.mj_model.joint(f"robot/{name}").id for name in DEFAULT_ROBOT.joints]
        joint_limits = torch.tensor(
            env.sim.mj_model.jnt_range[joint_ids], device=device, dtype=torch.float32
        )
        env.reset(seed=seed)
        if cuda_graph_mode == "plain":
            if not device.startswith("cuda"):
                raise ValueError("Plain graph requires CUDA")
            env.sim.wp_model.opt.graph_conditional = False
            env.sim.step()
            wp.synchronize_device(env.sim.wp_device)
            env.sim.use_cuda_graph = True
            env.sim.create_graph()
            if env.sim.step_graph is None or env.sim.reset_graph is None:
                raise RuntimeError("Plain graph capture did not produce step/reset graphs")
        elif cuda_graph_mode != "auto":
            raise ValueError(f"Unknown graph mode: {cuda_graph_mode}")
        for category, kind in enumerate(RESET_TYPES):
            if category == 1:
                sampler.set_resting_poses(torch.cat(collected[RESET_TYPES[0]]["lemon_pose"]))
            for batch in range(max_batches):
                if counts[kind] >= per_type:
                    break
                batch_start = time.perf_counter()
                candidates = sampler.batch(kind, num_envs, max_attempts=500 * num_envs)
                proposal_end = time.perf_counter()
                env.reset(seed=seed + category * 100_000 + batch)
                write_states(env, {key: value.to(device) for key, value in candidates.items()})
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.sim.sense()
                start = snapshot(env)
                target_pos, target_quat = (
                    value.clone() for value in observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)
                )
                safe = _safe_contacts(env, sampler.cfg.max_penetration)
                safe &= within_workspace(start, sampler.workspace_corners)
                if category == 0:
                    safe &= ~_touch(env, "lemon_robot_contact")
                stable = torch.zeros(num_envs, dtype=torch.int32, device=device)
                # A released lemon needs time to roll to its natural resting
                # orientation. Do not relax its
                # initial-state speed thresholds just to accept it after 1 s.
                steps = settle_steps if category == 0 else hold_steps
                for _step in range(steps):
                    action = hold_action(
                        env,
                        target_pos,
                        target_quat,
                        gripper_command=1.0 if category == 0 else -1.0,
                        action_cfg=action_cfg,
                    )
                    _, _, terminal, timeout, _ = env.step(action)
                    if bool((terminal | timeout).any()):
                        raise RuntimeError("Unexpected episode end during reset acceptance")
                    if bool((env.sim.data.overflow != 0).any()):
                        raise RuntimeError("MuJoCo-Warp capacity overflow during reset acceptance")
                    safe &= _safe_contacts(env, sampler.cfg.max_penetration)
                    safe &= within_workspace(snapshot(env), sampler.workspace_corners)
                    eligible = accepted(
                        env,
                        category,
                        start,
                        max_penetration=sampler.cfg.max_penetration,
                        criteria=criteria,
                        joint_limits=joint_limits,
                    )
                    stable = torch.where(eligible, stable + 1, 0)
                    if category == 0 and int((safe & (stable >= required_steps)).sum()) >= min(
                        num_envs, per_type - counts[kind]
                    ):
                        break
                executed_steps[kind].append(_step + 1)
                good = safe & (stable >= required_steps)
                # Save the dynamically reached state, not a penetrated initial
                # pose that merely happened to settle into a good configuration.
                final = snapshot(env)
                wp.synchronize_device(env.sim.wp_device)
                physics_end = time.perf_counter()
                if category == 0:
                    # Natural rolling changes orientation. A reachable upright
                    # proposal is not proof the saved settled lemon is graspable.
                    lemon_poses = final["lemon_pose"].cpu().numpy()
                    plate_poses = final["plate_pose"].cpu().numpy()
                    proposed = good.nonzero()[:, 0]
                    cpu_indices = proposed.cpu().numpy()
                    reachable = sampler.settled_poses_reachable(
                        lemon_poses[cpu_indices], plate_poses[cpu_indices]
                    )
                    good[proposed] = torch.as_tensor(reachable, device=device)
                post_ik_end = time.perf_counter()
                replay_counts[kind]["tested"] += int(good.sum())
                good = reloaded_hold_mask(
                    env,
                    final,
                    good,
                    SPEC,
                    limits=joint_limits,
                    finger_upper_tolerance=criteria.finger_upper_tolerance,
                    action_cfg=action_cfg,
                    gripper_command=1.0 if category == 0 else -1.0,
                    seed=seed + 700_000 + category * 100_000 + batch,
                    steps=2 * required_steps,
                    required_steps=2 * required_steps,
                    acceptance=lambda replay_env, replay_start, category=category: accepted(
                        replay_env,
                        category,
                        replay_start,
                        max_penetration=sampler.cfg.max_penetration,
                        criteria=criteria,
                        joint_limits=joint_limits,
                    ),
                    safety=lambda replay_env: (
                        _safe_contacts(replay_env, sampler.cfg.max_penetration)
                        & within_workspace(snapshot(replay_env), sampler.workspace_corners)
                    ),
                )
                replay_counts[kind]["passed"] += int(good.sum())
                eligible_count = int(good.sum())
                replay_end = time.perf_counter()
                take = min(eligible_count, per_type - counts[kind])
                indices = good.nonzero()[:take, 0]
                if take:
                    for field in FIELDS:
                        collected[kind][field].append(final[field][indices].cpu())
                counts[kind] += take
                attempts[kind] += num_envs
                timing = {
                    "cpu_proposals": proposal_end - batch_start,
                    "physics_and_acceptance": physics_end - proposal_end,
                    "post_settling_ik": post_ik_end - physics_end,
                    "reloaded_hold": replay_end - post_ik_end,
                    "total": time.perf_counter() - batch_start,
                }
                stage_seconds[kind].append(timing)
                print(
                    f"{kind}: batch={batch + 1} eligible={eligible_count}/{num_envs} "
                    f"saved={take} total={counts[kind]}/{per_type}",
                    flush=True,
                )
                print(f"{kind}: batch_seconds={timing}", flush=True)
            if counts[kind] < per_type:
                raise RuntimeError(f"Incomplete {kind}: {counts[kind]}/{per_type}; no bank written")
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            sampler.close()
    bank = {
        "schema": SCHEMA,
        "contract": bank_contract(),
        "split": split,
        "provenance": {
            "method": "fresh_ik_static_contact_and_mjlab_hold",
            "workspace": workspace_metadata(),
            "seed": seed,
            "proposal_config": sampler.metadata,
            "hold_steps": hold_steps,
            "replay_validation": {
                "steps": 2 * required_steps,
                "required_stable_steps": 2 * required_steps,
                "counts": replay_counts,
                "saved_state": "before_replay_with_projected_fingers",
                "command": "fixed_initial_ee_pose",
                "seed_offset": 700_000,
            },
            "settle_steps_max": settle_steps,
            "executed_control_steps": executed_steps,
            "stage_seconds": stage_seconds,
            "control_dt": cfg.decimation * cfg.sim.mujoco.timestep,
            "accepted_consecutive_steps": required_steps,
            "saved_state": "final_physics_state_with_measured_velocities",
            "num_envs": num_envs,
            "physics_candidates": attempts,
            "cuda_graph_mode": cuda_graph_mode,
            "nconmax": cfg.sim.nconmax,
            "njmax": cfg.sim.njmax,
            "physics_dt": cfg.sim.mujoco.timestep,
            "decimation": cfg.decimation,
            "physics_randomization": False,
            "free_joint_damping": free_joint_damping,
            "object_physics": object_physics,
            "settled_task0_ik_rechecked": True,
            "hold_command": "fixed_initial_ee_pose_via_relative_osc_action",
            "controller": asdict(action_cfg),
            "acceptance": {
                **asdict(criteria),
                "penetration_max": sampler.cfg.max_penetration,
                "grasp_axis": "lemon_body_x_narrow_side",
            },
        },
        "states": {
            kind: {key: torch.cat(parts) for key, parts in rows.items()}
            for kind, rows in collected.items()
        },
    }
    DEFAULT_ROBOT.project_reset_fingers(
        bank, torch.tensor(sampler.limits, dtype=torch.float32), criteria.finger_upper_tolerance
    )
    bank["provenance"]["finger_position_projection"]["projected_contact_hold_revalidated"] = True
    validate_bank(bank, model=sampler.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(bank, stream)
    print(f"Saved {output}: {counts}", flush=True)


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Collect a hold-validated Lemon reset bank from IK/contact-checked proposals."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    parser.add_argument("--proposal-workers", type=int, default=DEFAULT_PROPOSAL_WORKERS)
    parser.add_argument("--nconmax", type=int)
    parser.add_argument("--njmax", type=int)
    parser.add_argument(
        "--split", choices=("development", "train", "validation", "test"), default="train"
    )
    collect(**vars(parser.parse_args(argv)))
