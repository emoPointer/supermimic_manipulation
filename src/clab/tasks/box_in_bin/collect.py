"""Collect a hold-validated BoxInBin reset bank (``clab resets collect --task box_in_bin``).

Candidates are procedural; ``--proposal-bank`` instead revalidates an existing bank.
Startup physics randomization is off during collection (training keeps it). Each candidate
holds for ``--hold-steps`` control steps (default 10, 0.5 s), then must survive a second
normal reset from its exact serialized state under the same hold actions and criteria.
The output is a new bank; existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from clab.controllers.pose import pose_target_action
from clab.observations.state import observation_pose
from clab.robots.parallel_kinematics import DEFAULT_PROPOSAL_WORKERS
from clab.scenes.workspace import contains_bounds, workspace_metadata

from .config import make_env_cfg
from .procedural import ProceduralSampler
from .reset_data import FIELDS, bank_contract, validate_bank
from .resets import world_pose
from .settings import BIN_OUTER_HALF_EXTENTS, BOX_HALF_EXTENTS, RESET_TYPES
from .variants import DEFAULT_ROBOT

HOLD_SECONDS = 0.5
SPLITS = ("development", "train", "validation", "test")


def compiled_physics_provenance(env) -> dict:
    """Read nominal compiled physics.

    Capacities are the requested Warp allocation limits, not contact counts or
    a capacity certificate. This is metadata only; model arrays are not changed.
    """
    model = env.sim.mj_model
    return {
        "gripper": DEFAULT_ROBOT.gripper_contract(model),
        "gripper_contact": DEFAULT_ROBOT.gripper_contact_contract(model),
        "contact_solver": {
            "cone": {0: "pyramidal", 1: "elliptic"}[int(model.opt.cone)],
            "impratio": float(model.opt.impratio),
        },
        "allocation_limits": {
            "nconmax": env.cfg.sim.nconmax,
            "njmax": env.cfg.sim.njmax,
        },
        "parameter_scope": "nominal compiled model; startup physics randomization disabled",
    }


def collection_env_cfg(**kwargs):
    """Nominal collection physics; do not mutate the training config factory."""
    cfg = make_env_cfg(**kwargs)
    cfg.events = {name: event for name, event in cfg.events.items() if event.mode != "startup"}
    # Without autoreset a diverged candidate stays non-finite; the hold rejects it.
    cfg.terminations.pop("physics_divergence")
    for group in cfg.observations.values():
        group.nan_policy = "sanitize"
    return cfg


def _within_workspace(env):
    """Require both oriented objects' complete outer bounds inside the shared XY region."""
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name, half in (("box", BOX_HALF_EXTENTS), ("bin", BIN_OUTER_HALF_EXTENTS)):
        pose = env.scene[name].data.root_link_pose_w
        corners = pose.new_tensor(list(product((-1.0, 1.0), repeat=3))) * pose.new_tensor(half)
        points = (
            quat_apply(
                pose[:, None, 3:].expand(-1, 8, -1), corners[None].expand(env.num_envs, -1, -1)
            )
            + (pose[:, :3] - env.scene.env_origins)[:, None]
        )
        valid &= contains_bounds(points.amin(dim=1), points.amax(dim=1))
    return valid


def required_stable_steps(control_dt: float) -> int:
    """Preserve physical acceptance time when the robot control frequency changes."""
    return math.ceil(HOLD_SECONDS / control_dt - 1e-9)


def _validate_hold(cfg, hold_steps: int) -> tuple[float, int]:
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    minimum = required_stable_steps(control_dt)
    if hold_steps < minimum:
        raise ValueError(
            f"hold_steps={hold_steps} covers only {hold_steps * control_dt:g} s; "
            f"BoxInBin acceptance requires at least {HOLD_SECONDS:g} s ({minimum} control steps)"
        )
    return control_dt, minimum


def _snapshot(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    robot, box, bin_asset = (env.scene[name] for name in ("robot", "box", "bin"))
    position, rotation = robot.data.root_link_pos_w, robot.data.root_link_quat_w
    poses = [
        torch.cat(subtract_frame_transforms(position, rotation, *pose.split((3, 4), -1)), -1)
        for pose in (box.data.root_link_pose_w, bin_asset.data.root_link_pose_w)
    ]
    return {
        "_ee_pos": env.sim.data.xpos[
            :, env.sim.mj_model.body(f"robot/{DEFAULT_ROBOT.ee_body}").id
        ].clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        # Acceptance uses world heights; exported FIELDS remain base-local.
        "_box_pos_w": box.data.root_link_pos_w.clone(),
        "box_pose": poses[0],
        "box_vel": torch.cat(
            (
                quat_apply_inverse(rotation, box.data.root_link_lin_vel_w),
                quat_apply_inverse(rotation, box.data.root_link_ang_vel_w),
            ),
            dim=-1,
        ),
        "bin_pose": poses[1],
    }


def _accepted(env: ManagerBasedRlEnv, kind: str, start: dict[str, torch.Tensor]) -> torch.Tensor:
    box = env.scene["box"]
    now = box.data.root_link_pos_w - env.scene.env_origins
    initial = start["_box_pos_w"] - env.scene.env_origins
    bin_pos = env.scene["bin"].data.root_link_pos_w - env.scene.env_origins
    finger_touch = (env.scene["left_finger_contact"].data.found[:, 0] > 0) & (
        env.scene["right_finger_contact"].data.found[:, 0] > 0
    )
    finite = torch.isfinite(now).all(-1) & torch.isfinite(box.data.root_link_quat_w).all(-1)
    drift = torch.linalg.vector_norm(now - initial, dim=-1)
    if kind == RESET_TYPES[0]:
        # Unattached object resting on the tabletop, separated from the bin.
        return finite & (drift < 0.02) & (now[:, 2] - 0.0515).abs().lt(0.01) & ~finger_touch
    if kind == RESET_TYPES[1]:
        return finite & (drift < 0.01) & (now[:, 2] - 0.0515).abs().lt(0.01) & finger_touch
    if kind == RESET_TYPES[2]:
        return finite & (drift < 0.05) & (now[:, 2] > 0.0715) & finger_touch
    delta = now[:, :2] - bin_pos[:, :2]
    return (
        finite
        & (drift < 0.025)
        & finger_touch
        & (delta[:, 0].abs() < 0.20)
        & (delta[:, 1].abs() < 0.16)
        & (now[:, 2] > 0.05)
        & (now[:, 2] < 0.15)
    )


def _box_bottom(pose: torch.Tensor) -> torch.Tensor:
    """Lowest oriented box corner relative to the local table top at Z=0."""
    w, x, y, z = pose[:, 3:].unbind(-1)
    row_z = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
    half = torch.tensor(BOX_HALF_EXTENTS, device=pose.device, dtype=pose.dtype)
    return pose[:, 2] - (row_z.abs() * half).sum(-1)


def _accepted_procedural(
    env: ManagerBasedRlEnv, kind: str, start: dict[str, torch.Tensor]
) -> torch.Tensor:
    box = env.scene["box"]
    now_pose = box.data.root_link_pose_w.clone()
    now_pose[:, :3] -= env.scene.env_origins
    bottom = _box_bottom(now_pose)
    initial = start["_box_pos_w"] - env.scene.env_origins
    drift = torch.linalg.vector_norm(now_pose[:, :3] - initial, dim=-1)
    speed = torch.linalg.vector_norm(box.data.root_link_lin_vel_w, dim=-1)
    spin = torch.linalg.vector_norm(box.data.root_link_ang_vel_w, dim=-1)
    ee_pos = env.sim.data.xpos[:, env.sim.mj_model.body(f"robot/{DEFAULT_ROBOT.ee_body}").id]
    ee_drift = torch.linalg.vector_norm(ee_pos - start["_ee_pos"], dim=-1)
    finger_touch = (env.scene["left_finger_contact"].data.found[:, 0] > 0) & (
        env.scene["right_finger_contact"].data.found[:, 0] > 0
    )
    bin_touch = env.scene["box_bin_contact"].data.found[:, 0] > 0
    good = (
        torch.isfinite(now_pose).all(-1)
        & torch.isfinite(speed)
        & torch.isfinite(spin)
        & (speed < 0.1)
        & (spin < 1.0)
        & (ee_drift < 0.1)
    )
    if kind == RESET_TYPES[0]:
        quat = now_pose[:, 3:]
        upright = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        return (
            good
            & (drift < 0.02)
            & (bottom.abs() < 0.01)
            & (upright > 0.93969)
            & ~finger_touch
            & ~bin_touch
        )
    if kind == RESET_TYPES[1]:
        return good & (drift < 0.01) & (bottom.abs() < 0.01) & finger_touch & ~bin_touch
    if kind == RESET_TYPES[2]:
        return good & (drift < 0.05) & (bottom > 0.015) & finger_touch & ~bin_touch
    bin_pos = env.scene["bin"].data.root_link_pos_w - env.scene.env_origins
    delta = now_pose[:, :2] - bin_pos[:, :2]
    return (
        good
        & (drift < 0.025)
        & finger_touch
        & bin_touch
        & (delta[:, 0].abs() < 0.20)
        & (delta[:, 1].abs() < 0.16)
    )


def _write_candidates(env: ManagerBasedRlEnv, rows: dict[str, torch.Tensor]) -> None:
    ids = torch.arange(env.num_envs, device=env.device)
    robot, box, bin_asset = (env.scene[name] for name in ("robot", "box", "bin"))
    robot.write_joint_state_to_sim(
        rows["joint_pos"].to(env.device), rows["joint_vel"].to(env.device), env_ids=ids
    )
    position, rotation = robot.data.root_link_pos_w, robot.data.root_link_quat_w
    box_pose = world_pose(rows["box_pose"].to(env.device), position, rotation)
    bin_pose = world_pose(rows["bin_pose"].to(env.device), position, rotation)
    velocity = rows["box_vel"].to(env.device)
    velocity = torch.cat(
        (quat_apply(rotation, velocity[:, :3]), quat_apply(rotation, velocity[:, 3:])), dim=-1
    )
    box.write_root_state_to_sim(torch.cat((box_pose, velocity), -1), env_ids=ids)
    bin_asset.write_mocap_pose_to_sim(bin_pose, env_ids=ids)
    robot.set_joint_velocity_target(
        torch.zeros((env.num_envs, len(DEFAULT_ROBOT.joints)), device=env.device), env_ids=ids
    )
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()


def _procedural_hold(env, kind, action, hold_steps, stable_window_steps):
    """Hold the initial end-effector pose with unchanged physical acceptance criteria."""
    start = _snapshot(env)
    target = tuple(v.clone() for v in observation_pose(env, "robot", DEFAULT_ROBOT.ee_body))
    action_cfg = env.cfg.actions[DEFAULT_ROBOT.action_name]
    clear = env.scene["box_robot_contact"].data.found[:, 0] == 0
    workspace_valid = _within_workspace(env)
    persistent_touch = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    stable_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    finite = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(hold_steps):
        hold_action = pose_target_action(
            *observation_pose(env, "robot", DEFAULT_ROBOT.ee_body),
            *target,
            action_cfg=action_cfg,
            gripper_command=0.0,
        )
        hold_action[:, 6] = action[:, 6]
        _, _, terminated, truncated, _ = env.step(hold_action)
        if bool((terminated | truncated).any()) or bool((env.sim.data.overflow != 0).any()):
            raise RuntimeError(f"Episode end or capacity overflow during {kind} hold")
        qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
        finite &= torch.isfinite(qpos).all(-1) & torch.isfinite(qvel).all(-1)
        workspace_valid &= _within_workspace(env)
        if kind != RESET_TYPES[0]:
            persistent_touch &= (env.scene["left_finger_contact"].data.found[:, 0] > 0) & (
                env.scene["right_finger_contact"].data.found[:, 0] > 0
            )
        robot_speed = env.scene["robot"].data.joint_vel.abs().sum(-1)
        box_speed = env.scene["box"].data.root_link_lin_vel_w.abs().sum(-1)
        box_spin = env.scene["box"].data.root_link_ang_vel_w.abs().sum(-1)
        stable = (robot_speed < 5.0) & (box_speed < 0.1) & (box_spin < 1.0)
        stable_steps = torch.where(stable, stable_steps + 1, 0)
    good = (
        _accepted_procedural(env, kind, start)
        & persistent_touch
        & (stable_steps >= stable_window_steps)
        & workspace_valid
        & finite
    )
    if kind == RESET_TYPES[0]:
        good &= clear & (env.scene["box_robot_contact"].data.found[:, 0] == 0)
    return start, good


def collect_procedural(
    *,
    output: Path,
    per_type: int,
    num_envs: int,
    max_batches: int,
    hold_steps: int,
    seed: int,
    device: str,
    max_proposals_per_batch: int,
    proposal_workers: int = DEFAULT_PROPOSAL_WORKERS,
    nconmax: int = 24,
    njmax: int = 128,
    cuda_graph_mode: str = "auto",
    split: str = "train",
) -> dict[str, int]:
    """Generate and validate the four categories from procedural candidates."""
    if split not in SPLITS:
        raise ValueError(f"Unknown dataset split: {split}")
    if output.exists():
        raise FileExistsError(output)
    if min(per_type, num_envs, max_batches, hold_steps) < 1:
        raise ValueError("per_type, num_envs, max_batches, and hold_steps must be positive")
    if max_proposals_per_batch < num_envs:
        raise ValueError("max_proposals_per_batch must be at least num_envs")
    if min(nconmax, njmax) < 1:
        raise ValueError("Contact capacities must be positive")
    cfg = collection_env_cfg()
    control_dt, stable_window_steps = _validate_hold(cfg, hold_steps)
    sampler = ProceduralSampler(seed, workers=proposal_workers)
    cfg.scene.num_envs = num_envs
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name="box_robot_contact",
            primary=ContactMatch(mode="body", pattern="box", entity="box"),
            secondary=ContactMatch(mode="subtree", pattern=DEFAULT_ROBOT.root_body, entity="robot"),
            fields=("found",),
            reduce="maxforce",
        ),
    )
    cfg.sim.nconmax = nconmax
    cfg.sim.njmax = njmax
    cfg.events.pop("reset_from_bank")
    cfg.terminations.pop("success")  # Hold acceptance, not episode success, decides validity.
    cfg.rewards = {}  # Reward terms are not used for reset-state validation.
    cfg.episode_length_s = max(
        cfg.episode_length_s, hold_steps * cfg.decimation * cfg.sim.mujoco.timestep + 1
    )
    cfg.auto_reset = False
    counts = {kind: 0 for kind in RESET_TYPES}
    replay_counts = {kind: {"tested": 0, "passed": 0} for kind in RESET_TYPES}
    collected = {kind: {field: [] for field in FIELDS} for kind in RESET_TYPES}
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        compiled_physics = compiled_physics_provenance(env)
        if cuda_graph_mode == "plain":
            if not device.startswith("cuda"):
                raise ValueError("Plain CUDA Graph requires a CUDA device")
            env.sim.wp_model.opt.graph_conditional = False
            env.reset(seed=seed)  # Compile reset kernels before graph capture.
            env.sim.step()
            wp.synchronize_device(env.sim.wp_device)
            env.sim.use_cuda_graph = True
            env.sim.create_graph()
            if env.sim.step_graph is None or env.sim.reset_graph is None:
                raise RuntimeError("Plain CUDA Graph capture failed")
            env.reset()
            print("Plain CUDA Graph active for reset collection", flush=True)
        elif cuda_graph_mode != "auto":
            raise ValueError(f"Unsupported CUDA Graph mode: {cuda_graph_mode}")
        for type_id, kind in enumerate(RESET_TYPES):
            for batch in range(max_batches):
                if counts[kind] >= per_type:
                    break
                proposal_start = time.perf_counter()
                candidates = sampler.batch(kind, num_envs, max_proposals_per_batch)
                proposal_seconds = time.perf_counter() - proposal_start
                physics_start = time.perf_counter()
                env.reset(seed=seed + 101 * type_id + batch)
                _write_candidates(env, candidates)
                action = torch.zeros((num_envs, 7), device=device)
                action[:, 6] = 1.0 if type_id == 0 else -1.0
                start, good = _procedural_hold(
                    env,
                    kind,
                    action,
                    hold_steps,
                    stable_window_steps,
                )
                replay_counts[kind]["tested"] += int(good.sum())
                env.reset(seed=seed + 700_000 + 101 * type_id + batch)
                _write_candidates(env, start)
                _, replay_good = _procedural_hold(
                    env,
                    kind,
                    action,
                    max(hold_steps, 2 * stable_window_steps),
                    max(hold_steps, 2 * stable_window_steps),
                )
                good &= replay_good
                replay_counts[kind]["passed"] += int(good.sum())
                eligible = int(good.sum().item())
                take = min(per_type - counts[kind], eligible)
                if take:
                    indices = good.nonzero(as_tuple=False)[:take, 0]
                    for field in FIELDS:
                        collected[kind][field].append(start[field][indices].cpu())
                    counts[kind] += take
                print(
                    f"{kind}: batch {batch + 1}, eligible {eligible}/{num_envs}, "
                    f"accepted {counts[kind]}/{per_type}; "
                    f"proposal={proposal_seconds:.2f}s physics={time.perf_counter() - physics_start:.2f}s",
                    flush=True,
                )
            if counts[kind] < per_type:
                raise RuntimeError(
                    f"Insufficient {kind} states: {counts[kind]}/{per_type}; no output written"
                )
    finally:
        env.close()
        if isinstance(sampler, ProceduralSampler):
            sampler.close()
    states = {
        kind: {field: torch.cat(chunks, 0) for field, chunks in rows.items()}
        for kind, rows in collected.items()
    }
    bank = {
        "schema": "clab.box_in_bin.reset_bank/1",
        "states": states,
        "contract": bank_contract(),
        "split": split,
        "provenance": {
            "workspace": workspace_metadata(),
            "proposal_workers": proposal_workers,
            "proposal_seed_schedule": "base_seed_ordered_chunk_seedsequence"
            if proposal_workers > 1
            else "serial_generator",
            "proposal_chunking": "balanced_contiguous_chunks_up_to_worker_count",
            "method": "procedural_mjlab_contact_hold",
            "candidate_source": "mujoco_fk",
            "physics_randomization": False,
            "compiled_physics": compiled_physics,
            "hold_command": "fixed_initial_ee_pose_via_relative_osc_action",
            "replay_validation": {
                "steps": max(hold_steps, 2 * stable_window_steps),
                "required_stable_steps": max(hold_steps, 2 * stable_window_steps),
                "counts": replay_counts,
                "saved_state": "input_before_replay",
                "command": "fixed_initial_ee_pose_and_same_gripper",
                "seed_offset": 700_000,
            },
            "floating_grasp_source": "mujoco_fk",
            "seed": seed,
            "hold_steps": hold_steps,
            "hold_seconds": hold_steps * control_dt,
            "accepted_consecutive_steps": stable_window_steps,
            "stable_seconds": HOLD_SECONDS,
            "control_dt": control_dt,
            "physics_dt": cfg.sim.mujoco.timestep,
            "decimation": cfg.decimation,
            "controller": asdict(cfg.actions[DEFAULT_ROBOT.action_name]),
            "num_envs": num_envs,
            "max_proposals_per_batch": max_proposals_per_batch,
            "robot_table_penetration_limit_m": 0.001,
            "initial_box_robot_contact_filter": True,
            "ungrasped_box_xy_source": "collision_checked_resting_grasp",
        },
    }
    validate_bank(bank)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    return counts


def collect(
    *,
    proposal_bank: Path,
    output: Path,
    per_type: int,
    num_envs: int,
    max_batches: int,
    hold_steps: int,
    seed: int,
    device: str,
    split: str = "train",
) -> dict[str, int]:
    if split not in SPLITS:
        raise ValueError(f"Unknown dataset split: {split}")
    if output.exists():
        raise FileExistsError(output)
    if min(per_type, num_envs, max_batches, hold_steps) < 1:
        raise ValueError("per_type, num_envs, max_batches, and hold_steps must be positive")
    validate_bank(torch.load(proposal_bank, map_location="cpu", weights_only=True))
    torch.manual_seed(seed)
    cfg = collection_env_cfg(reset_bank=proposal_bank)
    control_dt, _ = _validate_hold(cfg, hold_steps)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = max(
        cfg.episode_length_s, hold_steps * cfg.decimation * cfg.sim.mujoco.timestep + 1
    )
    cfg.auto_reset = False
    collected = {kind: {field: [] for field in FIELDS} for kind in RESET_TYPES}
    counts = {kind: 0 for kind in RESET_TYPES}
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        reset_term = env.event_manager.get_term_cfg("reset_from_bank").func
        for type_id, kind in enumerate(RESET_TYPES):
            reset_term.type_override = type_id
            for batch in range(max_batches):
                if counts[kind] >= per_type:
                    break
                env.reset(seed=seed + 101 * type_id + batch)
                start = _snapshot(env)
                workspace_valid = _within_workspace(env)
                action = torch.zeros((num_envs, 7), device=device)
                action[:, 6] = 1.0 if type_id == 0 else -1.0
                for _ in range(hold_steps):
                    _, _, terminated, truncated, _ = env.step(action)
                    workspace_valid &= _within_workspace(env)
                    if bool((terminated | truncated).any()):
                        raise RuntimeError(f"Unexpected episode termination during {kind} hold")
                good = _accepted(env, kind, start) & workspace_valid
                take = min(per_type - counts[kind], int(good.sum().item()))
                if take:
                    indices = good.nonzero(as_tuple=False)[:take, 0]
                    for field in FIELDS:
                        collected[kind][field].append(start[field][indices].cpu())
                    counts[kind] += take
                print(f"{kind}: batch {batch + 1}, accepted {counts[kind]}/{per_type}", flush=True)
            if counts[kind] < per_type:
                raise RuntimeError(
                    f"Insufficient {kind} candidates: {counts[kind]}/{per_type}; no output written"
                )
    finally:
        env.close()
    states = {
        kind: {field: torch.cat(chunks, dim=0) for field, chunks in rows.items()}
        for kind, rows in collected.items()
    }
    bank = {
        "schema": "clab.box_in_bin.reset_bank/1",
        "states": states,
        "contract": bank_contract(),
        "split": split,
        "provenance": {
            "method": "mujoco_proposal_replay",
            "physics_randomization": False,
            "workspace": workspace_metadata(),
            "hold_steps": hold_steps,
            "hold_seconds": hold_steps * control_dt,
            "control_dt": control_dt,
            "physics_dt": cfg.sim.mujoco.timestep,
            "decimation": cfg.decimation,
            "controller": asdict(cfg.actions[DEFAULT_ROBOT.action_name]),
            "seed": seed,
        },
    }
    validate_bank(bank)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    return counts


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Collect a hold-validated BoxInBin reset bank from procedural candidates, "
        "or revalidate an existing bank.",
    )
    parser.add_argument("--proposal-bank", type=Path, help="Revalidate an existing bank's rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=10000)
    parser.add_argument("--max-batches", type=int, default=5000)
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=required_stable_steps(DEFAULT_ROBOT.control_dt),
        help="Number of current 20 Hz control steps; minimum/default 10 = 0.5 seconds",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-proposals-per-batch",
        type=int,
        default=100_000_000,
        help="Total candidate attempts per batch, split across ordered worker chunks",
    )
    parser.add_argument("--proposal-workers", type=int, default=DEFAULT_PROPOSAL_WORKERS)
    parser.add_argument("--nconmax", type=int, default=24)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    parser.add_argument("--split", choices=SPLITS, default="train")
    args = parser.parse_args(argv)
    if args.proposal_bank is None:
        print(
            collect_procedural(
                output=args.output,
                per_type=args.per_type,
                num_envs=args.num_envs,
                max_batches=args.max_batches,
                hold_steps=args.hold_steps,
                seed=args.seed,
                device=args.device,
                max_proposals_per_batch=args.max_proposals_per_batch,
                proposal_workers=args.proposal_workers,
                nconmax=args.nconmax,
                njmax=args.njmax,
                cuda_graph_mode=args.cuda_graph_mode,
                split=args.split,
            )
        )
    else:
        print(
            collect(
                proposal_bank=args.proposal_bank,
                output=args.output,
                per_type=args.per_type,
                num_envs=args.num_envs,
                max_batches=args.max_batches,
                hold_steps=args.hold_steps,
                seed=args.seed,
                device=args.device,
                split=args.split,
            )
        )
