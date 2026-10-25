"""Collect MugHang reset states, including a physically executed threading path.

Run through ``clab resets collect --task mug_hang``. The last category starts with the mug
outside the bar tip and moves it along the rod under ordinary OSC actions while holding a
verified rim grasp; neither mug nor robot is teleported. ``--insertion-per-stage N`` splits
it into N outside-tip and N threaded states (``--per-type 2*N``), labelled beside the
unchanged state tensors. Outside-tip alignment allows 2 cm transverse handle-anchor error
and no tree contact. All geometry envelopes must stay inside the shared workspace.

``--progress-path PATH`` atomically keeps accepted rows after each batch and ``--resume``
continues an identical run; the progress file is not a bank. ``--source-bank`` revalidates
a compatible bank under current control and collects only the missing rows; its threading
witnesses stay historical. Grasped categories prefetch bounded CPU batches while physics
runs, without changing acceptance.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from clab.controllers.pose import pose_target_action
from clab.logging.model import joint_damping_metadata
from clab.observations.state import observation_pose, pose
from clab.robots.parallel_kinematics import DEFAULT_PROPOSAL_WORKERS
from clab.scenes.workspace import poses_within_workspace, workspace_metadata
from clab.tasks.rigid_resets import reloaded_hold_mask

from .config import make_env_cfg
from .mdp import contact_found, handle_geometry
from .proposals import Sampler
from .props import BRANCH_LENGTH, BRANCH_OUTER, HANDLE_ANCHOR
from .reset_data import FIELDS, SCHEMA, SPEC, WIDTHS, bank_contract, validate_bank
from .resets import snapshot, write_states
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


@dataclass(frozen=True)
class AcceptanceConfig:
    max_penetration: float = 0.001
    stable_seconds: float = 0.5
    linear_speed_max: float = 0.03
    angular_speed_max: float = 0.3
    resting_linear_speed_max: float = 0.003
    resting_angular_speed_max: float = 0.05
    joint_speed_max: float = 2.0
    finger_upper_tolerance: float = DEFAULT_ROBOT.reset_finger_upper_tolerance
    object_drift_max: float = 0.02
    minimum_thread_displacement: float = 0.04
    outside_tip_clearance: tuple[float, float] = (0.015, 0.030)
    outside_tip_transverse_jitter: float = 0.003
    outside_tip_transverse_max: float = 0.02


def _safe(env, criteria):
    good = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    for name in ("robot_contact", "mug_table_contact", "mug_tree_contact"):
        data = env.scene[name].data
        good &= ~((data.found > 0) & (data.dist < -criteria.max_penetration)).any(-1)
    return good


def _anchor_tree(env):
    mug_pos, mug_quat = pose(env, "mug")
    tree_pos, tree_quat = pose(env, "tree")
    anchor = torch.tensor(HANDLE_ANCHOR, device=env.device).expand_as(mug_pos)
    return quat_apply_inverse(tree_quat, mug_pos + quat_apply(mug_quat, anchor) - tree_pos)


def accepted(env, category, start, criteria, limits, *, insertion_stage="threaded"):
    rows = snapshot(env)
    good = _safe(env, criteria)
    for value in rows.values():
        good &= torch.isfinite(value).all(-1)
    good &= DEFAULT_ROBOT.reset_joint_position_mask(
        rows["joint_pos"], limits, criteria.finger_upper_tolerance
    )
    good &= rows["joint_vel"].abs().amax(-1) < criteria.joint_speed_max
    speed = rows["mug_vel"][:, :3].norm(dim=-1)
    angular = rows["mug_vel"][:, 3:].norm(dim=-1)
    good &= (speed < criteria.linear_speed_max) & (angular < criteria.angular_speed_max)
    dual = contact_found(env, "left_finger_contact") & contact_found(env, "right_finger_contact")
    table = contact_found(env, "mug_table_contact")
    tree = contact_found(env, "mug_tree_contact")
    if category < 3:
        good &= (rows["mug_pose"][:, :3] - start["mug_pose"][:, :3]).norm(
            dim=-1
        ) < criteria.object_drift_max
    if category == 0:
        return (
            good
            & table
            & ~tree
            & ~contact_found(env, "mug_robot_contact")
            & (speed < criteria.resting_linear_speed_max)
            & (angular < criteria.resting_angular_speed_max)
        )
    if category == 1:
        return good & dual & table & ~tree
    if category == 2:
        return good & dual & ~table & ~tree & (rows["mug_pose"][:, 2] > 0.06)
    radial, projected, alignment = handle_geometry(
        rows["mug_pose"][:, :3],
        rows["mug_pose"][:, 3:],
        rows["tree_pose"][:, :3],
        rows["tree_pose"][:, 3:],
    )
    if insertion_stage == "outside_tip":
        anchor = _anchor_tree(env)
        return (
            good
            & dual
            & ~table
            & ~tree
            & (projected > BRANCH_LENGTH + 0.005)
            & (anchor[:, 0] > BRANCH_OUTER[0] + 0.008)
            & (anchor[:, 0] < BRANCH_OUTER[0] + 0.045)
            & (
                (anchor[:, 1:] - anchor.new_tensor(BRANCH_OUTER[1:])).norm(dim=-1)
                <= criteria.outside_tip_transverse_max
            )
            & (alignment >= 0.75)
        )
    if insertion_stage != "threaded":
        raise ValueError(f"Unknown MugHang insertion stage: {insertion_stage}")
    return (
        good
        & dual
        & ~table
        & (radial <= 0.018)
        & (projected >= 0.015)
        & (projected <= 0.07)
        & (alignment >= 0.75)
    )


def load_reuse_source(path, *, split, insertion_per_stage, criteria):
    """Reject incompatible source semantics before allocating workers or a GPU."""
    bank = torch.load(path, map_location="cpu", weights_only=True)
    validate_bank(bank)
    provenance = bank["provenance"]
    if bank["split"] != split or provenance.get("workspace") != workspace_metadata():
        raise ValueError("Source split and workspace must match this collection")
    expected = asdict(criteria)
    # Older collectors did not permit finger overshoot; this new projection is
    # bounded and independently replayed. All original physical criteria match.
    expected.pop("finger_upper_tolerance")
    if any(provenance.get("acceptance", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Source MugHang acceptance criteria do not match")
    if provenance.get("method") != "fresh_rim_ik_and_osc_threading":
        raise ValueError("Source requires an explicit original physical threading witness")
    if (insertion_per_stage is not None) != ("insertion_subtypes" in bank):
        raise ValueError("Source insertion subtype mode must match collection")
    return bank


def padded_candidates(rows, num_envs, device):
    """Pad with copies for fixed capacity; only original rows may be accepted."""
    count = len(rows["joint_pos"])
    if not 0 < count <= num_envs:
        raise ValueError("Candidate count must be within simulation capacity")
    indices = torch.arange(num_envs).clamp(max=count - 1)
    return (
        {key: value[indices].to(device) for key, value in rows.items()},
        torch.arange(num_envs, device=device) < count,
    )


def atomic_save(value, path, *, replace=False):
    """Publish a fully serialized file; immutable outputs use no-clobber linking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_progress(path, expected_config):
    """Read task-local partial rows without passing them off as a complete bank."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("kind") != "mug_hang_collection_progress":
        raise ValueError("Not a MugHang collection progress file")
    if value.get("config") != expected_config:
        raise ValueError("MugHang progress configuration/source differs; do not mix collections")
    if value.get("contract") != bank_contract():
        raise ValueError("MugHang progress state contract differs")
    states, counts, indices = value["states"], value["counts"], value["source_indices"]
    if (
        tuple(states) != RESET_TYPES
        or tuple(counts) != RESET_TYPES
        or tuple(indices) != RESET_TYPES
    ):
        raise ValueError("MugHang progress categories differ")
    for kind, rows in states.items():
        count = counts[kind]
        if type(count) is not int or not 0 <= count <= expected_config["per_type"]:
            raise ValueError(f"Invalid MugHang progress count: {kind}")
        if tuple(rows) != FIELDS:
            raise ValueError(f"Invalid MugHang progress fields: {kind}")
        for field, width in zip(FIELDS, WIDTHS, strict=True):
            tensor = rows[field]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.shape != (count, width)
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"Invalid MugHang progress tensor: {kind}.{field}")
        for field in ("mug_pose", "tree_pose"):
            if count and not torch.allclose(
                rows[field][:, 3:].norm(dim=-1), torch.ones(count), atol=1e-3, rtol=0
            ):
                raise ValueError(f"Invalid MugHang progress quaternion: {kind}.{field}")
        index = indices[kind]
        if (
            not isinstance(index, torch.Tensor)
            or index.dtype != torch.int64
            or index.shape != (count,)
            or not bool((index >= -1).all())
        ):
            raise ValueError(f"Invalid MugHang progress source indices: {kind}")
        if expected_config["source"] is None and bool((index != -1).any()):
            raise ValueError("Fresh MugHang progress cannot contain historical source indices")
    labels = value["insertion_labels"]
    mixed = expected_config["insertion_per_stage"] is not None
    label_count = counts[RESET_TYPES[3]] if mixed else 0
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.int64
        or labels.shape != (label_count,)
        or not bool(((labels == 0) | (labels == 1)).all())
    ):
        raise ValueError("Invalid MugHang progress insertion labels")
    for category, kind in enumerate(RESET_TYPES):
        stages = ("outside_tip", "threaded") if category == 3 and mixed else ("threaded",)
        total = 0
        for stage in stages:
            entry = value["stages"].get(f"{kind}/{stage}", {})
            for name in ("retained", "source_next_offset", "next_batch"):
                if type(entry.get(name)) is not int or entry[name] < 0:
                    raise ValueError("Invalid MugHang progress stage cursor")
            target = (
                expected_config["insertion_per_stage"]
                if category == 3 and mixed
                else expected_config["per_type"]
            )
            if entry["retained"] > target or entry["next_batch"] > expected_config["max_batches"]:
                raise ValueError("MugHang progress exceeds its collection budgets")
            if (
                type(entry.get("source_complete")) is not bool
                or type(entry.get("complete")) is not bool
            ):
                raise ValueError("Invalid MugHang progress stage status")
            if entry["complete"] and entry["retained"] != target:
                raise ValueError("Incomplete MugHang progress stage marked complete")
            if (
                category == 3
                and mixed
                and entry["retained"] != int((labels == int(stage == "threaded")).sum())
            ):
                raise ValueError("MugHang progress subtype and stage counts differ")
            total += entry["retained"]
        if total != counts[kind]:
            raise ValueError("MugHang progress stage and category counts differ")
    return value


def validate_source_cursors(progress, source, jobs, sampler, criteria, num_envs):
    """Reject source cursors outside the inspected prefix before GPU allocation."""
    limits = torch.as_tensor(sampler.limits)
    mixed = "insertion_subtypes" in source
    for category, kind, stage, target_count in jobs:
        key = f"{kind}/{stage}"
        cursor = progress["stages"][key]
        rows = source["states"][kind]
        selected = DEFAULT_ROBOT.reset_joint_position_mask(
            rows["joint_pos"], limits, criteria.finger_upper_tolerance
        )
        selected &= poses_within_workspace(rows, sampler.workspace_corners)
        retained = progress["source_indices"][kind] >= 0
        if category == 3 and mixed:
            subtype = int(stage == "threaded")
            selected &= source["insertion_subtypes"]["labels"] == subtype
            retained &= progress["insertion_labels"] == subtype
        source_ids = selected.nonzero()[:, 0]
        offset = cursor["source_next_offset"]
        if offset > len(source_ids) or (offset != len(source_ids) and offset % num_envs):
            raise ValueError(f"Invalid MugHang progress source cursor boundary: {key}")
        kept_ids = progress["source_indices"][kind][retained]
        if not bool(torch.isin(kept_ids, source_ids[:offset]).all()):
            raise ValueError(f"MugHang progress retains uninspected source rows: {key}")
        report = progress["reuse_counts"].get(key, {})
        if report.get("hold_tested", 0) != offset or report.get("retained", 0) != len(kept_ids):
            raise ValueError(f"MugHang progress source cursor and report disagree: {key}")
        if cursor["source_complete"] and offset < len(source_ids) and len(kept_ids) != target_count:
            raise ValueError(f"MugHang progress skipped remaining source rows: {key}")


def collect(
    output: Path,
    *,
    per_type=100,
    num_envs=64,
    max_batches=50,
    hold_steps=20,
    settle_steps=200,
    thread_steps=60,
    seed=42,
    proposal_workers=DEFAULT_PROPOSAL_WORKERS,
    proposal_attempts_per_state=20000,
    device="cuda:0",
    cuda_graph_mode="auto",
    split="development",
    nconmax=None,
    njmax=None,
    insertion_per_stage=None,
    source_bank: Path | None = None,
    progress_path: Path | None = None,
    resume=False,
):
    output = Path(output)
    source_bank = None if source_bank is None else Path(source_bank)
    if output.exists():
        raise FileExistsError(output)
    if resume and progress_path is None:
        raise ValueError("--resume requires --progress-path")
    if progress_path is not None:
        progress_path = Path(progress_path)
        if progress_path.resolve() in {
            output.resolve(),
            None if source_bank is None else source_bank.resolve(),
        }:
            raise ValueError("Progress, output and source must use separate paths")
        if progress_path.exists() and not resume:
            raise FileExistsError(f"{progress_path}; use --resume to continue explicitly")
        if resume and not progress_path.is_file():
            raise FileNotFoundError(progress_path)
    if (
        min(
            per_type,
            num_envs,
            max_batches,
            hold_steps,
            settle_steps,
            thread_steps,
            proposal_workers,
            proposal_attempts_per_state,
        )
        < 1
    ):
        raise ValueError("All collection budgets must be positive")
    if split not in ("development", "train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    criteria = AcceptanceConfig()
    if insertion_per_stage is not None and (
        insertion_per_stage < 1 or per_type != 2 * insertion_per_stage
    ):
        raise ValueError("Mixed insertion requires per_type == 2 * insertion_per_stage > 0")
    source = (
        None
        if source_bank is None
        else load_reuse_source(
            source_bank, split=split, insertion_per_stage=insertion_per_stage, criteria=criteria
        )
    )
    cfg = make_env_cfg()
    action_cfg = cfg.actions[DEFAULT_ROBOT.action_name]
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    required = math.ceil(criteria.stable_seconds / dt - 1e-9)
    if min(hold_steps, settle_steps) < required:
        raise ValueError("At least 0.5 s of consecutive stable hold is required")
    cfg.scene.num_envs = num_envs
    cfg.events = {
        key: value
        for key, value in cfg.events.items()
        if key != "reset_from_bank" and value.mode != "startup"
    }
    cfg.terminations.pop("success")
    cfg.rewards = {}
    cfg.auto_reset = False
    cfg.episode_length_s = max(
        cfg.episode_length_s, (settle_steps + thread_steps + 2 * hold_steps + 2) * dt
    )
    for name, value in (("nconmax", nconmax), ("njmax", njmax)):
        if value is not None:
            if value < 1:
                raise ValueError(f"{name} must be positive")
            setattr(cfg.sim, name, value)
    jobs = [(i, kind, "threaded", per_type) for i, kind in enumerate(RESET_TYPES)]
    if insertion_per_stage is not None:
        jobs[-1:] = [
            (3, RESET_TYPES[3], "outside_tip", insertion_per_stage),
            (3, RESET_TYPES[3], "threaded", insertion_per_stage),
        ]
    source_identity = None
    if source_bank is not None:
        source_stat = source_bank.stat()
        source_identity = {
            "path": str(source_bank.resolve()),
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        }
    progress_config = {
        "output": str(output.resolve()),
        "source": source_identity,
        "per_type": per_type,
        "num_envs": num_envs,
        "max_batches": max_batches,
        "hold_steps": hold_steps,
        "settle_steps": settle_steps,
        "thread_steps": thread_steps,
        "seed": seed,
        "proposal_workers": proposal_workers,
        "proposal_attempts_per_state": proposal_attempts_per_state,
        "split": split,
        "insertion_per_stage": insertion_per_stage,
        "cuda_graph_mode": cuda_graph_mode,
        "nconmax": cfg.sim.nconmax,
        "njmax": cfg.sim.njmax,
        "acceptance": asdict(criteria),
        "controller": asdict(action_cfg),
        "physics_dt": cfg.sim.mujoco.timestep,
        "decimation": cfg.decimation,
        "workspace": workspace_metadata(),
        "candidate_replay": {"steps": hold_steps, "required_consecutive_steps": required},
    }
    progress = load_progress(progress_path, progress_config) if resume else None
    if progress is not None and source is not None:
        for kind, indices in progress["source_indices"].items():
            retained = indices >= 0
            ids = indices[retained]
            original = source["states"][kind]
            if len(ids) and (
                int(ids.max()) >= len(original["joint_pos"]) or len(ids.unique()) != len(ids)
            ):
                raise ValueError(f"Invalid MugHang progress original source rows: {kind}")
            for field in FIELDS:
                if not torch.equal(progress["states"][kind][field][retained], original[field][ids]):
                    raise ValueError(
                        f"MugHang progress changed historical source field: {kind}.{field}"
                    )
            if kind == RESET_TYPES[3] and insertion_per_stage is not None:
                expected_labels = source["insertion_subtypes"]["labels"][ids]
                if not torch.equal(progress["insertion_labels"][retained], expected_labels):
                    raise ValueError("MugHang progress changed historical source subtype labels")
    started = perf_counter()
    sampler = Sampler(seed, workers=proposal_workers)
    collected = {kind: {field: [] for field in FIELDS} for kind in RESET_TYPES}
    counts = dict.fromkeys(RESET_TYPES, 0)
    insertion_labels = []
    source_indices = {kind: [] for kind in RESET_TYPES}
    reuse_counts = {}
    candidates_count = dict.fromkeys(RESET_TYPES, 0)
    proposal_attempts = dict.fromkeys(RESET_TYPES, 0)
    timings = {
        kind: {"proposal_seconds": 0.0, "physics_seconds": 0.0, "postsettle_seconds": 0.0}
        for kind in RESET_TYPES
    }
    stages = {
        f"{kind}/{stage}": {
            "retained": 0,
            "source_next_offset": 0,
            "source_complete": source is None,
            "next_batch": 0,
            "complete": False,
        }
        for _, kind, stage, _ in jobs
    }
    previous_elapsed = 0.0
    resumed_from_progress = resume
    progress_written = resume
    if progress is not None:
        collected = {
            kind: {field: [value] if len(value) else [] for field, value in rows.items()}
            for kind, rows in progress["states"].items()
        }
        counts = progress["counts"]
        source_indices = {
            kind: [value] if len(value) else []
            for kind, value in progress["source_indices"].items()
        }
        insertion_labels = (
            [progress["insertion_labels"]] if len(progress["insertion_labels"]) else []
        )
        reuse_counts = progress["reuse_counts"]
        candidates_count = progress["physics_candidates"]
        proposal_attempts = progress["proposal_attempts"]
        timings = progress["timings"]
        stages = progress["stages"]
        previous_elapsed = progress["elapsed_seconds"]
        sampler.load_state_dict(progress["sampler"])
        print(f"Resuming {progress_path}: {counts}", flush=True)

    def checkpoint():
        nonlocal progress_written
        if progress_path is None:
            return
        value = {
            "kind": "mug_hang_collection_progress",
            "contract": bank_contract(),
            "config": progress_config,
            "states": {
                kind: {
                    field: torch.cat(parts)
                    if parts
                    else torch.empty((0, width), dtype=torch.float32)
                    for (field, parts), width in zip(rows.items(), WIDTHS, strict=True)
                }
                for kind, rows in collected.items()
            },
            "counts": counts,
            "source_indices": {
                kind: torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)
                for kind, parts in source_indices.items()
            },
            "insertion_labels": torch.cat(insertion_labels)
            if insertion_labels
            else torch.empty(0, dtype=torch.int64),
            "reuse_counts": reuse_counts,
            "physics_candidates": candidates_count,
            "proposal_attempts": proposal_attempts,
            "timings": timings,
            "stages": stages,
            "sampler": sampler.state_dict(),
            "elapsed_seconds": previous_elapsed + perf_counter() - started,
        }
        atomic_save(value, progress_path, replace=progress_written)
        progress_written = True

    env = None
    candidate_stream = None
    try:
        if progress is not None and source is not None:
            validate_source_cursors(progress, source, jobs, sampler, criteria, num_envs)
        env = ManagerBasedRlEnv(cfg, device=device)
        free_joint_damping = joint_damping_metadata(env.sim.mj_model, "mug/mug_free")
        DEFAULT_ROBOT.validate_env(env)
        DEFAULT_ROBOT.validate_model(env.sim.mj_model)
        limits = torch.tensor(sampler.limits, device=device)
        env.reset(seed=seed)
        if cuda_graph_mode == "plain":
            env.sim.wp_model.opt.graph_conditional = False
            env.sim.step()
            wp.synchronize_device(env.sim.wp_device)
            env.sim.use_cuda_graph = True
            env.sim.create_graph()
            if env.sim.step_graph is None or env.sim.reset_graph is None:
                raise RuntimeError("Plain graph capture failed")
        elif cuda_graph_mode != "auto":
            raise ValueError(f"Unknown graph mode: {cuda_graph_mode}")
        initialization_seconds = perf_counter() - started

        def step(target_pos, target_quat, closed):
            current = observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)
            action = pose_target_action(
                *current,
                target_pos,
                target_quat,
                action_cfg=action_cfg,
                gripper_command=-1.0 if closed else 1.0,
            )
            _, _, terminal, timeout, _ = env.step(action)
            if bool((terminal | timeout).any()) or bool((env.sim.data.overflow != 0).any()):
                raise RuntimeError("Unexpected episode end or MuJoCo capacity overflow")

        checkpoint()
        for category, kind, insertion_stage, target_count in jobs:
            cursor = stages[f"{kind}/{insertion_stage}"]
            if cursor["complete"]:
                continue
            stage_count = cursor["retained"]
            class_started = perf_counter()
            if stage_count >= target_count:
                cursor["source_complete"] = True

            def replay(
                rows, eligible, replay_seed, category=category, insertion_stage=insertion_stage
            ):
                return reloaded_hold_mask(
                    env,
                    rows,
                    eligible,
                    SPEC,
                    limits=limits,
                    finger_upper_tolerance=criteria.finger_upper_tolerance,
                    action_cfg=action_cfg,
                    gripper_command=1.0 if category == 0 else -1.0,
                    seed=replay_seed,
                    steps=hold_steps,
                    required_steps=required,
                    acceptance=lambda replay_env, initial: accepted(
                        replay_env,
                        category,
                        initial,
                        criteria,
                        limits,
                        insertion_stage=insertion_stage,
                    ),
                    safety=lambda replay_env: (
                        _safe(replay_env, criteria)
                        & poses_within_workspace(snapshot(replay_env), sampler.workspace_corners)
                        & (
                            ~contact_found(replay_env, "mug_robot_contact")
                            if category == 0
                            else torch.ones(num_envs, dtype=torch.bool, device=device)
                        )
                    ),
                )

            if source is not None and not cursor["source_complete"]:
                source_rows = source["states"][kind]
                selected = torch.ones(len(source_rows["joint_pos"]), dtype=torch.bool)
                if category == 3 and insertion_per_stage is not None:
                    selected &= source["insertion_subtypes"]["labels"] == int(
                        insertion_stage == "threaded"
                    )
                total = int(selected.sum())
                selected &= DEFAULT_ROBOT.reset_joint_position_mask(
                    source_rows["joint_pos"], limits.cpu(), criteria.finger_upper_tolerance
                )
                selected &= poses_within_workspace(source_rows, sampler.workspace_corners)
                original_ids = selected.nonzero()[:, 0]
                report = reuse_counts.setdefault(
                    f"{kind}/{insertion_stage}",
                    {
                        "source_rows": total,
                        "range_workspace_valid": len(original_ids),
                        "hold_tested": 0,
                        "hold_passed": 0,
                        "retained": 0,
                    },
                )
                for offset in range(cursor["source_next_offset"], len(original_ids), num_envs):
                    ids = original_ids[offset : offset + num_envs]
                    rows, eligible = padded_candidates(
                        {key: value[ids] for key, value in source_rows.items()}, num_envs, device
                    )
                    good = replay(rows, eligible, seed + 900_000 + category * 10_000 + offset)
                    report["hold_tested"] += len(ids)
                    report["hold_passed"] += int(good.sum())
                    if category == 0:
                        indices = good.nonzero()[:, 0]
                        if len(indices):
                            good[indices] = torch.tensor(
                                sampler.settled_poses_reachable(
                                    rows["mug_pose"][indices].cpu().numpy(),
                                    rows["tree_pose"][indices].cpu().numpy(),
                                ),
                                device=device,
                            )
                    take = min(int(good.sum()), target_count - stage_count)
                    indices = good.nonzero()[:take, 0]
                    if take:
                        for field in FIELDS:
                            collected[kind][field].append(rows[field][indices].cpu())
                        source_indices[kind].append(ids[indices.cpu()])
                        stage_count += take
                        counts[kind] += take
                        if category == 3 and insertion_per_stage is not None:
                            insertion_labels.append(
                                torch.full(
                                    (take,), int(insertion_stage == "threaded"), dtype=torch.int64
                                )
                            )
                    report["retained"] = stage_count
                    cursor["retained"] = stage_count
                    cursor["source_next_offset"] = offset + len(ids)
                    checkpoint()
                    print(
                        f"REUSE {kind}/{insertion_stage}: {report} deficit={target_count - stage_count}",
                        flush=True,
                    )
                    if stage_count >= target_count:
                        break
                cursor["source_complete"] = True
                checkpoint()
            if category == 1:
                sampler.set_resting_poses(torch.cat(collected[RESET_TYPES[0]]["mug_pose"]).numpy())
            candidate_stream = (
                None
                if source is not None or progress_path is not None
                else sampler.iter_batches(
                    kind,
                    num_envs,
                    max_attempts=proposal_attempts_per_state * num_envs,
                    num_batches=max_batches,
                    prefetch=category != 0,
                )
            )
            for batch in range(cursor["next_batch"], max_batches):
                if stage_count >= target_count:
                    break
                batch_started = perf_counter()
                candidate_count = (
                    num_envs
                    if candidate_stream is not None
                    else min(num_envs, target_count - stage_count)
                )
                candidates = (
                    next(candidate_stream)
                    if candidate_stream is not None
                    else sampler.batch(
                        kind,
                        candidate_count,
                        max_attempts=proposal_attempts_per_state * candidate_count,
                    )
                )
                requested_count = candidate_count
                candidate_count = len(candidates["joint_pos"])
                proposal_seconds = perf_counter() - batch_started
                proposal_attempts[kind] += sampler.last_batch_attempts
                if not candidate_count:
                    timings[kind]["proposal_seconds"] += proposal_seconds
                    cursor["next_batch"] = batch + 1
                    checkpoint()
                    print(
                        f"{kind}/{insertion_stage}: batch={batch + 1} "
                        f"no static candidates in bounded budget (requested={requested_count}); "
                        f"saved={counts[kind]}/{per_type}",
                        flush=True,
                    )
                    continue
                candidates, active = padded_candidates(candidates, num_envs, device)
                physics_started = perf_counter()
                env.reset(
                    seed=seed
                    + 100_000 * category
                    + batch
                    + (50_000 if insertion_stage == "outside_tip" else 0)
                )
                write_states(env, {key: value.to(device) for key, value in candidates.items()})
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.sim.sense()
                start = snapshot(env)
                target_pos, target_quat = (
                    value.clone() for value in observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)
                )
                safe = _safe(env, criteria) & active
                safe &= poses_within_workspace(start, sampler.workspace_corners)
                stable = torch.zeros(num_envs, device=device, dtype=torch.int32)
                for _ in range(settle_steps if category == 0 else hold_steps):
                    step(target_pos, target_quat, category != 0)
                    safe &= _safe(env, criteria)
                    safe &= poses_within_workspace(snapshot(env), sampler.workspace_corners)
                    if category == 0:
                        safe &= ~contact_found(env, "mug_robot_contact")
                    eligible = accepted(env, min(category, 2), start, criteria, limits)
                    stable = torch.where(eligible, stable + 1, 0)
                good = safe & (stable >= required)
                if category == 3:
                    initial_anchor = _anchor_tree(env).clone()
                    good &= initial_anchor[:, 0] > BRANCH_OUTER[0] + 0.02
                    desired = initial_anchor.clone()
                    desired[:, 0] = torch.tensor(
                        sampler.rng.uniform(*sampler.cfg.final_anchor_x, num_envs), device=device
                    )
                    desired[:, 1] = 0.0
                    desired[:, 2] = BRANCH_OUTER[2]
                    if insertion_stage == "outside_tip":
                        desired[:, 0] = BRANCH_OUTER[0] + torch.empty(
                            num_envs, device=device
                        ).uniform_(*criteria.outside_tip_clearance)
                        desired[:, 1:] += torch.empty_like(desired[:, 1:]).uniform_(
                            -criteria.outside_tip_transverse_jitter,
                            criteria.outside_tip_transverse_jitter,
                        )
                    tree_quat = pose(env, "tree")[1]
                    displacement = quat_apply(tree_quat, desired - initial_anchor)
                    initial_ee_pos = observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)[
                        0
                    ].clone()
                    for index in range(thread_steps):
                        t = (index + 1) / thread_steps
                        alpha = 10 * t**3 - 15 * t**4 + 6 * t**5
                        step(initial_ee_pos + alpha * displacement, target_quat, True)
                        good &= _safe(env, criteria)
                        good &= poses_within_workspace(snapshot(env), sampler.workspace_corners)
                        if insertion_stage == "outside_tip":
                            good &= ~contact_found(env, "mug_tree_contact")
                            good &= _anchor_tree(env)[:, 0] > BRANCH_OUTER[0] + 0.005
                        good &= contact_found(env, "left_finger_contact") & contact_found(
                            env, "right_finger_contact"
                        )
                    stable.zero_()
                    for _ in range(hold_steps):
                        step(initial_ee_pos + displacement, target_quat, True)
                        good &= _safe(env, criteria)
                        good &= poses_within_workspace(snapshot(env), sampler.workspace_corners)
                        eligible = accepted(
                            env, category, start, criteria, limits, insertion_stage=insertion_stage
                        )
                        stable = torch.where(eligible, stable + 1, 0)
                    good &= stable >= required
                    if insertion_stage == "threaded":
                        good &= (
                            initial_anchor[:, 0] - _anchor_tree(env)[:, 0]
                            >= criteria.minimum_thread_displacement
                        )
                final = snapshot(env)
                wp.synchronize_device(env.sim.wp_device)
                physics_seconds = perf_counter() - physics_started
                postsettle_started = perf_counter()
                if category == 0:
                    indices = good.nonzero()[:, 0]
                    if len(indices):
                        good[indices] = torch.tensor(
                            sampler.settled_poses_reachable(
                                final["mug_pose"][indices].cpu().numpy(),
                                final["tree_pose"][indices].cpu().numpy(),
                            ),
                            device=device,
                        )
                wp.synchronize_device(env.sim.wp_device)
                postsettle_seconds = perf_counter() - postsettle_started
                for label, elapsed in (
                    ("proposal_seconds", proposal_seconds),
                    ("physics_seconds", physics_seconds),
                    ("postsettle_seconds", postsettle_seconds),
                ):
                    timings[kind][label] += elapsed
                candidates_count[kind] += candidate_count
                # Export the input of a normal reset replay, including bounded
                # finger projection, for both fresh and source-deficit runs.
                good = replay(final, good, seed + 800_000 + category * 10_000 + batch)
                take = min(int(good.sum()), target_count - stage_count)
                indices = good.nonzero()[:take, 0]
                if take:
                    for field in FIELDS:
                        collected[kind][field].append(final[field][indices].cpu())
                    counts[kind] += take
                    stage_count += take
                    source_indices[kind].append(torch.full((take,), -1, dtype=torch.int64))
                    if category == 3 and insertion_per_stage is not None:
                        insertion_labels.append(
                            torch.full(
                                (take,), int(insertion_stage == "threaded"), dtype=torch.int64
                            )
                        )
                cursor["retained"] = stage_count
                cursor["next_batch"] = batch + 1
                checkpoint()
                print(
                    f"{kind}/{insertion_stage}: batch={batch + 1} eligible={int(good.sum())}/{candidate_count} stage={stage_count}/{target_count} "
                    f"saved={counts[kind]}/{per_type} proposal={proposal_seconds:.2f}s "
                    f"physics={physics_seconds:.2f}s postsettle={postsettle_seconds:.2f}s",
                    flush=True,
                )
            if candidate_stream is not None:
                candidate_stream.close()
            candidate_stream = None
            if stage_count < target_count:
                raise RuntimeError(
                    f"Incomplete {kind}/{insertion_stage}: {stage_count}/{target_count}; "
                    f"no bank written; progress={progress_path}"
                )
            timings[kind]["total_seconds"] = (
                timings[kind].get("total_seconds", 0.0) + perf_counter() - class_started
            )
            cursor["complete"] = True
            checkpoint()
    finally:
        if candidate_stream is not None:
            candidate_stream.close()
        sampler.close()
        if env is not None:
            env.close()
    bank = {
        "schema": SCHEMA,
        "contract": bank_contract(),
        "split": split,
        "provenance": {
            "method": "source_hold_revalidation_and_osc_deficit_collection"
            if source is not None
            else "fresh_rim_ik_and_osc_threading",
            "workspace": workspace_metadata(),
            "workspace_geometry": "conservative_local_envelopes_of_all_visual_collision_geometry",
            "seed": seed,
            "proposal_config": {
                **sampler.metadata,
                "proposal_prefetch": "disabled_for_source_reuse_or_progress"
                if source is not None or progress_path is not None
                else sampler.metadata.get("proposal_prefetch"),
            },
            "proposal_attempts_per_state": proposal_attempts_per_state,
            "acceptance": asdict(criteria),
            "num_envs": num_envs,
            "physics_candidates": candidates_count,
            "proposal_attempts": proposal_attempts,
            "proposal_attempts_scope": "consumed_batches_only_excludes_unused_prefetch",
            "wall_time": {
                "initialization_seconds": initialization_seconds,
                "categories": timings,
                "total_seconds": previous_elapsed + perf_counter() - started,
                "proposal_includes_worker_startup": True,
                "proposal_seconds_scope": "consumer_wait_not_total_parallel_CPU_work",
                "resume_time_scope": "saved_checkpoint_segments_excludes_unsaved_interrupted_work",
                "category_total_scope": "completed_stage_timers_excludes_interrupted_stage_prefix",
            },
            "physics_dt": cfg.sim.mujoco.timestep,
            "free_joint_damping": free_joint_damping,
            "control_dt": dt,
            "decimation": cfg.decimation,
            "hold_steps": hold_steps,
            "settle_steps": settle_steps,
            "thread_steps": thread_steps,
            "accepted_consecutive_steps": required,
            "hold_command": "fixed_initial_ee_pose_via_relative_osc_action",
            "task3_path": "outside_tip_to_handle_engaged_by_osc_action_and_free_mug_dynamics",
            "controller": asdict(action_cfg),
            "physics_randomization": False,
            "cuda_graph_mode": cuda_graph_mode,
            "nconmax": cfg.sim.nconmax,
            "njmax": cfg.sim.njmax,
            "contact_sensor_maxmatch": cfg.sim.contact_sensor_maxmatch,
            "candidate_replay": {
                "all_exported_rows": True,
                "steps": hold_steps,
                "required_consecutive_steps": required,
                "saved_state": "replay_input_with_projected_fingers",
            },
            "collection_progress": {
                "enabled": progress_path is not None,
                "resumed": resumed_from_progress,
                "boundary": "accepted_source_or_fresh_batch_no_inflight_proposals"
                if progress_path is not None
                else None,
                "physical_continuation": "normal_reset_with_deterministic_batch_seed_not_inflight_simulation",
            },
        },
        "states": {
            kind: {field: torch.cat(parts) for field, parts in rows.items()}
            for kind, rows in collected.items()
        },
    }
    if insertion_per_stage is not None:
        bank["insertion_subtypes"] = {
            "names": ["outside_tip", "threaded"],
            "labels": torch.cat(insertion_labels),
            "counts": [insertion_per_stage, insertion_per_stage],
        }
        bank["provenance"]["insertion_subtypes"] = {
            "names": ["outside_tip", "threaded"],
            "counts": [insertion_per_stage, insertion_per_stage],
            "label_field": "insertion_subtypes.labels",
            "legacy_category_key": RESET_TYPES[3],
            "outside_tip_motion": "minimum_jerk_fixed_EE_displacement_via_OSC",
            "threaded_motion": "minimum_jerk_fixed_EE_displacement_via_OSC",
        }
    if source is not None:
        bank["source_indices"] = {kind: torch.cat(parts) for kind, parts in source_indices.items()}
        bank["provenance"]["source_revalidation"] = {
            "path": str(source_bank),
            "original_provenance": source["provenance"],
            "counts": reuse_counts,
            "source_index_field": "source_indices",
            "fresh_row_index": -1,
            "saved_state": "replay_input_with_projected_fingers",
            "hold_steps": hold_steps,
            "required_consecutive_steps": required,
            "historical_threading_reexecuted": False,
            "all_retained_and_fresh_rows_hold_replayed": True,
            "deficit_batch_policy": "min_capacity_remaining_without_prefetch",
        }
        bank["provenance"]["task3_path"] = (
            "historical_source_witness_or_fresh_osc_path_see_source_indices"
        )
        if insertion_per_stage is not None:
            for stage in ("outside_tip", "threaded"):
                bank["provenance"]["insertion_subtypes"][f"{stage}_motion"] = (
                    "historical_source_witness_or_fresh_osc_see_source_indices"
                )
    DEFAULT_ROBOT.project_reset_fingers(
        bank, torch.tensor(sampler.limits, dtype=torch.float32), criteria.finger_upper_tolerance
    )
    bank["provenance"]["finger_position_projection"]["projected_contact_hold_revalidated"] = True
    validate_bank(bank, model=sampler.model)
    atomic_save(bank, output)
    print(f"Saved {output}: {counts}", flush=True)


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Collect a hold-validated MugHang reset bank; --source-bank revalidates a "
            "compatible bank and --progress-path makes the run resumable."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (
        ("per-type", 10000),
        ("num-envs", 1024),
        ("max-batches", 100),
        ("hold-steps", 20),
        ("settle-steps", 200),
        ("thread-steps", 60),
        ("seed", 42),
        ("proposal-workers", DEFAULT_PROPOSAL_WORKERS),
        ("proposal-attempts-per-state", 20000),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    parser.add_argument(
        "--split", choices=("development", "train", "validation", "test"), default="train"
    )
    parser.add_argument("--nconmax", type=int)
    parser.add_argument("--njmax", type=int)
    parser.add_argument("--insertion-per-stage", type=int)
    parser.add_argument("--source-bank", type=Path)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    collect(**vars(parser.parse_args(argv)))
