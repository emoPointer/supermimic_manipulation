"""Collect the four Square reset categories (``clab resets collect --task square``).

Collection disables training randomization, and every grasp needs force-bearing contacts on
the exposed side handle. The last category starts above the peg and descends under real
actions (``--descent-steps``, six seconds by default) with bounded lateral corrections from
the measured nut position; no inserted pose is written into physics.
``--insertion-per-stage N`` collects N above-peg and N threaded states (``--per-type 2*N``)
and saves their labels in ``insertion_subtypes``. Above-peg states descend towards a
15-40 mm nominal tip gap with +/-12 mm XY jitter and must hold without peg contact.

Fresh candidates pass a 1 s reset/reload hold without the descent feedback, meeting the
per-step predicates for the whole second. Tabletop grasps must also lift the nut 60 mm and
hold it unsupported for 1 s in four replays whose world slots shift by
``floor(k*num_envs/4)``, k=0..3; all four must pass. The exported row is the pre-lift
tabletop state with the approved finger projection. ``--source-bank`` reuses a compatible
bank's first three categories without rechecking them. A failed run writes no bank.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv

from clab.controllers.pose import pose_target_action
from clab.logging.model import joint_damping_metadata
from clab.observations.state import observation_pose
from clab.robots.parallel_kinematics import DEFAULT_PROPOSAL_WORKERS
from clab.scenes.workspace import workspace_metadata
from clab.tasks.rigid_resets import reloaded_hold_mask

from .config import make_env_cfg
from .mdp import contact_found, handle_grasp_status, insertion_diagnostic
from .proposals import ProposalConfig, Sampler
from .props import NUT_HALF_HEIGHT, PEG_HALF_HEIGHT, within_workspace
from .reset_data import FIELDS, SCHEMA, SPEC, bank_contract, validate_bank
from .resets import snapshot, write_states
from .settings import DESCENT_STEPS, RESET_TYPES, AssemblyCriteria
from .variants import DEFAULT_ROBOT

TABLETOP_REPLAY_PASSES = 4


@dataclass(frozen=True)
class AcceptanceConfig:
    """Reset quality thresholds, distinct from released assembly success."""

    stable_seconds: float = 0.5
    linear_speed_max: float = 0.03
    angular_speed_max: float = 0.3
    resting_linear_speed_max: float = 0.003
    resting_angular_speed_max: float = 0.05
    joint_speed_max: float = 2.0
    finger_upper_tolerance: float = DEFAULT_ROBOT.reset_finger_upper_tolerance
    held_drift_max: float = 0.02
    insertion_height: tuple[float, float] = (0.052, 0.078)
    insertion_xy_max: float = 0.006
    above_peg_clearance: tuple[float, float] = (0.015, 0.040)
    above_peg_xy_max: float = 0.02
    above_peg_xy_jitter: float = 0.012


@dataclass(frozen=True)
class DescentConfig:
    """Collection-only lateral feedback, not acceptance or robot speed limits."""

    lateral_gain: float = 0.2
    lateral_correction_step: float = 0.001


@dataclass(frozen=True)
class LiftConfig:
    """Additional tabletop-grasp export witness, not a policy success criterion."""

    height: float = 0.06
    minimum_rise: float = 0.04
    move_steps: int = 60
    hold_steps: int = 20


def validate_source_grasp_witness(provenance, *, per_type, proposal, lift):
    """Reused tabletop rows must already carry this collection's stronger proof."""
    recorded = provenance.get("tabletop_lift_validation", {})
    contexts = provenance.get("tabletop_replay_contexts", {})
    context_counts = contexts.get("counts", [])
    if (
        tuple(provenance.get("proposal", {}).get("grasp_point_nut_local", ()))
        != proposal.grasp_point_nut_local
        or any(recorded.get(name) != value for name, value in asdict(lift).items())
        or recorded.get("category") != RESET_TYPES[1]
        or recorded.get("counts", {}).get("passed", 0) < per_type
        or recorded.get("saved_state") != "original_projected_tabletop_input_not_lifted_endpoint"
        or contexts.get("passes") != TABLETOP_REPLAY_PASSES
        or contexts.get("combination") != "all_fixed_passes"
        or contexts.get("saved_state") != "same_original_projected_tabletop_input"
        or contexts.get("hold_steps") != 20
        or contexts.get("required_stable_steps") != 20
        or contexts.get("retained", 0) < per_type
        or type(provenance.get("num_envs")) is not int
        or provenance["num_envs"] < 1
        or contexts.get("world_slot_shifts")
        != [
            index * provenance.get("num_envs", 0) // TABLETOP_REPLAY_PASSES
            for index in range(TABLETOP_REPLAY_PASSES)
        ]
        or len(context_counts) != TABLETOP_REPLAY_PASSES
        or any(entry.get("lift_passed", 0) < per_type for entry in context_counts)
    ):
        raise ValueError(
            "Source Square bank lacks matching deeper-grasp/tabletop-lift evidence; "
            "omit source_bank and recollect all four categories"
        )


def tabletop_context_mask(rows, eligible, *, replay, lift, counts):
    """Three fixed additional contexts after pass zero, always from original bytes.

    Callbacks return masks in the shifted simulation-slot order. Rotate them back
    before ANDing; rejected rows and padding can never recover. No trajectory end
    states feed a later pass, and each original input is exported at most once.
    """
    good = eligible.clone()
    for index in range(1, TABLETOP_REPLAY_PASSES):
        if not bool(good.any()):
            break
        shift = index * len(good) // TABLETOP_REPLAY_PASSES
        shifted_rows = {name: value.roll(shift, 0) for name, value in rows.items()}
        shifted_mask = good.roll(shift, 0)
        counts[index]["replay_tested"] += int(shifted_mask.sum())
        replayed = replay(shifted_rows, shifted_mask.clone()) & shifted_mask
        counts[index]["replay_passed"] += int(replayed.sum())
        lifted = lift(replayed.clone()) & replayed
        counts[index]["lift_passed"] += int(lifted.sum())
        good &= lifted.roll(-shift, 0)
    return good


def tabletop_lift_mask(
    env, eligible, *, limits, criteria, max_penetration, corners, action_cfg, config
):
    """Continue the normal saved-state replay with a real lift; never export its end.

    Call only after the one-second projected-input reload hold. Its eligible
    mask retains original row indices; padded/rejected worlds never enter the
    result. The target is an end-effector translation, but rise, contacts and
    final stability are measured on the physical nut. No state is written.
    """
    if (
        not 0 < config.minimum_rise <= config.height
        or min(config.move_steps, config.hold_steps) < 1
    ):
        raise ValueError("Square lift needs positive duration and 0 < minimum rise <= height")
    if not bool(eligible.any()):
        return eligible.clone()
    start = snapshot(env)
    target_pos, target_quat = (
        value.clone() for value in observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)
    )
    expected = {name: value.clone() for name, value in start.items()}
    expected["nut_pose"][:, 2] += config.height
    good = eligible.clone()
    for index in range(config.move_steps + config.hold_steps):
        position = target_pos.clone()
        position[:, 2] += config.height * min(1.0, (index + 1) / config.move_steps)
        action = pose_target_action(
            *observation_pose(env, "robot", DEFAULT_ROBOT.ee_body),
            position,
            target_quat,
            action_cfg=action_cfg,
            gripper_command=-1.0,
            **(
                {"base_pos": env.scene["robot"].data.root_link_pos_w}
                if getattr(action_cfg, "input_mode", "relative") == "absolute"
                else {}
            ),
        )
        _, _, terminal, timeout, _ = env.step(action)
        if bool((terminal | timeout).any()) or bool((env.sim.data.overflow != 0).any()):
            raise RuntimeError("Square lift witness encountered episode end or capacity overflow")
        rows = snapshot(env)
        if not all(bool(torch.isfinite(value).all()) for value in rows.values()):
            raise RuntimeError("Square lift witness encountered non-finite physical state")
        handle, wrong = handle_grasp_status(env, AssemblyCriteria())
        good &= safe_contacts(env, max_penetration) & within_workspace(rows, corners)
        good &= handle & ~wrong & ~contact_found(env, "nut_peg_contact")
        good &= DEFAULT_ROBOT.reset_joint_position_mask(
            rows["joint_pos"], limits, criteria.finger_upper_tolerance
        )
        good &= rows["joint_vel"].abs().amax(-1) < criteria.joint_speed_max
        if index >= config.move_steps:
            good &= accepted(env, 2, expected, limits, criteria, max_penetration)
            good &= rows["nut_pose"][:, 2] - start["nut_pose"][:, 2] >= config.minimum_rise
        if not bool(good.any()):
            break
    return good


def update_lateral_correction(correction, actual_nut, desired_nut, config):
    """Integrate XY tracking error into an EE target; never change physical state.

    Bound each target adjustment by its Euclidean norm. Z/orientation remain
    owned by the original descent path. All input tensors remain unchanged.
    """
    adjustment = config.lateral_gain * (desired_nut[:, :2] - actual_nut[:, :2])
    length = adjustment.norm(dim=-1, keepdim=True)
    adjustment *= (config.lateral_correction_step / length.clamp_min(1e-12)).clamp(max=1)
    return correction + adjustment


def safe_contacts(env, max_penetration, *, include_nut_table=True):
    """Contact safety; only task0's natural drop excludes transient table impact.

    Final acceptance always includes the table. No robot/peg contact exemption
    is made, and no pose/velocity is changed to hide impact or penetration.
    """
    safe = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    names = ("robot_contact", "nut_peg_contact")
    if include_nut_table:
        names += ("nut_table_contact",)
    for name in names:
        sensor = env.scene[name].data
        safe &= ~((sensor.found > 0) & (sensor.dist < -max_penetration)).any(-1)
    return safe & ~contact_found(env, "nut_arm_contact") & ~contact_found(env, "robot_peg_contact")


def accepted(
    env, category, start, limits, criteria, max_penetration, *, insertion_stage="threaded"
):
    """Check current physical state; no hidden pose fix or successful-row override."""
    rows = snapshot(env)
    valid = safe_contacts(env, max_penetration)
    for value in rows.values():
        valid &= torch.isfinite(value).all(-1)
    valid &= DEFAULT_ROBOT.reset_joint_position_mask(
        rows["joint_pos"], limits, criteria.finger_upper_tolerance
    )
    valid &= rows["joint_vel"].abs().amax(-1) < criteria.joint_speed_max
    speed = rows["nut_vel"][:, :3].norm(dim=-1)
    spin = rows["nut_vel"][:, 3:].norm(dim=-1)
    valid &= (speed < criteria.linear_speed_max) & (spin < criteria.angular_speed_max)
    table = contact_found(env, "nut_table_contact")
    peg = contact_found(env, "nut_peg_contact")
    if category == 0:
        return (
            valid
            & table
            & ~peg
            & ~contact_found(env, "nut_robot_contact")
            & (speed < criteria.resting_linear_speed_max)
            & (spin < criteria.resting_angular_speed_max)
        )
    # Force-bearing contact positions must lie on the exposed side handle.
    fingers, wrong = handle_grasp_status(env, AssemblyCriteria())
    fingers &= ~wrong
    valid &= fingers
    if category in (1, 2):
        valid &= (rows["nut_pose"][:, :3] - start["nut_pose"][:, :3]).norm(
            dim=-1
        ) < criteria.held_drift_max
    if category == 1:
        return valid & table & ~peg
    if category == 2:
        return valid & ~table & ~peg & (rows["nut_pose"][:, 2] > 0.05)
    relative = rows["nut_pose"][:, :3] - rows["peg_pose"][:, :3]
    if insertion_stage == "above_peg":
        from mjlab.utils.lab_api.math import quat_apply_inverse

        local = quat_apply_inverse(rows["peg_pose"][:, 3:], relative)
        return (
            valid
            & ~table
            & ~peg
            & ~insertion_diagnostic(env)
            & (local[:, :2].norm(dim=-1) < criteria.above_peg_xy_max)
            & (local[:, 2] > PEG_HALF_HEIGHT + NUT_HALF_HEIGHT)
            & (
                local[:, 2]
                < PEG_HALF_HEIGHT + NUT_HALF_HEIGHT + criteria.above_peg_clearance[1] + 0.01
            )
        )
    if insertion_stage != "threaded":
        raise ValueError(f"Unknown insertion stage: {insertion_stage}")
    return (
        valid
        & ~table
        & insertion_diagnostic(env)
        & (relative[:, :2].norm(dim=-1) < criteria.insertion_xy_max)
    )


def collect(
    output: Path,
    *,
    per_type=100,
    num_envs=64,
    max_batches=50,
    hold_steps=20,
    settle_steps=100,
    descent_steps=DESCENT_STEPS,
    seed=42,
    device="cuda:0",
    cuda_graph_mode="auto",
    proposal_workers=DEFAULT_PROPOSAL_WORKERS,
    nconmax=None,
    njmax=None,
    split="development",
    source_bank: Path | None = None,
    insertion_per_stage: int | None = None,
):
    """Write only a complete bank; all final rows come from simulated states."""
    if output.exists():
        raise FileExistsError(output)
    if (
        min(
            per_type,
            num_envs,
            max_batches,
            hold_steps,
            settle_steps,
            descent_steps,
            proposal_workers,
        )
        < 1
    ):
        raise ValueError("Collection budgets must be positive")
    if split not in ("development", "train", "validation", "test"):
        raise ValueError(f"Unknown reset split: {split}")
    criteria = AcceptanceConfig()
    descent = DescentConfig()
    lift = LiftConfig()
    replay_counts = {kind: {"tested": 0, "passed": 0} for kind in RESET_TYPES}
    lift_counts = {"tested": 0, "passed": 0}
    context_counts = [
        dict(replay_tested=0, replay_passed=0, lift_passed=0) for _ in range(TABLETOP_REPLAY_PASSES)
    ]
    if insertion_per_stage is not None and (
        insertion_per_stage < 1 or per_type != 2 * insertion_per_stage
    ):
        raise ValueError("Mixed insertion requires per_type == 2 * insertion_per_stage > 0")
    if source_bank is not None and insertion_per_stage is None:
        raise ValueError("Reusing first three classes requires explicit insertion_per_stage")
    source = None
    if source_bank is not None:
        source = torch.load(source_bank, map_location="cpu", weights_only=True)
        source_states = validate_bank(source)
        if source["provenance"].get("workspace") != workspace_metadata():
            raise ValueError("Source bank must use the current complete-geometry workspace")
        if source["split"] != split:
            raise ValueError("Source split must match the requested output split")
        if any(len(source_states[k]["joint_pos"]) != per_type for k in RESET_TYPES[:3]):
            raise ValueError("Source first three categories must each match per_type")
        validate_source_grasp_witness(
            source["provenance"], per_type=per_type, proposal=ProposalConfig(), lift=lift
        )
    cfg = make_env_cfg()
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    required = math.ceil(criteria.stable_seconds / control_dt - 1e-9)
    if min(hold_steps, settle_steps) < required:
        raise ValueError("Reset acceptance requires at least 0.5 s of continuous hold")
    cfg.scene.num_envs = num_envs
    cfg.events = {
        k: v for k, v in cfg.events.items() if v.mode != "startup" and k != "reset_from_bank"
    }
    cfg.rewards = {}
    cfg.terminations.pop("success")
    cfg.terminations.pop("invalid_grasp")
    cfg.auto_reset = False
    cfg.episode_length_s = max(
        cfg.episode_length_s, (settle_steps + 2 * hold_steps + descent_steps + 2) * control_dt
    )
    for name, value in (("nconmax", nconmax), ("njmax", njmax)):
        if value is not None:
            if value < 1:
                raise ValueError(f"{name} must be positive")
            setattr(cfg.sim, name, value)
    sampler = Sampler(seed, workers=proposal_workers)
    action_cfg = cfg.actions[DEFAULT_ROBOT.action_name]
    if source is not None:
        validate_bank(source, sampler.model)
        provenance = source["provenance"]
        if (
            provenance.get("controller") != asdict(action_cfg)
            or provenance["physics_dt"] != cfg.sim.mujoco.timestep
        ):
            raise ValueError("Source bank controller/physics differs from current collection")
        if provenance["control_dt"] != control_dt or any(
            not bool(within_workspace(source_states[kind], sampler.workspace_corners).all())
            for kind in RESET_TYPES[:3]
        ):
            raise ValueError("Source bank timing or complete geometry workspace mismatch")
    collected = {kind: {field: [] for field in FIELDS} for kind in RESET_TYPES}
    counts, attempts = dict.fromkeys(RESET_TYPES, 0), dict.fromkeys(RESET_TYPES, 0)
    if source is not None:
        for kind in RESET_TYPES[:3]:
            collected[kind] = {field: [source_states[kind][field]] for field in FIELDS}
            counts[kind] = per_type
    insertion_labels = []
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        free_joint_damping = joint_damping_metadata(env.sim.mj_model, "nut/nut_free")
        DEFAULT_ROBOT.validate_model(env.sim.mj_model)
        DEFAULT_ROBOT.validate_env(env)
        limits = torch.tensor(sampler.limits, device=device, dtype=torch.float32)
        env.reset(seed=seed)
        if cuda_graph_mode == "plain":
            env.sim.wp_model.opt.graph_conditional = False
            env.sim.step()
            wp.synchronize_device(env.sim.wp_device)
            env.sim.use_cuda_graph = True
            env.sim.create_graph()
            if env.sim.step_graph is None or env.sim.reset_graph is None:
                raise RuntimeError("Plain CUDA graph capture failed")
        elif cuda_graph_mode != "auto":
            raise ValueError(f"Unsupported graph mode: {cuda_graph_mode}")

        def step(target_pos, target_quat, category):
            action = pose_target_action(
                *observation_pose(env, "robot", DEFAULT_ROBOT.ee_body),
                target_pos,
                target_quat,
                action_cfg=action_cfg,
                gripper_command=1.0 if category == 0 else -1.0,
            )
            _, _, terminal, timeout, _ = env.step(action)
            if bool((terminal | timeout).any()):
                raise RuntimeError("Unexpected episode termination during Square collection")
            if bool((env.sim.data.overflow != 0).any()):
                raise RuntimeError("MuJoCo-Warp contact/constraint capacity overflow")

        jobs = [(i, kind, "threaded", per_type) for i, kind in enumerate(RESET_TYPES)]
        if insertion_per_stage is not None:
            jobs[-1:] = [
                (3, RESET_TYPES[3], "above_peg", insertion_per_stage),
                (3, RESET_TYPES[3], "threaded", insertion_per_stage),
            ]
        for category, kind, insertion_stage, target_count in jobs:
            if source is not None and category < 3:
                continue
            stage_count = 0
            if category == 1:
                sampler.set_resting_poses(torch.cat(collected[RESET_TYPES[0]]["nut_pose"]).numpy())
            for batch in range(max_batches):
                if stage_count >= target_count:
                    break
                candidates = sampler.batch(kind, num_envs, max_attempts=500 * num_envs)
                env.reset(
                    seed=seed
                    + category * 100_000
                    + batch
                    + (50_000 if insertion_stage == "above_peg" else 0)
                )
                write_states(env, {k: v.to(device) for k, v in candidates.items()})
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.sim.sense()
                start = snapshot(env)
                target_pos, target_quat = (
                    v.clone() for v in observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)
                )
                safe = safe_contacts(env, sampler.cfg.max_penetration)
                safe &= within_workspace(start, sampler.workspace_corners)
                stable = torch.zeros(num_envs, dtype=torch.int32, device=device)
                for _ in range(settle_steps if category == 0 else hold_steps):
                    step(target_pos, target_quat, category)
                    safe &= safe_contacts(
                        env, sampler.cfg.max_penetration, include_nut_table=category != 0
                    )
                    safe &= within_workspace(snapshot(env), sampler.workspace_corners)
                    if category == 0:
                        safe &= ~contact_found(env, "nut_robot_contact")
                        safe &= ~contact_found(env, "nut_peg_contact")
                    eligible = accepted(
                        env,
                        2 if category == 3 else category,
                        start,
                        limits,
                        criteria,
                        sampler.cfg.max_penetration,
                    )
                    stable = torch.where(eligible, stable + 1, 0)
                good = safe & (stable >= required)
                if category == 3:
                    # Preserve the actual held nut-to-link transform. Nut/body
                    # state is NOT written again during this physical approach.
                    reached = snapshot(env)
                    target_pos, target_quat = (
                        v.clone() for v in observation_pose(env, "robot", DEFAULT_ROBOT.ee_body)
                    )
                    start_target = target_pos.clone()
                    heights = torch.empty(num_envs, device=device).uniform_(
                        *(
                            criteria.above_peg_clearance
                            if insertion_stage == "above_peg"
                            else criteria.insertion_height
                        )
                    )
                    delta = torch.zeros_like(target_pos)
                    delta[:, :2] = reached["peg_pose"][:, :2] - reached["nut_pose"][:, :2]
                    if insertion_stage == "above_peg":
                        delta[:, :2] += torch.empty_like(delta[:, :2]).uniform_(
                            -criteria.above_peg_xy_jitter, criteria.above_peg_xy_jitter
                        )
                    delta[:, 2] = heights - reached["nut_pose"][:, 2]
                    if insertion_stage == "above_peg":
                        delta[:, 2] += reached["peg_pose"][:, 2] + PEG_HALF_HEIGHT + NUT_HALF_HEIGHT
                    correction = torch.zeros_like(target_pos[:, :2])
                    for index in range(descent_steps):
                        correction = update_lateral_correction(
                            correction,
                            snapshot(env)["nut_pose"][:, :3],
                            reached["nut_pose"][:, :3] + delta * (index / descent_steps),
                            descent,
                        )
                        path_target = start_target + delta * ((index + 1) / descent_steps)
                        path_target[:, :2] += correction
                        step(path_target, target_quat, category)
                        good &= safe_contacts(env, sampler.cfg.max_penetration)
                        good &= within_workspace(snapshot(env), sampler.workspace_corners)
                        handle, wrong = handle_grasp_status(env, AssemblyCriteria())
                        good &= handle & ~wrong
                        if insertion_stage == "above_peg":
                            good &= ~insertion_diagnostic(env) & ~contact_found(
                                env, "nut_peg_contact"
                            )
                    stable.zero_()
                    for _ in range(hold_steps):
                        correction = update_lateral_correction(
                            correction,
                            snapshot(env)["nut_pose"][:, :3],
                            reached["nut_pose"][:, :3] + delta,
                            descent,
                        )
                        path_target = start_target + delta
                        path_target[:, :2] += correction
                        step(path_target, target_quat, category)
                        good &= safe_contacts(env, sampler.cfg.max_penetration)
                        good &= within_workspace(snapshot(env), sampler.workspace_corners)
                        eligible = accepted(
                            env,
                            category,
                            reached,
                            limits,
                            criteria,
                            sampler.cfg.max_penetration,
                            insertion_stage=insertion_stage,
                        )
                        stable = torch.where(eligible, stable + 1, 0)
                    good &= stable >= required
                final = snapshot(env)
                if category == 0:
                    indices = good.nonzero()[:, 0]
                    reachable = sampler.settled_poses_reachable(
                        final["nut_pose"][indices].cpu().numpy(),
                        final["peg_pose"][indices].cpu().numpy(),
                    )
                    good[indices] = torch.as_tensor(reachable, device=device)
                replay_counts[kind]["tested"] += int(good.sum())

                def replay(
                    rows,
                    eligible,
                    *,
                    category=category,
                    insertion_stage=insertion_stage,
                    batch=batch,
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
                        seed=seed + 700_000 + category * 100_000 + batch,
                        steps=2 * required,
                        required_steps=2 * required,
                        acceptance=lambda replay_env, replay_start: accepted(
                            replay_env,
                            category,
                            replay_start,
                            limits,
                            criteria,
                            sampler.cfg.max_penetration,
                            insertion_stage=insertion_stage,
                        ),
                        safety=lambda replay_env: (
                            safe_contacts(replay_env, sampler.cfg.max_penetration)
                            & within_workspace(snapshot(replay_env), sampler.workspace_corners)
                        ),
                    )

                if category == 1:
                    context_counts[0]["replay_tested"] += int(good.sum())
                good = replay(final, good)
                replay_counts[kind]["passed"] += int(good.sum())
                if category == 1:
                    context_counts[0]["replay_passed"] += int(good.sum())
                    lift_counts["tested"] += int(good.sum())

                    def lift_replay(eligible):
                        return tabletop_lift_mask(
                            env,
                            eligible,
                            limits=limits,
                            criteria=criteria,
                            max_penetration=sampler.cfg.max_penetration,
                            corners=sampler.workspace_corners,
                            action_cfg=action_cfg,
                            config=lift,
                        )

                    good = lift_replay(good)
                    context_counts[0]["lift_passed"] += int(good.sum())
                    good = tabletop_context_mask(
                        final, good, replay=replay, lift=lift_replay, counts=context_counts
                    )
                    lift_counts["passed"] += int(good.sum())
                eligible_count = int(good.sum())
                take = min(eligible_count, target_count - stage_count)
                indices = good.nonzero()[:take, 0]
                if take:
                    for field in FIELDS:
                        collected[kind][field].append(final[field][indices].cpu())
                    if category == 3 and insertion_per_stage is not None:
                        insertion_labels.append(
                            torch.full(
                                (take,), int(insertion_stage == "threaded"), dtype=torch.int64
                            )
                        )
                counts[kind] += take
                stage_count += take
                attempts[kind] += num_envs
                print(
                    f"{kind}/{insertion_stage}: batch={batch + 1} eligible={eligible_count}/{num_envs} stage={stage_count}/{target_count} total={counts[kind]}/{per_type}",
                    flush=True,
                )
            if stage_count < target_count:
                raise RuntimeError(
                    f"Incomplete {kind}/{insertion_stage}: {stage_count}/{target_count}; no bank written"
                )
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
            "method": "fresh_handle_IK_mjlab_hold_and_physical_peg_descent",
            "workspace": workspace_metadata(),
            "workspace_geometry": "all_visual_collision_box_corners",
            "seed": seed,
            "proposal": sampler.metadata,
            "acceptance": asdict(criteria),
            "per_type": per_type,
            "num_envs": num_envs,
            "hold_steps": hold_steps,
            "settle_steps": settle_steps,
            "descent_steps": descent_steps,
            "descent_feedback": asdict(descent),
            "descent_feedback_source": "physical_nut_xy_collection_only",
            "replay_validation": {
                "steps": 2 * required,
                "required_stable_steps": 2 * required,
                "counts": replay_counts,
                "saved_state": "before_replay_with_projected_fingers",
                "command": "fixed_initial_ee_pose_without_descent_feedback",
                "seed_offset": 700_000,
            },
            "tabletop_lift_validation": {
                **asdict(lift),
                "counts": lift_counts,
                "category": RESET_TYPES[1],
                "start": "after_projected_input_reload_hold",
                "command": "fixed_orientation_world_vertical_translation",
                "physical_requirements": "unbroken_legal_handle_contact_safe_path_then_unsupported_stable_hold",
                "saved_state": "original_projected_tabletop_input_not_lifted_endpoint",
                "reused_categories_validated": False,
            },
            "tabletop_replay_contexts": {
                "passes": TABLETOP_REPLAY_PASSES,
                "world_slot_shifts": [
                    index * num_envs // TABLETOP_REPLAY_PASSES
                    for index in range(TABLETOP_REPLAY_PASSES)
                ],
                "slot_mapping": "cyclic_shift_floor_pass_times_num_envs_over_four",
                "seed_policy": "same_batch_replay_seed_for_all_passes",
                "combination": "all_fixed_passes",
                "hold_steps": 2 * required,
                "required_stable_steps": 2 * required,
                "counts": context_counts,
                "retained": counts[RESET_TYPES[1]] if source is None else 0,
                "saved_state": "same_original_projected_tabletop_input",
                "coverage": "four_fixed_contexts_only_small_batches_may_repeat_world_slots",
                "reused_categories_validated": False,
            },
            "control_dt": control_dt,
            "physics_dt": cfg.sim.mujoco.timestep,
            "decimation": cfg.decimation,
            "controller": asdict(action_cfg),
            "cuda_graph_mode": cuda_graph_mode,
            "nconmax": cfg.sim.nconmax,
            "njmax": cfg.sim.njmax,
            "attempts": attempts,
            "domain_randomization": False,
            "free_joint_damping": free_joint_damping,
            "task0_table_impact": "source_0.051m_drop_then_strict_continuous_steady_window",
            "contact_check_frequency_hz": 1.0 / control_dt,
            "handle_contact_criteria": asdict(AssemblyCriteria()),
        },
        "states": {
            kind: {field: torch.cat(values) for field, values in rows.items()}
            for kind, rows in collected.items()
        },
    }
    if source is not None:
        bank["provenance"]["reused_categories"] = {
            "names": list(RESET_TYPES[:3]),
            "source_bank": str(source_bank),
            "source_provenance": source["provenance"],
        }
    if insertion_per_stage is not None:
        bank["insertion_subtypes"] = {
            "names": ["above_peg", "threaded"],
            "labels": torch.cat(insertion_labels),
            "counts": [insertion_per_stage, insertion_per_stage],
        }
        bank["provenance"]["insertion_subtypes"] = {
            "names": ["above_peg", "threaded"],
            "counts": [insertion_per_stage, insertion_per_stage],
            "label_field": "insertion_subtypes.labels",
            "legacy_category_key": RESET_TYPES[3],
        }
    DEFAULT_ROBOT.project_reset_fingers(
        bank, torch.tensor(sampler.limits, dtype=torch.float32), criteria.finger_upper_tolerance
    )
    bank["provenance"]["finger_position_projection"]["projected_contact_hold_revalidated"] = (
        source is None
    )
    validate_bank(bank, sampler.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(bank, stream)
    print(f"Saved {counts} to {output}", flush=True)


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Collect a hold-validated Square reset bank; --source-bank reuses the first "
            "three categories of a compatible bank."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    for name, default in (
        ("per-type", 10000),
        ("num-envs", 1024),
        ("max-batches", 100),
        ("hold-steps", 20),
        ("settle-steps", 100),
        ("descent-steps", DESCENT_STEPS),
        ("seed", 42),
        ("proposal-workers", DEFAULT_PROPOSAL_WORKERS),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    parser.add_argument("--nconmax", type=int)
    parser.add_argument("--njmax", type=int)
    parser.add_argument("--source-bank", type=Path)
    parser.add_argument("--insertion-per-stage", type=int)
    parser.add_argument(
        "--split", choices=("development", "train", "validation", "test"), default="train"
    )
    collect(**vars(parser.parse_args(argv)))
