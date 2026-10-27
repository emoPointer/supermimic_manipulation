"""Collect open-start Drawer resets from real grasp/lift, release and push trajectories.

Run through ``clab resets collect --task drawer``. Each world runs one complete physical
sequence and owns its failure mask, process history and three paired snapshots; only a
complete success within the task0-relative horizon is accepted. No prefix is assigned
directly: each is earned by the task's physical state machine. ``--nconmax``/``--njmax``
set simulator capacities, and an overflow fails the run rather than individual rows. A
sibling diagnostics JSON records failures; existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from clab.controllers.pose import pose_target_action
from clab.logging.model import joint_damping_metadata
from clab.observations.state import pose
from clab.robots.kinematics import SiteIK
from clab.robots.parallel_kinematics import DEFAULT_PROPOSAL_WORKERS, CpuProposalPool

from ..pregrasp import append_pregrasp
from . import mdp
from .config import make_env_cfg
from .props import DRAWER_BOXES, POD_INITIAL_Z, POD_TOP_OFFSET
from .reset_data import FIELDS, SCHEMA, WIDTHS, bank_contract, validate_bank
from .resets import write_states
from .settings import (
    COLLECTED_TYPES,
    EPISODE_SECONDS,
    HALF_CLOSED_SLIDE,
    INITIAL_SLIDE_RANGE,
    PREFIX_FLAGS,
    RESET_TYPES,
)
from .variants import DEFAULT_ROBOT


class CandidateRejected(RuntimeError):
    """An expected physics/kinematics rejection, not an implementation failure."""


def downward_quaternion(yaw):
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    result = np.empty(4)
    mujoco.mju_mulQuat(
        result,
        np.array((np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))),
        np.array((2**-0.5, 0, 2**-0.5, 0)),
    )
    return result


def initial_candidate(rng, ik):
    """Independent safe joint pose; source placement bounds, no scale adjustment."""
    rows = {key: torch.zeros((1, width)) for key, width in zip(FIELDS, WIDTHS, strict=True)}
    yaw = rng.uniform(np.pi - np.pi / 12, np.pi + np.pi / 12)
    housing = np.array((rng.uniform(0.325, 0.335), rng.uniform(-0.08, -0.04), 0))
    pod = np.array((rng.uniform(0.075, 0.205), rng.uniform(0.09, 0.25), POD_INITIAL_Z))
    q = np.array(DEFAULT_ROBOT.home)
    q[:6] += rng.uniform(-0.08, 0.08, 6)
    q[6:] = 0.044
    if ((q[:6] < ik.limits[:, 0]) | (q[:6] > ik.limits[:, 1])).any():
        raise CandidateRejected("initial joint pose exceeds limits")
    quat = np.array((np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
    slide = rng.uniform(*INITIAL_SLIDE_RANGE)
    rows["joint_pos"][:] = torch.from_numpy(q)
    pod_yaw = rng.uniform(-0.1 * np.pi, 0.1 * np.pi)
    rows["pod_pose"][0] = torch.tensor((*pod, np.cos(pod_yaw / 2), 0, 0, np.sin(pod_yaw / 2)))
    rows["housing_pose"][0] = torch.tensor((*housing, *quat))
    rows["slide_pos"][:] = rows["initial_drawer_qpos"][:] = slide
    rows["initial_pod_height"][:] = pod[2]
    rows["close_start_qpos"][:] = slide
    # Check key targets with the real grasp site before consuming a GPU rollout.
    down = downward_quaternion(yaw)
    # High-side grasp leaves fingertip clearance for later supported placement
    # even when the cone moves upward relative to the closed parallel fingers.
    grasp = pod + (0, 0, POD_TOP_OFFSET - 0.004)
    targets = (
        grasp + (0, 0, 0.1),
        grasp,
        grasp + (0, 0, 0.12),
    )
    seed = q[:6]
    for target in targets:
        solved = ik.solve(target, down, seed)
        if solved is None:
            raise CandidateRejected("key grasp-site target has no in-limit IK witness")
        seed = solved
    return rows, down


def interpolate_quaternions(start, target, fraction):
    """Shortest-arc batched SLERP, without mutating either input quaternion."""
    dot = (start * target).sum(-1, keepdim=True)
    target = torch.where(dot < 0, -target, target)
    dot = dot.abs().clamp(0, 1)
    angle = torch.acos(dot)
    sine = torch.sin(angle).clamp_min(torch.finfo(start.dtype).eps)
    spherical = (
        start * torch.sin((1 - fraction) * angle) + target * torch.sin(fraction * angle)
    ) / sine
    linear = (1 - fraction) * start + fraction * target
    result = torch.where(dot > 1 - 4 * torch.finfo(start.dtype).eps, linear, spherical)
    return result / result.norm(dim=-1, keepdim=True)


def snapshot(env, category, active=None):
    """Save the actual end state, with only prefixes already physically witnessed."""
    seq = mdp.sequence(env)
    flags = seq.prefix | seq.accomplished[:, :2]
    if active is None:
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    expected = torch.tensor(PREFIX_FLAGS[category], device=env.device).bool().expand_as(flags)
    if not torch.equal(flags[active], expected[active]) or seq.failed[active].any():
        raise CandidateRejected(f"category {category} lacks exactly its physical prefix")
    robot, pod, fixture = env.scene["robot"], env.scene["pod"], env.scene["fixture"]
    pod_pose, housing_pose = (
        pod.data.root_link_pose_w.clone(),
        fixture.data.root_link_pose_w.clone(),
    )
    pod_pose[:, :3] -= env.scene.env_origins
    housing_pose[:, :3] -= env.scene.env_origins
    result = {
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "pod_pose": pod_pose,
        "pod_vel": torch.cat((pod.data.root_link_lin_vel_w, pod.data.root_link_ang_vel_w), -1),
        "housing_pose": housing_pose,
        "slide_pos": fixture.data.joint_pos.clone(),
        "slide_vel": fixture.data.joint_vel.clone(),
        "sequence_prefix": flags.float(),
        "initial_drawer_qpos": seq.initial_q[:, None].clone(),
        "initial_pod_height": seq.initial_pod_z[:, None].clone(),
        # The policy's push begins at this physical state, as source begin_close.
        "close_start_qpos": fixture.data.joint_pos.clone(),
    }
    return {key: value.detach().cpu().float().clone() for key, value in result.items()}


class PhysicalTrajectory:
    """Physical-state expert; only sends actions, never writes objects mid-episode.

    Unlike policy observations, geometric waypoints intentionally use true
    physical poses. Their world-frame corrective increments are resolved in
    world-axis relative OSC commands re-anchored at the current pose.
    """

    def __init__(self, env):
        self.env = env
        self.action_cfg = env.cfg.actions[DEFAULT_ROBOT.action_name]
        self.events = [[] for _ in range(env.num_envs)]
        self.steps = 0
        self.active_control_steps = torch.zeros((), dtype=torch.long, device=env.device)
        self.task0_step = None
        self.phase = "initialization"
        self.alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.completed = torch.zeros_like(self.alive)
        self.completion_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.rejections = [None] * env.num_envs
        self.minimum_contact_distance = torch.zeros(env.num_envs, device=env.device)
        self.maximum_nefc = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.target_pos, self.target_quat = (
            value.clone() for value in pose(env, "robot", DEFAULT_ROBOT.ee_body)
        )
        ids = [env.sim.mj_model.joint(f"robot/{name}").id for name in DEFAULT_ROBOT.joints]
        self.limits = torch.tensor(env.sim.mj_model.jnt_range[ids], device=env.device)
        self.pod_geom = env.sim.mj_model.geom("pod/pod_collision").id
        mesh = int(env.sim.mj_model.geom_dataid[self.pod_geom])
        start = env.sim.mj_model.mesh_vertadr[mesh]
        vertices = env.sim.mj_model.mesh_vert[start : start + env.sim.mj_model.mesh_vertnum[mesh]]
        self.pod_vertices = torch.tensor(vertices.copy(), device=env.device)

    @property
    def active(self):
        return self.alive & ~self.completed

    def reject(self, mask, reason):
        """Discard complete paired trajectories; rejected worlds cannot revive.

        Reset only discarded worlds to a safe default to avoid integrating their
        damaged states alongside still-valid trajectories. Their old snapshots
        are never accepted, and no new trajectory begins until the next batch.
        """
        indices = torch.where(mask & self.active)[0]
        if not len(indices):
            return
        self.alive[indices] = False
        for index in indices.cpu().tolist():
            self.rejections[index] = reason
        self.park(indices)

    def park(self, indices):
        """Clear manual-reset pending for finished/discarded worlds only."""
        self.env.reset(env_ids=indices)
        state = mdp.sequence(self.env)
        state.failed[indices] = True
        state.success[indices] = False
        state.final_count[indices] = 0
        current_pos, current_quat = pose(self.env, "robot", DEFAULT_ROBOT.ee_body)
        self.target_pos[indices] = current_pos[indices]
        self.target_quat[indices] = current_quat[indices]

    def require(self, condition, reason):
        self.reject(~condition, reason)

    def save(self, category, worlds=None):
        """Apply the saved-row contract before pairing a snapshot of ``worlds``."""
        worlds = self.alive if worlds is None else worlds
        seq = mdp.sequence(self.env)
        flags = seq.prefix | seq.accomplished[:, :2]
        expected = torch.tensor(PREFIX_FLAGS[category], device=self.env.device).bool()
        q = self.env.scene["robot"].data.joint_pos
        slide = self.env.scene["fixture"].data.joint_pos[:, 0]
        valid = (flags == expected).all(-1) & ~seq.failed
        # Same approved finger soft-stop allowance as the other collectors; the
        # exported rows are projected back into range before saving.
        valid &= DEFAULT_ROBOT.reset_joint_position_mask(
            q, self.limits, DEFAULT_ROBOT.reset_finger_upper_tolerance
        )
        valid &= (slide >= 0) & (slide <= 0.14)
        if category == 0:
            valid &= (slide - seq.initial_q).abs() <= 1e-5
        if PREFIX_FLAGS[category][1]:
            valid &= slide >= 0.045
        self.require(
            valid | ~worlds, f"category {category} saved-row prefix/reference/limit contract"
        )
        return snapshot(self.env, category, worlds & self.alive)

    def check(self):
        env = self.env
        if wp.to_torch(env.sim.wp_data.overflow).any().item():
            raise RuntimeError("MuJoCo-Warp capacity overflow; collection is invalid")
        nefc = wp.to_torch(env.sim.wp_data.nefc).reshape(env.num_envs, -1).amax(-1)
        self.maximum_nefc = torch.where(
            self.active, torch.maximum(self.maximum_nefc, nefc.long()), self.maximum_nefc
        )
        if (
            not torch.isfinite(env.sim.data.qpos).all()
            or not torch.isfinite(env.sim.data.qvel).all()
        ):
            raise RuntimeError("Non-finite physics state")
        q = env.scene["robot"].data.joint_pos
        self.reject(
            ((q < self.limits[:, 0] - 1e-4) | (q > self.limits[:, 1] + 1e-4)).any(-1),
            f"{self.phase}: robot joint limit overshoot",
        )
        slide = env.scene["fixture"].data.joint_pos
        self.reject(
            ((slide < -1e-4) | (slide > 0.14 + 1e-4)).any(-1),
            f"{self.phase}: passive slide joint limit overshoot",
        )
        for name in ("robot_contact", "pod_table_contact", "pod_fixture_contact"):
            sensor = env.scene[name].data
            minimum = (
                torch.where(sensor.found > 0, sensor.dist, 0).reshape(env.num_envs, -1).amin(-1)
            )
            self.minimum_contact_distance = torch.where(
                self.active,
                torch.minimum(self.minimum_contact_distance, minimum),
                self.minimum_contact_distance,
            )
            self.reject(minimum < -0.001, f"{self.phase}: {name} penetration exceeds 1 mm")
        self.reject(mdp.sequence(env).failed, f"{self.phase}: physical process failed")

    def move(
        self, target_pos, target_quat, seconds, gripper, phase, *, interpolate=True, moving=None
    ):
        env = self.env
        if not self.active.any():
            return
        self.phase = phase
        if moving is not None:
            target_pos = torch.where(moving[:, None], target_pos, self.target_pos)
            target_quat = torch.where(moving[:, None], target_quat, self.target_quat)
        self.target_pos, self.target_quat = target_pos.clone(), target_quat.clone()
        start_pos, start_quat = (
            value.clone() for value in pose(env, "robot", DEFAULT_ROBOT.ee_body)
        )
        count = max(1, math.ceil(seconds / env.step_dt))
        for step in range(1, count + 1):
            if self.task0_step is not None and self.steps - self.task0_step >= math.floor(
                EPISODE_SECONDS / env.step_dt + 1e-9
            ):
                self.reject(self.active, "complete trajectory exceeds task0-relative RL horizon")
            if not self.active.any():
                return
            fraction = step / count if interpolate else 1.0
            fraction = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
            desired_pos = start_pos + fraction * (target_pos - start_pos)
            desired_quat = interpolate_quaternions(start_quat, target_quat, fraction)
            current_pos, current_quat = pose(env, "robot", DEFAULT_ROBOT.ee_body)
            action = pose_target_action(
                current_pos,
                current_quat,
                desired_pos,
                desired_quat,
                action_cfg=self.action_cfg,
                gripper_command=gripper,
            )
            action[~self.active] = 0
            action[~self.alive, 6] = 1
            old_stage = mdp.sequence(env).stage.clone()
            self.active_control_steps += self.active.sum()
            env.step(action)
            self.steps += 1
            self.check()
            seq = mdp.sequence(env)
            changed = torch.where(self.alive & (seq.stage != old_stage))[0]
            if len(changed):
                values = (
                    torch.stack(
                        (
                            seq.stage,
                            env.scene["fixture"].data.joint_pos[:, 0],
                            pose(env, "pod")[0][:, 2] - env.scene.env_origins[:, 2],
                            seq.grasp_count,
                            seq.release_count,
                            seq.push_contact,
                        ),
                        -1,
                    )[changed]
                    .cpu()
                    .tolist()
                )
                for index, (stage, slide, height, grasp, release, push) in zip(
                    changed.cpu().tolist(), values, strict=True
                ):
                    self.events[index].append(
                        {
                            "step": self.steps,
                            "stage": int(stage),
                            "slide": slide,
                            "pod_height": height,
                            "grasp_steps": int(grasp),
                            "release_steps": int(release),
                            "push_contact": bool(push),
                        }
                    )
            finished = self.active & seq.success
            self.completion_steps[finished] = self.steps
            self.completed |= finished
            finished_ids = torch.where(finished)[0]
            if len(finished_ids):
                self.park(finished_ids)

    def hold(self, seconds, gripper, phase):
        """Continue tracking the commanded target; do not freeze tracking lag."""
        self.move(self.target_pos, self.target_quat, seconds, gripper, phase, interpolate=False)

    def grasp_target(self, target, quat, seconds, gripper, phase):
        offset = torch.tensor(DEFAULT_ROBOT.grasp_offset, device=self.env.device).expand(
            self.env.num_envs, -1
        )
        self.move(target - quat_apply(quat, offset), quat, seconds, gripper, phase)

    def pod_height_bounds(self):
        """Actual collision-mesh Z bounds in the fixed housing frame, in metres."""
        env = self.env
        housing_pos, housing_quat = pose(env, "fixture", "housing")
        world = env.sim.data.geom_xpos[:, self.pod_geom, None] + torch.einsum(
            "nij,kj->nki",
            env.sim.data.geom_xmat[:, self.pod_geom].reshape(-1, 3, 3),
            self.pod_vertices,
        )
        local = quat_apply_inverse(
            housing_quat[:, None].expand(-1, len(self.pod_vertices), -1),
            world - housing_pos[:, None],
        )
        return local[:, :, 2].amin(-1), local[:, :, 2].amax(-1)

    def pod_bottom(self):
        return self.pod_height_bounds()[0]

    def place_above_floor(self, ee_quat):
        """Move the measured pod above support using only normal OSC actions.

        Recompute object/hand translation every control check because the cone
        can slip inside a valid grasp. Release clearance uses actual mesh points,
        not a frozen object/EE offset or an arbitrary object-root height.
        """
        env = self.env
        housing_pos, housing_quat = pose(env, "fixture", "housing")
        start_xy = quat_apply_inverse(housing_quat, pose(env, "pod")[0] - housing_pos)[:, :2]
        start_bottom = self.pod_bottom().clone()
        floor = DRAWER_BOXES[-1][0][2] + DRAWER_BOXES[-1][1][2]
        target_bottom = floor + 0.0003
        approach_steps = math.ceil(2.0 / env.step_dt)
        count = approach_steps + math.ceil(2.0 / env.step_dt)
        target_xy = torch.tensor([[0.145, 0.0]], device=env.device)
        ready = torch.zeros_like(self.alive)
        for step in range(1, count + 1):
            fraction = min(step / approach_steps, 1.0)
            fraction = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
            pod_local = quat_apply_inverse(housing_quat, pose(env, "pod")[0] - housing_pos)
            translation = torch.cat(
                (
                    start_xy + fraction * (target_xy - start_xy) - pod_local[:, :2],
                    (start_bottom + fraction * (target_bottom - start_bottom) - self.pod_bottom())[
                        :, None
                    ],
                ),
                -1,
            )
            current_ee = pose(env, "robot", DEFAULT_ROBOT.ee_body)[0]
            self.move(
                current_ee + quat_apply(housing_quat, translation),
                ee_quat,
                env.step_dt,
                -1,
                "pod_place",
                interpolate=False,
                moving=~ready,
            )
            if step >= approach_steps:
                ready |= self.active & (self.pod_bottom() <= floor + 0.0005)
            if (ready | ~self.active).all():
                return
        self.reject(~ready, "pod did not approach within 0.5 mm of drawer support")

    def close_drawer(self, down, half_category):
        """Push the drawer closed; return each world's snapshot at the half-closed slide."""
        housing_pos, housing_quat = pose(self.env, "fixture", "housing")
        half, half_seen = None, torch.zeros_like(self.alive)
        for local, seconds, phase in (
            ((0.257, 0.0, 0.11), 1.8, "close_hover"),
            ((0.257, 0.0, 0.04), 0.8, "close_outside"),
            ((0.222, 0.0, 0.04), 0.7, "close_contact"),
        ):
            target = housing_pos + quat_apply(
                housing_quat, torch.tensor(local, device=self.env.device).expand_as(housing_pos)
            )
            self.grasp_target(target, down, seconds, -1, phase)
            if phase == "close_outside":
                # Finish descending outside the panel before lateral contact;
                # otherwise tracking lag makes the hull press its top edge.
                self.hold(1.5, -1, "close_outside_hold")
        # Follow the physically moving panel rather than accumulating a large
        # displacement error against a stationary contact. This changes only
        # the scripted path; the common controller and binary gripper are used.
        offset = torch.tensor(DEFAULT_ROBOT.grasp_offset, device=self.env.device).expand_as(
            housing_pos
        )
        stopped = torch.zeros_like(self.alive)
        stop_pos = pose(self.env, "robot", DEFAULT_ROBOT.ee_body)[0].clone()
        for _ in range(math.ceil(30.0 / self.env.step_dt)):
            if not self.active.any():
                return half
            slide = self.env.scene["fixture"].data.joint_pos[:, 0]
            crossing = self.active & ~half_seen & (slide <= HALF_CLOSED_SLIDE)
            if crossing.any():
                rows = self.save(half_category, crossing)
                if half is None:
                    half = {key: torch.zeros_like(value) for key, value in rows.items()}
                mask = crossing.cpu()
                for key, value in rows.items():
                    half[key][mask] = value[mask]
                half_seen |= crossing
            current_pos, current_quat = pose(self.env, "robot", DEFAULT_ROBOT.ee_body)
            at_stop = self.active & (slide <= 0.025)
            stop_pos[at_stop & ~stopped] = current_pos[at_stop & ~stopped]
            stopped |= at_stop
            if (stopped | ~self.active).all():
                self.move(stop_pos, down, self.env.step_dt, -1, "push_stop", interpolate=False)
                break
            grip = current_pos + quat_apply(current_quat, offset)
            local = quat_apply_inverse(housing_quat, grip - housing_pos)
            local[:, 0] -= 0.05
            local[:, 1] = 0.0
            local[:, 2] = 0.04
            target = housing_pos + quat_apply(housing_quat, local) - quat_apply(down, offset)
            target = torch.where(stopped[:, None], stop_pos, target)
            self.move(target, down, self.env.step_dt, -1, "push", interpolate=False)
        else:
            self.reject(~stopped, "incremental contact push did not close drawer below 25 mm")
        self.require(half_seen, "push skipped the half-closed snapshot")
        self.hold(1.5, -1, "final_success_hold")
        return half


def collect_trajectory(env, down, eligible=None):
    trajectory = PhysicalTrajectory(env)
    if eligible is not None:
        trajectory.alive &= eligible
    trajectory.check()
    trajectory.hold(1.0, 1.0, "natural_settle")
    seq = mdp.sequence(env)
    local, height, slide, _ = mdp.geometry(env)
    del local
    # Measured start references after settling, as in the source collector.
    seq.reset(
        torch.arange(env.num_envs, device=env.device),
        torch.zeros((env.num_envs, 2), device=env.device),
        slide.clone(),
        height.clone(),
        slide.clone(),
    )
    trajectory.require(
        mdp.contact_found(env, "pod_table_contact"),
        "task0 pod is not supported on table after settling",
    )
    records = {RESET_TYPES[0]: trajectory.save(0)}
    trajectory.task0_step = trajectory.steps
    down = (
        torch.as_tensor(down, dtype=torch.float32, device=env.device)
        .reshape(-1, 4)
        .expand(env.num_envs, -1)
    )
    housing_pos, housing_quat = pose(env, "fixture", "housing")

    def fixture_target(local):
        return housing_pos + quat_apply(
            housing_quat, torch.tensor(local, device=env.device).expand_as(housing_pos)
        )

    pod_pos = pose(env, "pod")[0].clone()
    pod_local = quat_apply_inverse(housing_quat, pod_pos - housing_pos)
    grasp_offset = torch.zeros_like(pod_pos)
    grasp_offset[:, 2] = trajectory.pod_height_bounds()[1] - 0.004 - pod_local[:, 2]
    grasp_pos = pod_pos + quat_apply(housing_quat, grasp_offset)
    lift = torch.tensor([[0.0, 0.0, 0.12]], device=env.device)
    trajectory.grasp_target(grasp_pos + lift, down, 2.4, 1, "pod_hover")
    trajectory.grasp_target(grasp_pos, down, 1.0, 1, "pod_descend")
    # Closing while the compliant arm is still descending squeezes the pod's
    # upper rim. Finish the approach with open fingers before closing.
    trajectory.hold(1.5, 1, "pod_descend_hold")
    trajectory.hold(0.6, -1, "pod_grasp")
    records[RESET_TYPES[1]] = trajectory.save(1)
    trajectory.grasp_target(grasp_pos + lift, down, 1.2, -1, "pod_lift")
    trajectory.hold(0.5, -1, "task1_hold")
    trajectory.require(seq.stage == 1, "pod did not achieve six-step grasp and 40 mm physical lift")
    records[RESET_TYPES[2]] = trajectory.save(2)

    # Preserve measured object/EE translation at the current orientation. No
    # object teleport or force/constraint attachment is introduced.
    ee_pos, ee_quat = (value.clone() for value in pose(env, "robot", DEFAULT_ROBOT.ee_body))
    pod_offset = pose(env, "pod")[0].clone() - ee_pos
    trajectory.move(
        fixture_target((0.145, 0.0, 0.155)) - pod_offset, ee_quat, 2.4, -1, "pod_transport"
    )
    trajectory.require(seq.stage == 1, "pod was not still grasped above the drawer")
    records[RESET_TYPES[3]] = trajectory.save(3)
    trajectory.place_above_floor(ee_quat)
    trajectory.hold(0.6, 1, "pod_release")
    trajectory.require(seq.stage == 2, "pod did not remain released inside for six control checks")
    trajectory.move(
        fixture_target((0.145, 0.0, 0.10)) - pod_offset, ee_quat, 0.8, 1, "release_retreat"
    )
    trajectory.hold(0.5, 1, "task2_hold")
    records[RESET_TYPES[4]] = trajectory.save(4)

    half = trajectory.close_drawer(down, 5)
    if half is not None:
        records[RESET_TYPES[5]] = half
    trajectory.reject(
        ~trajectory.completed,
        "complete process did not achieve continuous one-second final success",
    )
    witnesses = []
    completion_steps = trajectory.completion_steps.cpu().tolist()
    minimum_distance = trajectory.minimum_contact_distance.cpu().tolist()
    maximum_nefc = trajectory.maximum_nefc.cpu().tolist()
    for index in torch.where(trajectory.alive & trajectory.completed)[0].cpu().tolist():
        steps = completion_steps[index]
        witnesses.append(
            {
                "world": index,
                "events": trajectory.events[index],
                "control_steps": steps,
                "task0_control_step": trajectory.task0_step,
                "remaining_control_steps": steps - trajectory.task0_step,
                "minimum_sampled_contact_distance": minimum_distance[index],
                "maximum_sampled_nefc": maximum_nefc[index],
            }
        )
    return (
        records,
        witnesses,
        Counter(reason for reason in trajectory.rejections if reason is not None),
        {
            "control_steps": trajectory.steps,
            "simulated_world_control_steps": env.num_envs * trajectory.steps,
            "active_world_control_steps": int(trajectory.active_control_steps),
        },
    )


def make_kinematic_witness():
    model = Scene(make_env_cfg().scene, device="cpu").compile()
    return SiteIK(
        model,
        tuple(f"robot/{name}" for name in DEFAULT_ROBOT.arm_joints),
        f"robot/{DEFAULT_ROBOT.grasp_site}",
    )


def propose_candidate(ik, seed_words):
    """CPU-pool job with ordinary array IPC, not one shared mapping per field.

    Thousands of tiny Torch storages exhaust the parent's VMA limit across
    batches. NumPy transport preserves the candidate values/dtypes and seed
    stream without creating shared Torch storage file descriptors.
    """
    try:
        rows, down = initial_candidate(np.random.default_rng(seed_words), ik)
        return ({key: value.numpy() for key, value in rows.items()}, down), None
    except CandidateRejected as error:
        return None, str(error)


def collect(
    output: Path,
    *,
    per_type=1,
    max_attempts=20,
    num_envs=1,
    proposal_workers=DEFAULT_PROPOSAL_WORKERS,
    seed=42,
    device="cuda:0",
    cuda_graph_mode="auto",
    split="development",
    nconmax=None,
    njmax=None,
):
    report_path = output.with_suffix(".diagnostics.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing Drawer bank or diagnostics")
    if min(per_type, num_envs, proposal_workers) < 1 or max_attempts < per_type:
        raise ValueError("Require positive world/worker counts and 1 <= per_type <= max_attempts")
    if split not in ("development", "train", "evaluation"):
        raise ValueError("Unknown Drawer dataset split")
    cfg = make_env_cfg()
    cfg.scene.num_envs = num_envs
    for name, value in (("nconmax", nconmax), ("njmax", njmax)):
        if value is not None:
            if value < 1:
                raise ValueError(f"{name} must be positive")
            setattr(cfg.sim, name, value)
    cfg.events.pop("reset_from_bank")
    cfg.rewards = {}
    cfg.terminations.pop("time_out")
    cfg.auto_reset = False
    model = Scene(cfg.scene, device="cpu").compile()
    ik = make_kinematic_witness() if proposal_workers == 1 else None
    pool = (
        CpuProposalPool(make_kinematic_witness, seed=seed, workers=proposal_workers)
        if proposal_workers > 1
        else None
    )
    rng = np.random.default_rng(seed)
    collected = {kind: {key: [] for key in FIELDS} for kind in COLLECTED_TYPES}
    successes, rejections, batches = [], Counter(), []
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    # Rows must come from one unbroken process: any lost grasp, outside release or
    # escape latches failure, and the collector rejects that world itself after the
    # control step (the task never fails, so no termination resets it).
    mdp.sequence(env).strict = True
    try:
        free_joint_damping = joint_damping_metadata(env.sim.mj_model, "pod/pod_free")
        # Compile reset/forward/sense before optional capture on old drivers.
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
                raise RuntimeError("Plain graph capture failed")
        elif cuda_graph_mode != "auto":
            raise ValueError("Unknown CUDA graph mode")
        attempt, batch = 0, 0
        while len(successes) < per_type and attempt < max_attempts:
            proposal_start = time.perf_counter()
            count = min(num_envs, max_attempts - attempt)
            proposals = []
            if pool is not None:
                results = []
                for proposal, error in pool.map(propose_candidate, [()] * count):
                    if proposal is not None:
                        fields, down = proposal
                        proposal = (
                            {key: torch.from_numpy(value) for key, value in fields.items()},
                            down,
                        )
                    results.append((proposal, error))
            else:
                results = []
                for _ in range(count):
                    try:
                        results.append((initial_candidate(rng, ik), None))
                    except CandidateRejected as error:
                        results.append((None, str(error)))
            for offset, (proposal, error) in enumerate(results):
                if error is not None:
                    rejections[error] += 1
                else:
                    proposals.append((attempt + offset, proposal))
            attempt += count
            batch += 1
            proposal_seconds = time.perf_counter() - proposal_start
            if not proposals:
                continue
            # Padding keeps the model's fixed CUDA batch size. Padded worlds are
            # permanently ineligible and never contribute rows or witnesses.
            padded = proposals + [proposals[0]] * (num_envs - len(proposals))
            rows = {
                key: torch.cat([proposal[0][key] for _, proposal in padded]).to(device)
                for key in FIELDS
            }
            down = np.stack([proposal[1] for _, proposal in padded])
            eligible = torch.arange(num_envs, device=device) < len(proposals)
            physics_start = time.perf_counter()
            env.reset(seed=seed + batch - 1)
            write_states(env, rows)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.sim.sense()
            records, witnesses, rejected, batch_stats = collect_trajectory(env, down, eligible)
            batch_stats.update(
                batch=batch,
                proposed=count,
                kinematic_candidates=len(proposals),
                complete_trajectories=len(witnesses),
                proposal_seconds=proposal_seconds,
                physics_seconds=time.perf_counter() - physics_start,
            )
            batches.append(batch_stats)
            rejections.update(rejected)
            for witness in witnesses[: per_type - len(successes)]:
                world = witness["world"]
                for kind, record in records.items():
                    for key, value in record.items():
                        collected[kind][key].append(value[world : world + 1])
                witness["attempt"] = proposals[world][0]
                witness["batch"] = batch
                successes.append(witness)
            print(
                f"Drawer batch {batch}: {len(successes)}/{per_type} full physical trajectories accepted; {attempt} proposals; proposal={proposal_seconds:.2f}s physics={batch_stats['physics_seconds']:.2f}s; rejections={dict(rejections)}",
                flush=True,
            )
        report = {
            "accepted_trajectories": len(successes),
            "rejections": dict(rejections),
            "physical_witnesses": successes,
            "batches": batches,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2)
        if len(successes) != per_type:
            raise RuntimeError(f"Drawer collection incomplete: {report}; no bank written")
        bank = dict(
            schema=SCHEMA,
            contract=bank_contract(),
            split=split,
            provenance={
                "method": "full_physical_open_start_drawer_prefix_trajectories",
                "prefix_validation": "physical_trajectory_witness",
                "seed": seed,
                "num_envs": num_envs,
                "proposal_workers": proposal_workers,
                "proposal_transport": "numpy_array_ipc" if pool else "in_process_tensors",
                "capacity": {"nconmax": cfg.sim.nconmax, "njmax": cfg.sim.njmax},
                "candidate_seed_stream": "ordered_seedsequence_jobs"
                if pool
                else "single_numpy_generator",
                "horizon_acceptance": {
                    "reference": "task0_snapshot",
                    "maximum_remaining_control_steps": math.floor(
                        EPISODE_SECONDS / env.step_dt + 1e-9
                    ),
                    "includes_continuous_final_success": True,
                },
                "controller": asdict(cfg.actions[DEFAULT_ROBOT.action_name]),
                "control_dt": env.step_dt,
                "physics_dt": cfg.sim.mujoco.timestep,
                "decimation": cfg.decimation,
                "parameter_randomization": "none_nominal_physics",
                "free_joint_damping": free_joint_damping,
                "initial_regions": "source_collection_regions_environment_local_world",
                "quality_contact_sampling_hz": 1 / env.step_dt,
                "physical_witnesses": successes,
                "batches": batches,
            },
            states={
                kind: {key: torch.cat(values) for key, values in fields.items()}
                for kind, fields in collected.items()
            },
        )
        limits = [model.joint(f"robot/{name}").range for name in DEFAULT_ROBOT.joints]
        DEFAULT_ROBOT.project_reset_fingers(
            bank,
            torch.tensor(np.stack(limits), dtype=torch.float32),
            DEFAULT_ROBOT.reset_finger_upper_tolerance,
        )
        bank = append_pregrasp(
            bank, "drawer", seed=seed, device=device, cuda_graph_mode=cuda_graph_mode
        )
        validate_bank(bank, model)
        with output.open("xb") as stream:
            torch.save(bank, stream)
    finally:
        env.close()
        if pool is not None:
            pool.close()


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=("Collect a Drawer reset bank from complete physical prefix trajectories."),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=10000)
    parser.add_argument("--max-attempts", type=int, help="Default: four times --per-type")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--proposal-workers", type=int, default=DEFAULT_PROPOSAL_WORKERS)
    parser.add_argument("--nconmax", type=int)
    parser.add_argument("--njmax", type=int)
    parser.add_argument("--split", choices=("development", "train", "evaluation"), default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    args = parser.parse_args(argv)
    if args.max_attempts is None:
        args.max_attempts = 4 * args.per_type
    collect(**vars(args))
