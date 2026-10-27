"""Open-gripper pre-grasp reset category, derived physically from grasped-on-table rows.

The reverse curriculum's first grasped category starts with the object already held.
Nothing bridged it to the full task (gripper free, often half a metre away): the
full-task policy never opened its hand at the object, so the critic never valued an
open hand there. Each derived row starts from one grasped-on-table row, opens the
gripper, re-aims at the same object-frame grasp point (opening may shift the object),
rises straight up and then moves to a small random offset above it, and holds. A row
is accepted only if the object then rests stably, no finger touches it during the
hold, the fingers are open and the grasp point is above the grasp region. The saved
state is the actual end state; nothing is projected except the shared finger-limit
projection of exported data.
"""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from clab.controllers.torch_kinematics import quat_apply, quat_inverse
from clab.observations.state import pose
from clab.simulation import configure_cuda_graph


@dataclass(frozen=True)
class PregraspConfig:
    open_seconds: float = 0.9
    move_seconds: float = 2.0
    hold_seconds: float = 0.5
    rise_range: tuple[float, float] = (0.01, 0.06)
    lateral: float = 0.01
    # Rise this far before any sideways motion: at grasp height it drags the object.
    clearance: float = 0.015
    max_tilt_degrees: float = 3.0
    max_linear_speed: float = 0.01
    max_angular_speed: float = 0.1
    min_finger_opening: float = 0.035
    max_passes: int = 6


def _contact(env, name):
    return (env.scene[name].data.found > 0).any(-1)


class SquarePregrasp:
    task = "square"
    object_name = "nut"
    source_index = 1  # NutRestingHandleGrasped
    paired = False

    def __init__(self):
        from .square import config, mdp, resets
        from .square.settings import RESET_TYPES, AssemblyCriteria
        from .square.variants import DEFAULT_ROBOT

        self.make_env_cfg, self.mdp, self.resets = config.make_env_cfg, mdp, resets
        self.robot = DEFAULT_ROBOT
        self.criteria = AssemblyCriteria()
        self.name = RESET_TYPES[-1]

    def configure(self, cfg):
        cfg.terminations.pop("success")

    def touching(self, env):
        return _contact(env, "nut_robot_contact") | _contact(env, "robot_peg_contact")

    def wrong(self, env):
        # Both fingers bearing on the ring (or ring and handle) while the gripper opens.
        return self.mdp.handle_grasp_status(env, self.criteria)[1]

    def above(self, grip, obj_pos, obj_quat, config):
        from .square.props import NUT_BOXES

        (centre, half), ring_outer = NUT_BOXES[4], max(p[0] + h[0] for p, h in NUT_BOXES[:4])
        local = quat_apply(quat_inverse(obj_quat), grip - obj_pos)
        return (
            (local[:, 0] >= ring_outer)
            & (local[:, 0] <= centre[0] + half[0])
            & (local[:, 1].abs() <= half[1] + config.lateral + 0.002)
            & (grip[:, 2] - obj_pos[:, 2] >= config.rise_range[0] / 2)
        )

    def rows(self, env, source):
        return self.resets.snapshot(env)


class DrawerPregrasp:
    task = "drawer"
    object_name = "pod"
    source_index = 1  # DrawerOpenedPodGraspedOnTable
    paired = True

    def __init__(self):
        from .drawer import config, resets
        from .drawer.settings import RESET_TYPES
        from .drawer.variants import DEFAULT_ROBOT

        self.make_env_cfg, self.resets = config.make_env_cfg, resets
        self.robot = DEFAULT_ROBOT
        self.name = RESET_TYPES[-1]

    def configure(self, cfg):
        for name in ("success", "process_failure"):
            cfg.terminations.pop(name, None)

    def touching(self, env):
        return (
            _contact(env, "left_pod_contact")
            | _contact(env, "right_pod_contact")
            | _contact(env, "left_drawer_contact")
            | _contact(env, "right_drawer_contact")
        )

    def wrong(self, env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def above(self, grip, obj_pos, obj_quat, config):
        offset = grip - obj_pos
        return (offset[:, :2].norm(dim=-1) <= math.sqrt(2) * config.lateral + 0.01) & (
            offset[:, 2] >= config.rise_range[0] / 2
        )

    def rows(self, env, source):
        robot, pod, fixture = env.scene["robot"], env.scene["pod"], env.scene["fixture"]
        pod_pose = pod.data.root_link_pose_w.clone()
        pod_pose[:, :3] -= env.scene.env_origins
        result = {
            "joint_pos": robot.data.joint_pos.clone(),
            "joint_vel": robot.data.joint_vel.clone(),
            "pod_pose": pod_pose,
            "pod_vel": torch.cat((pod.data.root_link_lin_vel_w, pod.data.root_link_ang_vel_w), -1),
            "housing_pose": source["housing_pose"],
            "slide_pos": fixture.data.joint_pos.clone(),
            "slide_vel": fixture.data.joint_vel.clone(),
        }
        # The grasped source row's references and (unaccomplished) prefix carry over.
        for key in (
            "sequence_prefix",
            "initial_drawer_qpos",
            "initial_pod_height",
            "close_start_qpos",
        ):
            result[key] = source[key]
        return result


TASKS = {"square": SquarePregrasp, "drawer": DrawerPregrasp}


def _attempt(env, spec, source, generator, config, limits):
    """One open-rise-hold attempt per world; returns (accepted, rows)."""
    robot = spec.robot
    n = env.num_envs
    env.reset()
    spec.resets.write_states(env, source)
    site = env.sim.mj_model.site(f"robot/{robot.grasp_site}").id
    scale = env.cfg.actions[robot.action_name].position_scale
    slide0 = env.scene["fixture"].data.joint_pos[:, 0].clone() if spec.paired else None
    low, high = config.rise_range
    rise = low + (high - low) * torch.rand(n, device=env.device, generator=generator)
    side = (torch.rand(n, 2, device=env.device, generator=generator) * 2 - 1) * config.lateral
    jitter = torch.cat((side, rise[:, None]), -1)
    env.sim.forward()
    grip0 = env.sim.data.site_xpos[:, site].clone()
    obj0, quat0 = (value.clone() for value in pose(env, spec.object_name))
    local0 = quat_apply(quat_inverse(quat0), grip0 - obj0)
    action = torch.zeros(n, robot.action_dim, device=env.device)
    action[:, 6] = 1.0
    wrong = torch.zeros(n, dtype=torch.bool, device=env.device)
    touching = torch.zeros_like(wrong)
    steps = {
        phase: round(seconds / env.step_dt)
        for phase, seconds in (
            ("open", config.open_seconds),
            ("move", config.move_seconds),
            ("hold", config.hold_seconds),
        )
    }
    goal = grip0 + jitter
    for phase, count in steps.items():
        if phase == "move":
            obj, quat = pose(env, spec.object_name)
            goal = obj + quat_apply(quat, local0) + jitter
        for _ in range(count):
            command = action.clone()
            if phase == "move":
                grip = env.sim.data.site_xpos[:, site]
                clear = (grip[:, 2] - grip0[:, 2] > config.clearance)[:, None]
                target = torch.where(clear, goal, torch.cat((grip[:, :2], goal[:, 2:]), -1))
                command[:, :3] = ((target - grip) / scale).clamp(-1.0, 1.0)
            env.step(command)
            wrong |= spec.wrong(env)
            if phase == "hold":
                touching |= spec.touching(env)
    obj, quat = pose(env, spec.object_name)
    grip = env.sim.data.site_xpos[:, site]
    data = env.scene[spec.object_name].data
    axis = obj.new_tensor((0.0, 0.0, 1.0)).expand_as(obj)
    cosine = (quat_apply(quat, axis) * quat_apply(quat0, axis)).sum(-1)
    q = env.scene["robot"].data.joint_pos
    fingers = q[:, len(robot.arm_joints) :]
    accepted = (
        ~wrong
        & ~touching
        & (cosine >= math.cos(math.radians(config.max_tilt_degrees)))
        & (data.root_link_lin_vel_w.norm(dim=-1) <= config.max_linear_speed)
        & (data.root_link_ang_vel_w.norm(dim=-1) <= config.max_angular_speed)
        & (fingers.min(-1).values >= config.min_finger_opening)
        & spec.above(grip, obj, quat, config)
        & robot.reset_joint_position_mask(
            q, limits.to(q.device), robot.reset_finger_upper_tolerance
        )
    )
    if slide0 is not None:
        accepted &= (env.scene["fixture"].data.joint_pos[:, 0] - slide0).abs() <= 1e-3
    return accepted, spec.rows(env, source)


def append_pregrasp(
    bank, task, *, seed=0, num_envs=1024, device="cuda:0", cuda_graph_mode="auto", config=None
):
    """Return ``bank`` with the task's pre-grasp category appended (validated by the caller).

    Unpaired banks (Square) draw rows until the category matches the source count;
    paired banks (Drawer) derive one row per trajectory and drop every trajectory whose
    row fails all attempts from every category and the witness list.
    """
    config = config or PregraspConfig()
    spec = TASKS[task]()
    states = bank["states"]
    names = list(states)
    if spec.name in states:
        raise ValueError(f"{task} bank already has {spec.name}")
    source_name = names[spec.source_index]
    source_rows = states[source_name]
    count = len(source_rows["joint_pos"])
    cfg = spec.make_env_cfg()
    cfg.scene.num_envs = min(num_envs, count)
    cfg.events = {
        k: v for k, v in cfg.events.items() if v.mode != "startup" and k != "reset_from_bank"
    }
    cfg.rewards = {}
    spec.configure(cfg)
    cfg.episode_length_s = 1000.0
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    robot = spec.robot
    limits = torch.from_numpy(
        np.stack([env.sim.mj_model.joint(f"robot/{name}").range for name in robot.joints])
    ).float()
    # Paired: at most one derived row per source trajectory. Unpaired: each pass
    # draws fresh offsets for every source row until the category has ``count`` rows.
    derived = {key: [] for key in source_rows}
    origin = []
    done = torch.zeros(count, dtype=torch.bool)
    tested = passed = 0
    try:
        configure_cuda_graph(env, cuda_graph_mode)
        width = env.num_envs
        for _ in range(config.max_passes):
            candidates = (~done).nonzero().flatten() if spec.paired else torch.arange(count)
            if not len(candidates) or len(origin) >= count:
                break
            for start in range(0, len(candidates), width):
                if not spec.paired and len(origin) >= count:
                    break
                chunk = candidates[start : start + width]
                index = torch.cat((chunk, chunk[:1].expand(width - len(chunk))))
                source = {key: value[index].to(device) for key, value in source_rows.items()}
                accepted, rows = _attempt(env, spec, source, generator, config, limits)
                accepted = accepted[: len(chunk)].cpu()
                tested += len(chunk)
                passed += int(accepted.sum())
                for slot in accepted.nonzero().flatten().tolist():
                    done[chunk[slot]] = True
                    origin.append(int(chunk[slot]))
                    for key in derived:
                        derived[key].append(rows[key][slot].cpu())
    finally:
        env.close()
    if spec.paired:
        # Keep source-trajectory order so every category stays row-aligned.
        order = sorted(range(len(origin)), key=origin.__getitem__)
    else:
        if len(origin) < count:
            raise ValueError(f"{task}: only {len(origin)} of {count} pre-grasp rows accepted")
        order = list(range(count))
    if not order:
        raise ValueError(f"No {task} pre-grasp row was accepted")
    new_rows = {key: torch.stack([derived[key][i] for i in order]).float() for key in derived}
    result = deepcopy(bank)
    if spec.paired:
        for name in names:
            result["states"][name] = {key: value[done] for key, value in states[name].items()}
        witnesses = result["provenance"].get("physical_witnesses")
        if isinstance(witnesses, list):
            result["provenance"]["physical_witnesses"] = [
                witness for witness, kept in zip(witnesses, done.tolist(), strict=True) if kept
            ]
    result["states"][spec.name] = new_rows
    result["contract"]["categories"] = [*names, spec.name]
    projected = {"states": {spec.name: new_rows}, "provenance": {}}
    robot.project_reset_fingers(projected, limits, robot.reset_finger_upper_tolerance)
    result["provenance"]["pregrasp"] = {
        "category": spec.name,
        "source_category": source_name,
        "method": "open_gripper_reaim_rise_then_offset_hold_actual_end_state",
        "config": asdict(config),
        "seed": seed,
        "tested": tested,
        "passed": passed,
        "source_rows": count,
        "derived_rows": len(new_rows["joint_pos"]),
        "source_rows_used": len(set(origin[i] for i in order)),
        "paired_trajectories_dropped": int((~done).sum()) if spec.paired else 0,
        "domain_randomization": False,
        "finger_position_projection": projected["provenance"]["finger_position_projection"],
    }
    return result


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument(
        "--source", required=True, type=Path, help="Bank without the pre-grasp category"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-graph-mode", choices=("auto", "plain"), default="auto")
    args = parser.parse_args(argv)
    bank = torch.load(args.source, map_location="cpu", weights_only=True)
    result = append_pregrasp(
        bank,
        args.task,
        seed=args.seed,
        num_envs=args.num_envs,
        device=args.device,
        cuda_graph_mode=args.cuda_graph_mode,
    )
    result["provenance"]["pregrasp"]["source_bank"] = str(args.source.resolve())
    module = __import__(f"clab.tasks.{args.task}.reset_data", fromlist=["validate_bank"])
    module.validate_bank(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(result, stream)
    print(f"{args.task}: {result['provenance']['pregrasp']}", flush=True)
