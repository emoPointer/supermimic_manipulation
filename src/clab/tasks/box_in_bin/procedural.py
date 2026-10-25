"""Generate robot BoxInBin proposals by MuJoCo forward kinematics and contacts.

This older procedural sampler is independent of the reference bank.
Physics-hold acceptance happens separately in ``collect``. Candidate geometry
is world-frame; exported bank poses are robot-base-local, with wxyz quaternions.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.utils.lab_api.math import subtract_frame_transforms

from clab.robots.parallel_kinematics import CpuProposalPool
from clab.scenes.workspace import XY_MAX, XY_MIN, contains_bounds

from .config import make_env_cfg
from .settings import BIN_OUTER_HALF_EXTENTS, BOX_HALF_EXTENTS, RESET_TYPES
from .variants import DEFAULT_ROBOT


def _proposal_chunk(sampler, seed_words, kind, count, budget):
    """Use NumPy IPC, avoiding per-field Torch shared-memory mappings."""
    sampler.rng = np.random.default_rng(seed_words)
    return {key: value.numpy() for key, value in sampler.batch(kind, count, budget).items()}


class ProceduralSampler:
    """Sample reachable joint states and place the box at the grasp site."""

    # Proposal bands, not hidden robot limits: acceptance checks model limits.
    # The broad bands concentrate forward kinematics in this table workspace.
    _ARM_LO = np.array(DEFAULT_ROBOT.arm_proposal_low)
    _ARM_HI = np.array(DEFAULT_ROBOT.arm_proposal_high)

    def __init__(self, seed: int, *, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError("Positive proposal worker count required")
        self.workers = workers
        self._pool = (
            CpuProposalPool(
                ProceduralSampler, factory_kwargs={"seed": 0}, seed=seed, workers=workers
            )
            if workers > 1
            else None
        )
        self.rng = np.random.default_rng(seed)
        cfg = make_env_cfg()
        self.model = Scene(cfg.scene, device="cpu").compile()
        DEFAULT_ROBOT.validate_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, f"robot/{DEFAULT_ROBOT.grasp_site}")
        self.box_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "box/box_free")
        self.box_qadr = int(self.model.jnt_qposadr[self.box_joint])
        self.ee_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, f"robot/{DEFAULT_ROBOT.ee_body}")
        self.body_ids = {
            name: self._id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "box/box",
                *(f"robot/{name}" for name in DEFAULT_ROBOT.finger_bodies),
                "table/table",
                "bin/bin",
            )
        }
        self.robot_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (self.model.geom(geom_id).name or "").startswith("robot/")
        }
        initial_bin = cfg.scene.entities["bin"].init_state
        self.bin_pose = np.array((*initial_bin.pos, *initial_bin.rot))
        self.bin_rotation = np.empty(9)
        mujoco.mju_quat2Mat(self.bin_rotation, self.bin_pose[3:])
        self.bin_rotation = self.bin_rotation.reshape(3, 3)
        self.bin_mocap_id = int(self.model.body("bin/mocap_base").mocapid[0])
        self.root_body_id = self.model.body(f"robot/{DEFAULT_ROBOT.root_body}").id
        for index in range(len(DEFAULT_ROBOT.arm_joints)):
            lo, hi = self.model.jnt_range[index]
            if self._ARM_LO[index] < lo or self._ARM_HI[index] > hi:
                raise ValueError(f"Proposal band for joint{index + 1} exceeds MuJoCo limits")

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Current MuJoCo scene has no {kind.name} named {name}")
        return result

    def _box_contacts(self) -> set[int]:
        box_id = self.body_ids["box/box"]
        touching: set[int] = set()
        for contact in self.data.contact:
            first = int(self.model.geom_bodyid[contact.geom1])
            second = int(self.model.geom_bodyid[contact.geom2])
            if first == box_id:
                touching.add(second)
            elif second == box_id:
                touching.add(first)
        return touching

    def _robot_table_penetration_depth(self) -> float:
        """Deepest robot/table contact at the saved initial pose, in metres."""
        table_body = self.body_ids["table/table"]
        depth = 0.0
        for contact in self.data.contact:
            if (
                self.model.geom_bodyid[contact.geom1] == table_body
                and contact.geom2 in self.robot_geom_ids
            ) or (
                self.model.geom_bodyid[contact.geom2] == table_body
                and contact.geom1 in self.robot_geom_ids
            ):
                depth = max(depth, -float(contact.dist))
        return depth

    def _candidate(self, kind: str) -> dict[str, np.ndarray] | None:
        # A class-0 box must still lie in the robot's graspable tabletop
        # workspace. Draw its XY from a collision-checked resting grasp,
        # then sample an independent, open-hand robot configuration below.
        reachable = self._candidate(RESET_TYPES[1]) if kind == RESET_TYPES[0] else None
        if kind == RESET_TYPES[0] and reachable is None:
            return None
        if reachable is not None:
            self.bin_pose = reachable["bin_pose"].copy()
        else:
            yaw = self.rng.uniform(-math.pi, math.pi)
            self.bin_pose[3:] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            mujoco.mju_quat2Mat(self.bin_rotation.reshape(-1), self.bin_pose[3:])
            extent = np.abs(self.bin_rotation) @ np.asarray(BIN_OUTER_HALF_EXTENTS)
            self.bin_pose[:2] = self.rng.uniform(
                np.asarray(XY_MIN) + extent[:2], np.asarray(XY_MAX) - extent[:2]
            )
        mujoco.mju_quat2Mat(self.bin_rotation.reshape(-1), self.bin_pose[3:])
        self.data.mocap_pos[self.bin_mocap_id] = self.bin_pose[:3]
        self.data.mocap_quat[self.bin_mocap_id] = self.bin_pose[3:]
        q = np.zeros(len(DEFAULT_ROBOT.joints))
        q[: len(DEFAULT_ROBOT.arm_joints)] = self.rng.uniform(self._ARM_LO, self._ARM_HI)
        # A box centred at the grasp site first touches both convex finger
        # meshes near q7=q8=0.022 m. Starting at the old 0.002 m target put
        # the meshes about 20 mm inside the box before physics even stepped.
        q[len(DEFAULT_ROBOT.arm_joints) :] = 0.044 if kind == RESET_TYPES[0] else 0.0215
        self.data.qpos[: len(DEFAULT_ROBOT.joints)] = q
        mujoco.mj_forward(self.model, self.data)
        grasp = self.data.site_xpos[self.site_id].copy()
        box_quat = self.data.xquat[self.ee_body_id].copy()
        box_rot = self.data.xmat[self.ee_body_id].reshape(3, 3)
        half_height = np.abs(box_rot[2]) @ np.asarray(BOX_HALF_EXTENTS)
        if kind == RESET_TYPES[0]:
            assert reachable is not None
            yaw = self.rng.uniform(-math.pi, math.pi)
            box_pos = np.array((*reachable["box_pose"][:2], BOX_HALF_EXTENTS[2] - 0.0001))
            box_quat = np.array((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        else:
            box_pos = grasp.copy()
            if kind == RESET_TYPES[1]:
                if abs(grasp[2] - half_height) > 0.008:
                    return None
                box_pos[2] = half_height
            elif kind == RESET_TYPES[2]:
                if not 0.02 < grasp[2] - half_height < 0.24:
                    return None
            elif kind == RESET_TYPES[3]:
                delta = self.bin_rotation.T @ (grasp - self.bin_pose[:3])
                if not (abs(delta[0]) < 0.07 and abs(delta[1]) < 0.06):
                    return None
                if not 0.06 < grasp[2] < 0.14:
                    return None
            else:
                raise ValueError(f"Unknown reset type: {kind}")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, box_quat)
        extent = np.abs(rotation.reshape(3, 3)) @ np.asarray(BOX_HALF_EXTENTS)
        if not contains_bounds(box_pos[:2] - extent[:2], box_pos[:2] + extent[:2]):
            return None
        self.data.qpos[self.box_qadr : self.box_qadr + 3] = box_pos
        self.data.qpos[self.box_qadr + 3 : self.box_qadr + 7] = box_quat
        mujoco.mj_forward(self.model, self.data)
        if self._robot_table_penetration_depth() > 0.001:
            return None
        contacts = self._box_contacts()
        box_id = self.body_ids["box/box"]
        if any(
            contact.dist < -0.001
            and box_id
            in (
                self.model.geom_bodyid[contact.geom1],
                self.model.geom_bodyid[contact.geom2],
            )
            for contact in self.data.contact
        ):
            return None
        fingers = all(
            self.body_ids[f"robot/{name}"] in contacts for name in DEFAULT_ROBOT.finger_bodies
        )
        table = self.body_ids["table/table"] in contacts
        bin_contact = self.body_ids["bin/bin"] in contacts
        if kind == RESET_TYPES[0] and (fingers or bin_contact or not table):
            return None
        if kind == RESET_TYPES[1] and (not fingers or not table or bin_contact):
            return None
        if kind == RESET_TYPES[2] and (not fingers or table or bin_contact):
            return None
        if kind == RESET_TYPES[3] and (not fingers or not bin_contact or table):
            return None
        return {
            "joint_pos": q,
            "joint_vel": np.zeros(len(DEFAULT_ROBOT.joints)),
            "box_pose": np.concatenate((box_pos, box_quat)),
            "box_vel": np.zeros(6),
            "bin_pose": self.bin_pose.copy(),
        }

    def batch(self, kind: str, count: int, max_proposals: int) -> dict[str, torch.Tensor]:
        """Rejection-sample exactly ``count`` finite initial-contact candidates."""
        if count < 1 or max_proposals < count:
            raise ValueError("count must be positive and max_proposals must be at least count")
        if self._pool is not None:
            chunks = min(self.workers, count)
            sizes = [count // chunks + (index < count % chunks) for index in range(chunks)]
            budgets = [
                max_proposals // chunks + (index < max_proposals % chunks)
                for index in range(chunks)
            ]
            results = self._pool.map(
                _proposal_chunk,
                [(kind, size, budget) for size, budget in zip(sizes, budgets, strict=True)],
            )
            return {
                key: torch.from_numpy(np.concatenate([row[key] for row in results]))
                for key in results[0]
            }
        rows: list[dict[str, np.ndarray]] = []
        for _ in range(max_proposals):
            candidate = self._candidate(kind)
            if candidate is not None:
                rows.append(candidate)
                if len(rows) == count:
                    break
        if len(rows) != count:
            raise RuntimeError(
                f"Only {len(rows)}/{count} {kind} candidates in {max_proposals} proposals"
            )
        result = {
            field: torch.as_tensor(np.stack([row[field] for row in rows]), dtype=torch.float32)
            for field in rows[0]
        }
        # FK/contact checks use world coordinates; the bank contract is base-local.
        position = torch.as_tensor(self.data.xpos[self.root_body_id].copy()).to(result["box_pose"])
        rotation = torch.as_tensor(self.data.xquat[self.root_body_id].copy()).to(result["box_pose"])
        for field in ("box_pose", "bin_pose"):
            pose = result[field]
            result[field] = torch.cat(
                subtract_frame_transforms(
                    position.expand(count, 3), rotation.expand(count, 4), pose[:, :3], pose[:, 3:]
                ),
                dim=-1,
            )
        return result

    def close(self):
        if self._pool is not None:
            self._pool.close()
