"""Square-specific, contact-checked side-handle IK proposals in world coordinates.

This samples a declared reachable tabletop region, not the source expert's narrow
front/back rectangles. Task0 arm samples are independent of the nut grasp witness.
Task3 proposals are above the peg, NEVER inserted: the collector must physically
descend from the held hover to produce an accepted partially threaded reset.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene

from clab.robots.kinematics import SiteIK
from clab.robots.parallel_kinematics import CpuProposalPool
from clab.scenes.workspace import XY_MAX, XY_MIN, contains_bounds, workspace_metadata

from .config import make_env_cfg
from .props import (
    NUT_SPAWN_Z,
    PEG_ROOT_Z,
    handle_contact_bounds,
    workspace_corners,
)
from .settings import RESET_TYPES, AssemblyCriteria
from .variants import DEFAULT_ROBOT as robot


@dataclass(frozen=True)
class ProposalConfig:
    x_range: tuple[float, float] = (XY_MIN[0], XY_MAX[0])
    y_range: tuple[float, float] = (XY_MIN[1], XY_MAX[1])
    airborne_z: tuple[float, float] = (0.08, 0.25)
    peg_z: float = PEG_ROOT_Z
    hover_height: float = 0.181
    separation: float = 0.15
    max_penetration: float = 0.001
    # Collection geometry, deliberately separate from the policy reward target.
    # The old z=12 mm site was 2 mm above the 20 mm-thick handle. Moving the
    # finger-pad reference 8 mm deeper increases its vertical engagement while
    # retaining the existing whole-robot/table penetration rejection.
    grasp_point_nut_local: tuple[float, float, float] = (0.064, 0.0, 0.004)
    grasp_roll_range: tuple[float, float] = (-math.pi, math.pi)
    witness_attempts: int = 6


def yaw_quaternion(yaw):
    return np.array((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def multiply(a, b):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, a, b)
    return result


def rotate(quaternion, point):
    result = np.empty(3)
    mujoco.mju_rotVecQuat(result, np.asarray(point, dtype=float), quaternion)
    return result


def handle_grasp_pose(nut_pose, roll=0.0, turn=0.0, *, point=None):
    """Collection pad reference, with jaw closure on nut-local Y.

    Local-Y rotation changes the approach without changing the side-handle
    closure axis. The symmetric half-turn exchanges fingers, not grasp points.
    """
    downward = multiply(yaw_quaternion(turn), np.array((2**-0.5, 0, 2**-0.5, 0)))
    orientation = multiply(nut_pose[3:], downward)
    orientation = multiply(orientation, np.array((math.cos(roll / 2), 0, math.sin(roll / 2), 0)))
    if point is None:
        point = ProposalConfig().grasp_point_nut_local
    return nut_pose[:3] + rotate(nut_pose[3:], point), orientation


def _worker_context(sampler, seed_words, resting_poses):
    sampler.rng = np.random.default_rng(seed_words)
    if resting_poses is None:
        sampler.resting_poses = None
    else:
        sampler.set_resting_poses(resting_poses)


def _sample_chunk(sampler, seed_words, kind, count, max_attempts, resting_poses):
    _worker_context(sampler, seed_words, resting_poses)
    return sampler._batch_arrays(kind, count, max_attempts)


def _check_settled_chunk(sampler, seed_words, nuts, pegs, resting_poses):
    _worker_context(sampler, seed_words, resting_poses)
    return np.array(
        [sampler.settled_pose_reachable(nut, peg) for nut, peg in zip(nuts, pegs, strict=True)],
        dtype=bool,
    )


class Sampler:
    """CPU proposals and static reachability; physical acceptance is separate."""

    def __init__(self, seed=42, cfg=None, *, workers=1):
        if workers < 1:
            raise ValueError("Positive proposal worker count required")
        self.cfg = cfg or ProposalConfig()
        self.workers = workers
        self._pool = (
            CpuProposalPool(
                Sampler, factory_kwargs={"seed": 0, "cfg": self.cfg}, seed=seed, workers=workers
            )
            if workers > 1
            else None
        )
        self.proposal_attempts = dict.fromkeys(RESET_TYPES, 0)
        self.rng = np.random.default_rng(seed)
        self.model = Scene(make_env_cfg().scene, device="cpu").compile()
        self.data = mujoco.MjData(self.model)
        self.ik = SiteIK(
            self.model, tuple(f"robot/{j}" for j in robot.arm_joints), f"robot/{robot.grasp_site}"
        )
        self.qadr = np.array([self.model.joint(f"robot/{j}").qposadr[0] for j in robot.joints])
        self.limits = np.stack([self.model.joint(f"robot/{j}").range for j in robot.joints])
        self.nut_qadr = int(self.model.joint("nut/nut_free").qposadr[0])
        self.nut_body = self.model.body("nut/nut").id
        self.handle_geom = self.model.geom("nut/nut_collision_4").id
        self.peg_body = self.model.body("peg/peg").id
        self.table_body = self.model.body("table/table").id
        self.mocap_id = int(self.model.body("peg/peg").mocapid[0])
        self.fingers = {self.model.body(f"robot/{name}").id for name in robot.finger_bodies}
        self.robot_bodies = {self.model.body(f"robot/{name}").id for name in robot.rigid_bodies}
        self.seed_q = np.array((0.0, 1.7, 1.3, -1.0, 0.0, 0.0))
        self.resting_poses = None
        self.workspace_corners = workspace_corners()
        criteria = AssemblyCriteria()
        self.handle_bounds = handle_contact_bounds(
            criteria.handle_clearance, criteria.contact_tolerance
        )

    @property
    def metadata(self):
        return {
            **asdict(self.cfg),
            "workspace": workspace_metadata(),
            "grasp_point_nut_local": self.cfg.grasp_point_nut_local,
            "grasp_contact_geom": "nut_collision_4",
            "grasp_contact_region_nut_local": self.handle_bounds,
            "grasp_requires_both_fingers": True,
            "task0_arm": "independent_joint_sample_with_separate_grasp_and_goal_witness",
            "task3_proposal": "held_hover_above_peg_then_dynamic_descent",
            "target_witness": "independent_hover_IK_not_a_transfer_path_proof",
            "proposal_workers": self.workers,
            "proposal_seed_schedule": "serial_generator"
            if self.workers == 1
            else "seed_sequence_base_seed_and_ordered_chunk_index",
            "proposal_chunking": "balanced_contiguous_chunks_up_to_worker_count",
            "proposal_attempts": dict(self.proposal_attempts),
            "worker_device": "cpu",
        }

    def close(self):
        if self._pool is not None:
            self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def set_resting_poses(self, poses):
        values = np.asarray(poses, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 7
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Square resting poses must be nonempty finite [N,7]")
        if not np.allclose(np.linalg.norm(values[:, 3:], axis=-1), 1, atol=1e-3):
            raise ValueError("Square resting quaternions must be normalized")
        self.resting_poses = values.copy()

    def forward(self, q, nut, peg):
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.nut_qadr : self.nut_qadr + 7] = nut
        self.data.mocap_pos[self.mocap_id] = peg[:3]
        self.data.mocap_quat[self.mocap_id] = peg[3:]
        mujoco.mj_forward(self.model, self.data)

    def contacts(self, *, handle_only=False):
        """Return touched bodies plus safety of all robot/nut/peg contacts.

        Handle filtering changes only the contact witness, never the whole-nut
        penetration/arm/peg safety checks used by both grasp and free-arm paths.
        """
        touching, outside, safe = set(), set(), True
        lower, upper = self.handle_bounds
        for contact in self.data.contact:
            geoms = (contact.geom1, contact.geom2)
            pair = {int(self.model.geom_bodyid[g]) for g in geoms}
            if self.nut_body in pair and contact.dist <= 0:
                if handle_only:
                    local_point = (contact.pos - self.data.xpos[self.nut_body]) @ self.data.xmat[
                        self.nut_body
                    ].reshape(3, 3)
                    on_handle = (
                        self.handle_geom in geoms
                        and np.all(local_point >= lower)
                        and np.all(local_point <= upper)
                    )
                    if not on_handle:
                        outside.update(pair & self.fingers)
                    else:
                        touching.update(pair - {self.nut_body})
                else:
                    touching.update(pair - {self.nut_body})
            if (
                pair & self.robot_bodies or self.nut_body in pair
            ) and contact.dist < -self.cfg.max_penetration:
                safe = False
            if self.nut_body in pair and (pair & (self.robot_bodies - self.fingers)):
                safe = False
            if self.peg_body in pair and pair & self.robot_bodies and contact.dist < 0:
                safe = False
        return touching - outside, safe

    def grasp(self, nut, peg, attempts=None):
        """Both fingers must contact the actual handle without deep penetration."""
        for _ in range(attempts or self.cfg.witness_attempts):
            roll = self.rng.uniform(*self.cfg.grasp_roll_range)
            turn = float(self.rng.choice((0.0, math.pi)))
            point, quaternion = handle_grasp_pose(
                nut, roll, turn, point=self.cfg.grasp_point_nut_local
            )
            arm = self.ik.solve(point, quaternion, self.seed_q)
            if arm is None:
                continue
            q = np.concatenate((arm, (0.044, 0.044)))
            low, high = 0.002, 0.044
            for _ in range(14):
                q[-2:] = (low + high) / 2
                self.forward(q, nut, peg)
                if self.fingers <= self.contacts(handle_only=True)[0]:
                    low = q[-1]
                else:
                    high = q[-1]
            q[-2:] = max(0.002, low - 0.0001)
            self.forward(q, nut, peg)
            touching, safe = self.contacts(handle_only=True)
            if safe and self.fingers <= touching:
                return q.copy()
        return None

    def goal_reachable(self, nut, peg):
        hover = nut.copy()
        hover[:3] = (*peg[:2], self.cfg.hover_height)
        hover[3:] = yaw_quaternion(self.rng.uniform(-math.pi, math.pi))
        return self.grasp(hover, peg) is not None

    def settled_pose_reachable(self, nut, peg):
        return self.grasp(nut, peg) is not None and self.goal_reachable(nut, peg)

    def settled_poses_reachable(self, nuts, pegs):
        """Ordered saved-state IK recheck, using the same gate in each CPU worker."""
        nuts, pegs = np.asarray(nuts), np.asarray(pegs)
        if nuts.ndim != 2 or nuts.shape[1] != 7 or pegs.shape != nuts.shape:
            raise ValueError("Settled nut and peg poses must have matching [N,7] shapes")
        if not np.isfinite(nuts).all() or not np.isfinite(pegs).all():
            raise ValueError("Settled poses must be finite")
        if len(nuts) == 0:
            return np.empty(0, dtype=bool)
        if self._pool is None:
            return np.array(
                [
                    self.settled_pose_reachable(nut, peg)
                    for nut, peg in zip(nuts, pegs, strict=True)
                ],
                dtype=bool,
            )
        chunks = np.array_split(np.arange(len(nuts)), min(self.workers, len(nuts)))
        return np.concatenate(
            self._pool.map(
                _check_settled_chunk,
                [(nuts[chunk], pegs[chunk], self.resting_poses) for chunk in chunks],
            )
        )

    def free_arm(self, nut, peg):
        for _ in range(100):
            q = np.concatenate(
                (self.rng.uniform(self.limits[:6, 0], self.limits[:6, 1]), (0.044, 0.044))
            )
            self.forward(q, nut, peg)
            touching, safe = self.contacts()
            height = self.data.site_xpos[self.ik.site_id, 2]
            if safe and not touching & self.robot_bodies and 0.07 < height < 0.4:
                return q
        return None

    def xy(self):
        return np.array((self.rng.uniform(*self.cfg.x_range), self.rng.uniform(*self.cfg.y_range)))

    def candidate(self, kind):
        if kind not in RESET_TYPES:
            raise ValueError(f"Unknown Square reset type: {kind}")
        category = RESET_TYPES.index(kind)
        peg = np.concatenate(
            (self.xy(), (self.cfg.peg_z,), yaw_quaternion(self.rng.uniform(-math.pi, math.pi)))
        )
        nut = np.concatenate(
            (self.xy(), (NUT_SPAWN_Z,), yaw_quaternion(self.rng.uniform(-math.pi, math.pi)))
        )
        if category == 1:
            if self.resting_poses is None:
                raise ValueError(
                    "Task1 requires this collection's accepted, naturally settled task0 poses"
                )
            nut = self.resting_poses[self.rng.integers(len(self.resting_poses))].copy()
        elif category == 2:
            nut[2] = self.rng.uniform(*self.cfg.airborne_z)
            nut[3:] = self.rng.normal(size=4)
            nut[3:] /= np.linalg.norm(nut[3:])
        elif category == 3:
            nut[:3] = (*peg[:2], self.cfg.hover_height)
        if category != 3 and np.linalg.norm(nut[:2] - peg[:2]) < self.cfg.separation:
            return None
        for name, value in (("nut", nut), ("peg", peg)):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, value[3:])
            points = self.workspace_corners[name] @ rotation.reshape(3, 3).T + value[:3]
            if not contains_bounds(points.min(0), points.max(0)):
                return None
        witness = nut.copy()
        if category == 0:
            witness[2] = 0.0101
        q = self.grasp(witness, peg)
        if q is None or (category != 3 and not self.goal_reachable(nut, peg)):
            return None
        if category == 0:
            q = self.free_arm(nut, peg)
            if q is None:
                return None
        return {
            "joint_pos": q,
            "joint_vel": np.zeros(8),
            "nut_pose": nut,
            "nut_vel": np.zeros(6),
            "peg_pose": peg,
        }

    def _batch_arrays(self, kind, count, max_attempts):
        if count < 1 or max_attempts < count:
            raise ValueError("Positive candidate count and sufficient attempt budget required")
        rows = []
        for attempt in range(max_attempts):
            row = self.candidate(kind)
            if row is not None:
                rows.append(row)
            if len(rows) == count:
                return {
                    field: np.stack([row[field] for row in rows]) for field in rows[0]
                }, attempt + 1
        raise RuntimeError(
            f"Only {len(rows)}/{count} {kind} candidates after {max_attempts} attempts"
        )

    def batch(self, kind, count, max_attempts):
        """Keep serial RNG unchanged; parallel streams depend on declared chunking."""
        if count < 1 or max_attempts < count:
            raise ValueError("Positive candidate count and sufficient attempt budget required")
        if self._pool is None:
            arrays, attempts = self._batch_arrays(kind, count, max_attempts)
        else:
            sizes = [
                len(chunk) for chunk in np.array_split(np.arange(count), min(self.workers, count))
            ]
            results = self._pool.map(
                _sample_chunk,
                [
                    (kind, size, math.ceil(max_attempts * size / count), self.resting_poses)
                    for size in sizes
                ],
            )
            arrays = {
                field: np.concatenate([values[field] for values, _ in results])
                for field in results[0][0]
            }
            attempts = sum(result[1] for result in results)
        self.proposal_attempts[kind] += attempts
        return {field: torch.tensor(value, dtype=torch.float32) for field, value in arrays.items()}
