"""Independent IK/contact-checked Lemon candidates in environment-world coordinates.

Targets are sampled in a declared reachable-table proposal rectangle. An IK
solution and static geometry check are required for the object and plate approach;
the collector subsequently tests dynamics. No demonstration or other task bank
is used. Candidate reachability does not prove a collision-free transfer path.
"""

from __future__ import annotations

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import Lock

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene

from clab.robots.kinematics import SiteIK
from clab.scenes.workspace import XY_MAX, XY_MIN, contains_bounds, workspace_metadata

from .config import make_env_cfg
from .props import LEMON_BOTTOM_OFFSET, PLATE_CENTRE_Z, workspace_corners
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT

_SPAWN_LOCK = Lock()
_WORKER_SAMPLER: Sampler | None = None


@contextmanager
def _cpu_worker_environment():
    """Set inherited limits before spawn imports NumPy, then restore the parent."""
    settings = dict.fromkeys(
        (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ),
        "1",
    )
    settings["CUDA_VISIBLE_DEVICES"] = ""
    with _SPAWN_LOCK:
        previous = {name: os.environ.get(name) for name in settings}
        os.environ.update(settings)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _initialize_worker(cfg: ProposalConfig) -> None:
    """One private CPU model per spawned process; no GPU state is inherited."""
    global _WORKER_SAMPLER
    torch.set_num_threads(1)
    _WORKER_SAMPLER = Sampler(seed=0, cfg=cfg)


def _worker_sampler(seed: list[int], resting_poses: np.ndarray | None) -> Sampler:
    if _WORKER_SAMPLER is None:
        raise RuntimeError("CPU proposal worker has not been initialized")
    sampler = _WORKER_SAMPLER
    sampler.rng = np.random.default_rng(seed)
    if resting_poses is None:
        sampler._resting_poses = None
    else:
        sampler.set_resting_poses(resting_poses)
    return sampler


def _sample_chunk(seed, resting_poses, kind, count, max_attempts):
    sampler = _worker_sampler(seed, resting_poses)
    return sampler._batch_arrays(kind, count, max_attempts)


def _check_settled_chunk(seed, resting_poses, lemon_poses, plate_poses):
    sampler = _worker_sampler(seed, resting_poses)
    return np.array(
        [
            sampler.settled_pose_reachable(lemon, plate)
            for lemon, plate in zip(lemon_poses, plate_poses, strict=True)
        ],
        dtype=bool,
    )


@dataclass(frozen=True)
class ProposalConfig:
    """Sampling bounds, distinct from the success region and joint limits."""

    x_range: tuple[float, float] = (XY_MIN[0], XY_MAX[0])
    y_range: tuple[float, float] = (XY_MIN[1], XY_MAX[1])
    airborne_z: tuple[float, float] = (0.08, 0.25)
    near_plate_height: tuple[float, float] = (0.035, 0.080)
    plate_z: float = PLATE_CENTRE_Z
    resting_z: float = -LEMON_BOTTOM_OFFSET
    separation: float = 0.11
    max_penetration: float = 0.001
    grasp_roll_range: tuple[float, float] = (-math.pi, math.pi)
    near_plate_orientation_angle: float = math.pi / 2
    grasp_witness_attempts: int = 8


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def downward_quaternion(yaw: float) -> np.ndarray:
    result = np.empty(4)
    mujoco.mju_mulQuat(result, yaw_quaternion(yaw), np.array((2**-0.5, 0.0, 2**-0.5, 0.0)))
    return result


def uniform_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Haar SO(3) proposal through a normalized isotropic Gaussian in R4."""
    value = rng.normal(size=4)
    return value / np.linalg.norm(value)


def narrow_grasp_quaternion(object_quat: np.ndarray, roll: float, turn: float = 0.0) -> np.ndarray:
    """Align jaw closure to object X while leaving approach rotation free.

    Right multiplication by local-Y rotation leaves the gripper's closure axis
    unchanged. ``turn=pi`` exchanges the two symmetric fingers.
    """
    narrow = np.empty(4)
    mujoco.mju_mulQuat(narrow, object_quat, downward_quaternion(math.pi / 2 + turn))
    result = np.empty(4)
    mujoco.mju_mulQuat(result, narrow, np.array((math.cos(roll / 2), 0.0, math.sin(roll / 2), 0.0)))
    return result


class Sampler:
    """Generate fresh proposals without conditioning task0 arm on its object pose."""

    def __init__(self, seed: int, cfg: ProposalConfig | None = None, *, workers: int = 1):
        if workers < 1:
            raise ValueError("Proposal workers must be positive")
        self.cfg = cfg or ProposalConfig()
        self.workers = workers
        self._seed = seed
        self._job_counter = 0
        self._pool: ProcessPoolExecutor | None = None
        self.rng = np.random.default_rng(seed)
        self.model = Scene(make_env_cfg().scene, device="cpu").compile()
        self.data = mujoco.MjData(self.model)
        self.workspace_corners = workspace_corners(self.model)
        self.arm_names = tuple(f"robot/{name}" for name in DEFAULT_ROBOT.arm_joints)
        self.ik = SiteIK(self.model, self.arm_names, f"robot/{DEFAULT_ROBOT.grasp_site}")
        self.qadr = np.array(
            [self.model.joint(f"robot/{name}").qposadr[0] for name in DEFAULT_ROBOT.joints]
        )
        self.limits = np.stack(
            [self.model.joint(f"robot/{name}").range for name in DEFAULT_ROBOT.joints]
        )
        self.lemon_qadr = int(self.model.joint("lemon/lemon_free").qposadr[0])
        self.lemon_body = self.model.body("lemon/lemon").id
        self.plate_body = self.model.body("plate/plate").id
        self.table_body = self.model.body("table/table").id
        self.plate_mocap = int(self.model.body("plate/plate").mocapid[0])
        self.finger_ids = tuple(
            self.model.body(f"robot/{name}").id for name in DEFAULT_ROBOT.finger_bodies
        )
        self.fingers = set(self.finger_ids)
        self.robot_bodies = {
            self.model.body(f"robot/{name}").id for name in DEFAULT_ROBOT.rigid_bodies
        }
        self.seed_q = np.array((0.0, 1.7, 1.3, -1.0, 0.0, 0.0))
        self._resting_poses: np.ndarray | None = None

    @property
    def metadata(self) -> dict:
        return {
            **asdict(self.cfg),
            "workspace": workspace_metadata(),
            "geometry_boundary": "rotated_body_aabb_of_all_visual_and_collision_mesh_vertices",
            "pose_sources": {
                RESET_TYPES[0]: "fresh_table_xy_and_yaw_then_natural_settling",
                RESET_TYPES[1]: "accepted_task0_full_resting_pose_with_uniform_grasp_roll",
                RESET_TYPES[2]: "haar_object_orientation_fresh_airborne_xyz_uniform_grasp_roll",
                RESET_TYPES[
                    3
                ]: "perturbed_task0_orientation_fresh_near_plate_xyz_uniform_grasp_roll",
            },
            "resting_pose_count": 0 if self._resting_poses is None else len(self._resting_poses),
            "goal_witness": "independent_static_grasp_at_plate_hover_not_same_grasp_transfer",
            "near_plate_orientation_perturbation": "uniform_axis_and_uniform_signed_angle",
            "proposal_workers": self.workers,
            "proposal_seed_schedule": (
                "serial_generator"
                if self.workers == 1
                else "seed_sequence_base_seed_and_ordered_chunk_index"
            ),
            "proposal_chunking": "balanced_contiguous_chunks_up_to_worker_count",
            "worker_device": "cpu",
            "worker_threads": 1 if self.workers > 1 else None,
        }

    def _submit(self, function, *args):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(self.cfg,),
            )
        seed = np.random.SeedSequence([self._seed, self._job_counter]).generate_state(4).tolist()
        self._job_counter += 1
        # ProcessPoolExecutor starts spawn workers during submit. Limit their
        # libraries before module imports, without leaving the GPU parent masked.
        with _cpu_worker_environment():
            return self._pool.submit(function, seed, self._resting_poses, *args)

    def close(self) -> None:
        """Release CPU workers; safe to call from the collector's finally block."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def set_resting_poses(self, poses: torch.Tensor | np.ndarray) -> None:
        """Copy this collection's accepted task0 poses for subsequent grasp classes.

        The caller must supply naturally settled, physically and IK-validated
        task0 states. This method checks representation, not physical validity.
        No yaw-only fallback is allowed if the required pose pool is missing.
        """
        if isinstance(poses, torch.Tensor):
            poses = poses.detach().cpu().numpy()
        values = np.asarray(poses, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or len(values) == 0:
            raise ValueError("Resting Lemon poses must be a non-empty [N, 7] xyz+wxyz array")
        if not np.isfinite(values).all():
            raise ValueError("Resting Lemon poses contain non-finite values")
        if not np.isclose(np.linalg.norm(values[:, 3:], axis=-1), 1.0).all():
            raise ValueError("Resting Lemon pose quaternions must be normalized wxyz")
        self._resting_poses = values.copy()

    def _xy(self) -> np.ndarray:
        return np.array((self.rng.uniform(*self.cfg.x_range), self.rng.uniform(*self.cfg.y_range)))

    def _solve(
        self, pos: np.ndarray, object_quat: np.ndarray, grasp_roll: float
    ) -> np.ndarray | None:
        # Grasp across the narrower lemon axis. The symmetric gripper also
        # permits a half-turn, important with joint6 restricted to +/-90 deg.
        for turn in (0.0, math.pi):
            target = narrow_grasp_quaternion(object_quat, grasp_roll, turn)
            for attempt in range(2):
                seed = (
                    self.seed_q
                    if attempt == 0
                    else self.rng.uniform(
                        (-0.7, 1.0, 0.4, -1.5, -0.7, -1.4), (0.7, 2.3, 2.6, 0.5, 0.7, 1.4)
                    )
                )
                arm = self.ik.solve(pos, target, seed)
                if arm is not None:
                    return np.concatenate((arm, (0.044, 0.044)))
        return None

    def _forward(self, q, lemon_pose, plate_pose) -> None:
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.lemon_qadr : self.lemon_qadr + 7] = lemon_pose
        self.data.mocap_pos[self.plate_mocap] = plate_pose[:3]
        self.data.mocap_quat[self.plate_mocap] = plate_pose[3:]
        mujoco.mj_forward(self.model, self.data)

    def _contacts(self) -> tuple[set[int], bool]:
        touching = set()
        safe = True
        for contact in self.data.contact:
            first, second = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            pair = {first, second}
            if self.lemon_body in pair and contact.dist <= 0:
                touching.update(pair - {self.lemon_body})
            # Deep finger/object overlap is also rejected, not hidden as a grasp.
            relevant = bool(pair & self.robot_bodies) or self.lemon_body in pair
            if relevant and contact.dist < -self.cfg.max_penetration:
                safe = False
            if self.lemon_body in pair and bool((pair - self.fingers) & self.robot_bodies):
                safe = False
            if self.plate_body in pair and bool(pair & self.robot_bodies) and contact.dist < 0:
                safe = False
        return touching, safe

    def _close_to_contact(self, q, lemon_pose, plate_pose) -> np.ndarray | None:
        # A scanned lemon is not symmetric about its body origin. Find each
        # finger's first contact with the other fully open, then centre the
        # grasp frame, keeping the object's sampled pose and both equal finger
        # positions intact. This avoids forcing one finger through the object
        # just to make the other touch.
        apertures = []
        for index, finger in enumerate(self.finger_ids):
            lo, hi = 0.002, 0.044
            q[-2:] = 0.044
            for _ in range(13):
                aperture = (lo + hi) / 2
                q[6 + index] = aperture
                self._forward(q, lemon_pose, plate_pose)
                if finger in self._contacts()[0]:
                    lo = aperture
                else:
                    hi = aperture
            apertures.append(lo)
        q[-2:] = 0.044
        self._forward(q, lemon_pose, plate_pose)
        rotation = self.data.site_xmat[self.ik.site_id].reshape(3, 3)
        target = self.data.site_xpos[self.ik.site_id].copy()
        target += 0.5 * (apertures[0] - apertures[1]) * rotation[:, 1]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        centred = self.ik.solve(
            target,
            quaternion,
            q[:6],
            position_tolerance=1e-5,
            angle_tolerance=1e-4,
        )
        if centred is None:
            return None
        q[:6] = centred
        # Find the widest symmetric aperture with both fingers touching; never
        # teleport closed fingers through the lemon's actual convex geometry.
        lo, hi = 0.002, 0.044
        for _ in range(13):
            aperture = (lo + hi) / 2
            q[-2:] = aperture
            self._forward(q, lemon_pose, plate_pose)
            touching, _ = self._contacts()
            if self.fingers <= touching:
                lo = aperture
            else:
                hi = aperture
        q[-2:] = max(0.002, lo - 0.0001)
        self._forward(q, lemon_pose, plate_pose)
        touching, safe = self._contacts()
        return q.copy() if safe and self.fingers <= touching else None

    def _grasp(self, lemon_pose, plate_pose, roll: float) -> np.ndarray | None:
        q = self._solve(lemon_pose[:3], lemon_pose[3:], roll)
        return None if q is None else self._close_to_contact(q, lemon_pose, plate_pose)

    def _grasp_witness(self, lemon_pose, plate_pose) -> np.ndarray | None:
        """Try multiple closure-axis rolls, not only an object-downward approach."""
        for roll in self.rng.uniform(*self.cfg.grasp_roll_range, self.cfg.grasp_witness_attempts):
            q = self._grasp(lemon_pose, plate_pose, float(roll))
            if q is not None:
                return q
        return None

    def _goal_witness(self, lemon_pose, plate_pose) -> bool:
        """Independent static plate-hover witness, not a same-grasp transfer proof.

        Do not force an airborne object's random orientation to be the eventual
        placing orientation. That requirement would bias accepted airborne states
        toward downward placement grasps again.
        """
        goal = lemon_pose.copy()
        goal[:3] = plate_pose[:3] + (0.0, 0.0, 0.060)
        if self._resting_poses is not None:
            goal[3:] = self._resting_poses[self.rng.integers(len(self._resting_poses)), 3:]
        return self._grasp_witness(goal, plate_pose) is not None

    def _free_arm(self, lemon_pose, plate_pose) -> np.ndarray | None:
        for _ in range(80):
            q = np.concatenate(
                (self.rng.uniform(self.limits[:6, 0], self.limits[:6, 1]), (0.044, 0.044))
            )
            self._forward(q, lemon_pose, plate_pose)
            point = self.data.site_xpos[self.ik.site_id]
            touching, safe = self._contacts()
            if safe and not touching & self.robot_bodies and 0.07 < point[2] < 0.4:
                return q
        return None

    def candidate(self, kind: str) -> dict[str, np.ndarray] | None:
        if kind not in RESET_TYPES:
            raise ValueError(f"Unknown Lemon reset category: {kind}")
        category = RESET_TYPES.index(kind)
        if category != 0 and self._resting_poses is None:
            raise ValueError(
                f"{kind} requires this collection's accepted task0 resting poses; "
                "call set_resting_poses before generating grasped categories"
            )
        plate_pose = np.concatenate(
            (self._xy(), (self.cfg.plate_z,), yaw_quaternion(self.rng.uniform(-math.pi, math.pi)))
        )
        if category == 0:
            yaw = self.rng.uniform(-math.pi, math.pi)
            lemon_pose = np.concatenate((self._xy(), (self.cfg.resting_z,), yaw_quaternion(yaw)))
        else:
            assert self._resting_poses is not None
            lemon_pose = self._resting_poses[self.rng.integers(len(self._resting_poses))].copy()
            if category == 2:
                lemon_pose[:2] = self._xy()
                lemon_pose[2] = self.rng.uniform(*self.cfg.airborne_z)
                lemon_pose[3:] = uniform_quaternion(self.rng)
            elif category == 3:
                lemon_pose[:2] = plate_pose[:2] + self.rng.uniform(-0.012, 0.012, 2)
                lemon_pose[2] = self.cfg.plate_z + self.rng.uniform(*self.cfg.near_plate_height)
                axis = self.rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = self.rng.uniform(
                    -self.cfg.near_plate_orientation_angle, self.cfg.near_plate_orientation_angle
                )
                perturbation = np.concatenate(((math.cos(angle / 2),), axis * math.sin(angle / 2)))
                orientation = np.empty(4)
                mujoco.mju_mulQuat(orientation, lemon_pose[3:], perturbation)
                lemon_pose[3:] = orientation
        if category != 3 and np.linalg.norm(lemon_pose[:2] - plate_pose[:2]) < self.cfg.separation:
            return None
        for name, value in (("lemon", lemon_pose), ("plate", plate_pose)):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, value[3:])
            points = self.workspace_corners[name] @ rotation.reshape(3, 3).T + value[:3]
            if not contains_bounds(points.min(0), points.max(0)):
                return None
        q = (
            self._grasp_witness(lemon_pose, plate_pose)
            if category == 0
            else self._grasp(lemon_pose, plate_pose, self.rng.uniform(*self.cfg.grasp_roll_range))
        )
        if q is None or not self._goal_witness(lemon_pose, plate_pose):
            return None
        if category == 0:
            q = self._free_arm(lemon_pose, plate_pose)
            if q is None:
                return None
        return {
            "joint_pos": q,
            "joint_vel": np.zeros(8),
            "lemon_pose": lemon_pose,
            "lemon_vel": np.zeros(6),
            "plate_pose": plate_pose,
        }

    def settled_pose_reachable(self, lemon_pose: np.ndarray, plate_pose: np.ndarray) -> bool:
        """Recheck narrow-side grasps after natural settling changed object pose.

        Multiple grasp rolls are tried at the actual settled object. The plate
        witness is independent: it is neither a same-grasp orientation proof nor
        a claim that an entire transfer path has been planned.
        """
        return self._grasp_witness(lemon_pose, plate_pose) is not None and self._goal_witness(
            lemon_pose, plate_pose
        )

    def settled_poses_reachable(
        self, lemon_poses: np.ndarray, plate_poses: np.ndarray
    ) -> np.ndarray:
        """Ordered post-settling witnesses, optionally checked by CPU workers.

        Reproducibility requires the same seed, worker count, batch sizes and
        ordered calls. Completion order never selects or reorders accepted rows.
        """
        lemon_poses = np.asarray(lemon_poses, dtype=np.float64)
        plate_poses = np.asarray(plate_poses, dtype=np.float64)
        if lemon_poses.ndim != 2 or lemon_poses.shape[1] != 7:
            raise ValueError("Settled Lemon poses must have shape [N, 7]")
        if plate_poses.shape != lemon_poses.shape:
            raise ValueError("Settled plate poses must match the Lemon pose shape")
        if not np.isfinite(lemon_poses).all() or not np.isfinite(plate_poses).all():
            raise ValueError("Settled poses contain non-finite values")
        if len(lemon_poses) == 0:
            return np.empty(0, dtype=bool)
        if self.workers == 1:
            return np.array(
                [
                    self.settled_pose_reachable(lemon, plate)
                    for lemon, plate in zip(lemon_poses, plate_poses, strict=True)
                ],
                dtype=bool,
            )
        chunks = np.array_split(np.arange(len(lemon_poses)), min(self.workers, len(lemon_poses)))
        futures = [
            self._submit(_check_settled_chunk, lemon_poses[indices], plate_poses[indices])
            for indices in chunks
        ]
        return np.concatenate([future.result() for future in futures])

    def _batch_arrays(self, kind: str, count: int, max_attempts: int):
        if count < 1 or max_attempts < count:
            raise ValueError("count must be positive and max_attempts >= count")
        rows = []
        for attempt in range(max_attempts):
            row = self.candidate(kind)
            if row is not None:
                rows.append(row)
            if len(rows) == count:
                return (
                    {key: np.stack([row[key] for row in rows]) for key in rows[0]},
                    attempt + 1,
                )
        raise RuntimeError(f"Only {len(rows)}/{count} {kind} candidates in {max_attempts} attempts")

    def batch(self, kind: str, count: int, max_attempts: int) -> dict[str, torch.Tensor]:
        if count < 1 or max_attempts < count:
            raise ValueError("count must be positive and max_attempts >= count")
        if self.workers == 1:
            arrays, attempts = self._batch_arrays(kind, count, max_attempts)
        else:
            # Preserve the exact overall proposal budget while assigning every
            # chunk at least one attempt per requested accepted candidate.
            sizes = [
                len(part) for part in np.array_split(np.arange(count), min(self.workers, count))
            ]
            boundaries = np.cumsum([0, *sizes]) * max_attempts // count
            futures = [
                self._submit(_sample_chunk, kind, size, int(boundaries[i + 1] - boundaries[i]))
                for i, size in enumerate(sizes)
            ]
            results = [future.result() for future in futures]
            arrays = {
                key: np.concatenate([rows[key] for rows, _ in results]) for key in results[0][0]
            }
            attempts = sum(attempts for _, attempts in results)
        print(f"{kind}: IK/static accepted {count}/{attempts} proposals", flush=True)
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in arrays.items()}
