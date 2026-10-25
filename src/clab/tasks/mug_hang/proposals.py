"""Fresh rim-grasp IK candidates; task3 starts OUTSIDE the branch tip.

No state is teleported into a threaded configuration. The collector must move a
held mug along the branch through OSC actions and physics before saving task3. Static
reachability witnesses do not prove transport paths for the other categories.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from scipy.spatial.transform import Rotation

from clab.robots.kinematics import SiteIK
from clab.robots.parallel_kinematics import CpuProposalPool
from clab.scenes.workspace import (
    XY_MAX,
    XY_MIN,
    body_geometry_corners,
    contains_bounds,
    workspace_metadata,
)

from .config import make_env_cfg
from .props import BRANCH_OUTER, HANDLE_ANCHOR, MUG_GRASP_POINT, MUG_INITIAL_Z, TREE_INITIAL_Z
from .reset_data import FIELDS, WIDTHS
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


@dataclass(frozen=True)
class ProposalConfig:
    parallel_chunk_size: int = 4
    x_range: tuple[float, float] = (XY_MIN[0], XY_MAX[0])
    y_range: tuple[float, float] = (XY_MIN[1], XY_MAX[1])
    tree_y_range: tuple[float, float] = (XY_MIN[1], XY_MAX[1])
    airborne_z: tuple[float, float] = (0.09, 0.23)
    max_penetration: float = 0.001
    pre_thread_clearance: float = 0.04
    final_anchor_x: tuple[float, float] = (0.04, 0.065)
    witness_roll_samples: int = 129


def _worker_context(sampler, seed_words, resting_poses):
    sampler.rng = np.random.default_rng(seed_words)
    if resting_poses is None:
        sampler.resting_poses = None
    else:
        sampler.set_resting_poses(resting_poses)


def _worker_batch(sampler, seed_words, resting_poses, kind, count, max_attempts):
    _worker_context(sampler, seed_words, resting_poses)
    return sampler._batch_arrays(kind, count, max_attempts)


def _worker_reachable(sampler, seed_words, resting_poses, mug_poses, tree_poses):
    _worker_context(sampler, seed_words, resting_poses)
    return np.array(
        [sampler.reachable(mug, tree) for mug, tree in zip(mug_poses, tree_poses, strict=True)],
        dtype=bool,
    )


def rotation(quat):
    value = np.empty(9)
    mujoco.mju_quat2Mat(value, quat)
    return value.reshape(3, 3)


def quaternion(matrix):
    value = np.empty(4)
    mujoco.mju_mat2Quat(value, np.asarray(matrix).ravel())
    return value


def yaw_quaternion(yaw):
    return np.array((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))


def rim_grasp_pose(mug_pose, roll=0.0, jaw_turn=0.0):
    """Source grasp point (not root/handle anchor), with free closure-axis roll."""
    mug_rot = rotation(mug_pose[3:])
    source_grasp = Rotation.from_euler("z", -math.pi / 2).as_matrix()
    source_grasp = source_grasp @ Rotation.from_euler("y", math.pi / 2).as_matrix()
    source_grasp = source_grasp @ Rotation.from_euler("x", math.pi / 2).as_matrix()
    ee_rot = (
        mug_rot
        @ source_grasp
        @ Rotation.from_euler("x", jaw_turn).as_matrix()
        @ Rotation.from_euler("y", roll).as_matrix()
    )
    return mug_pose[:3] + mug_rot @ MUG_GRASP_POINT, quaternion(ee_rot)


def threaded_mug_pose(tree_pose, anchor_x):
    """Aligned geometry used as a motion TARGET, never as a saved reset row."""
    local_anchor = np.array((anchor_x, 0.0, BRANCH_OUTER[2]))
    position = tree_pose[:3] + rotation(tree_pose[3:]) @ (local_anchor - HANDLE_ANCHOR)
    return np.concatenate((position, tree_pose[3:]))


class Sampler:
    def __init__(self, seed, cfg=None, *, workers=1):
        if workers < 1:
            raise ValueError("Proposal worker count must be positive")
        self.cfg = cfg or ProposalConfig()
        if self.cfg.parallel_chunk_size < 1:
            raise ValueError("parallel_chunk_size must be positive")
        if self.cfg.witness_roll_samples < 3 or self.cfg.witness_roll_samples % 2 != 1:
            raise ValueError("Witness roll grid requires an odd sample count >= 3")
        self.workers = workers
        self.pool = (
            CpuProposalPool(
                Sampler, factory_kwargs={"seed": 0, "cfg": self.cfg}, seed=seed, workers=workers
            )
            if workers > 1
            else None
        )
        self.rng = np.random.default_rng(seed)
        self.model = Scene(make_env_cfg().scene, device="cpu").compile()
        self.data = mujoco.MjData(self.model)
        self.ik = SiteIK(
            self.model,
            tuple(f"robot/{name}" for name in DEFAULT_ROBOT.arm_joints),
            f"robot/{DEFAULT_ROBOT.grasp_site}",
        )
        self.qadr = np.array(
            [self.model.joint(f"robot/{name}").qposadr[0] for name in DEFAULT_ROBOT.joints]
        )
        self.limits = np.stack(
            [self.model.joint(f"robot/{name}").range for name in DEFAULT_ROBOT.joints]
        )
        self.mug_qadr = int(self.model.joint("mug/mug_free").qposadr[0])
        self.mug = self.model.body("mug/mug").id
        self.tree = self.model.body("tree/tree").id
        self.table = self.model.body("table/table").id
        self.mocap = int(self.model.body("tree/tree").mocapid[0])
        self.fingers = tuple(
            self.model.body(f"robot/{name}").id for name in DEFAULT_ROBOT.finger_bodies
        )
        self.robot = {self.model.body(f"robot/{name}").id for name in DEFAULT_ROBOT.rigid_bodies}
        self.resting_poses = None
        self.last_batch_attempts = 0
        self.workspace_corners = body_geometry_corners(self.model, ("mug", "tree"))

    @property
    def metadata(self):
        return {
            **asdict(self.cfg),
            "workspace": workspace_metadata(),
            "grasp_point_mug_local": MUG_GRASP_POINT,
            "handle_anchor_mug_local": HANDLE_ANCHOR,
            "task3": "outside_branch_tip_then_real_controller_threading_path",
            "orientation": "rim_closing_axis_roll; airborne_Haar_SO3; thread_axis_aligned",
            "jaw_turn_branches": (0.0, math.pi),
            "grasp_contact_search": "full_open_free_then_first_contiguous_contact_bracket",
            "witness_roll_order": "deterministic_uniform_grid_sorted_by_absolute_roll",
            "witness_scope": "independent_static_grasp_existence_not_transport_path",
            "resting_pose_count": 0 if self.resting_poses is None else len(self.resting_poses),
            "proposal_workers": self.workers,
            "proposal_seed_schedule": "serial_generator"
            if self.pool is None
            else "seed_sequence_base_seed_and_ordered_chunk_index",
            "proposal_chunking": "fixed_size_ordered_chunks_dynamic_worker_assignment",
            "proposal_prefetch": "two_batches_bounded_ordered_results_except_postsettle_category",
            "proposal_budget_exhaustion": "return_partial_valid_rows_without_relaxing_predicates",
        }

    def state_dict(self):
        """Sampling continuation at a consumed batch boundary with no prefetch."""
        return {
            "config": asdict(self.cfg),
            "workers": self.workers,
            "rng": deepcopy(self.rng.bit_generator.state),
            "pool": None if self.pool is None else self.pool.seed_state(),
            "resting_poses": None
            if self.resting_poses is None
            else torch.from_numpy(self.resting_poses.copy()),
        }

    def load_state_dict(self, state):
        """Restore only before submitting new work; physical replay restarts per batch."""
        if state["config"] != asdict(self.cfg) or state["workers"] != self.workers:
            raise ValueError("MugHang sampler continuation configuration differs")
        if (state["pool"] is None) != (self.pool is None):
            raise ValueError("MugHang sampler continuation pool mode differs")
        if self.pool is not None:
            self.pool.restore_seed_state(state["pool"])
        self.rng.bit_generator.state = deepcopy(state["rng"])
        if state["resting_poses"] is not None:
            self.set_resting_poses(state["resting_poses"].numpy())

    def close(self):
        if self.pool is not None:
            self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def set_resting_poses(self, poses):
        """Copy this collection's physically accepted task0 pose pool unchanged."""
        values = np.asarray(poses, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 7
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Mug resting poses require finite [N,7]")
        if not np.allclose(np.linalg.norm(values[:, 3:], axis=-1), 1, atol=1e-5):
            raise ValueError("Mug resting quaternions must be normalized wxyz")
        self.resting_poses = values.copy()

    def forward(self, q, mug_pose, tree_pose):
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.mug_qadr : self.mug_qadr + 7] = mug_pose
        self.data.mocap_pos[self.mocap] = tree_pose[:3]
        self.data.mocap_quat[self.mocap] = tree_pose[3:]
        mujoco.mj_forward(self.model, self.data)

    def contacts(self):
        touching, safe = set(), True
        for contact in self.data.contact:
            pair = {int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
            if self.mug in pair and contact.dist <= 0:
                touching.update(pair - {self.mug})
            if (pair & self.robot or self.mug in pair) and contact.dist < -self.cfg.max_penetration:
                safe = False
            if self.mug in pair and (pair & (self.robot - set(self.fingers))):
                safe = False
            if self.tree in pair and pair & self.robot and contact.dist < 0:
                safe = False
        return touching, safe

    def grasp(self, mug_pose, tree_pose, roll):
        # The source yaw fold can exchange the two jaws by a tool-X half turn.
        # Both branches retain the same mug-local grasp point and closing line.
        for jaw_turn in (0.0, math.pi):
            result = self._grasp_branch(mug_pose, tree_pose, roll, jaw_turn)
            if result is not None:
                return result
        return None

    def _grasp_branch(self, mug_pose, tree_pose, roll, jaw_turn):
        point, quat = rim_grasp_pose(mug_pose, roll, jaw_turn)
        arm = None
        for attempt in range(3):
            seed = (
                np.array((0, 1.7, 1.3, -1, 0, 0))
                if attempt == 0
                else self.rng.uniform(self.limits[:6, 0], self.limits[:6, 1])
            )
            arm = self.ik.solve(point, quat, seed)
            if arm is not None:
                break
        if arm is None:
            return None
        q = np.concatenate((arm, (0.044, 0.044)))
        self.forward(q, mug_pose, tree_pose)
        if self.contacts()[0] & set(self.fingers):
            # A finger already inside the cup wall is not a free search bound.
            # In particular, do not use that bound for a fictitious large
            # centre shift: this roll is not a full-open rim approach.
            return None
        # Centre the two mechanically symmetric fingers at actual first contacts.
        apertures = []
        for index, finger in enumerate(self.fingers):
            q[-2:] = 0.044
            hi = 0.044
            # Geometric centering uses the physical joint interval, not the
            # binary controller's 2 mm target. The final symmetric grasp below
            # must still contact both sides at or above that control target.
            for lo in np.linspace(0.044, self.limits[6 + index, 0], 65)[1:]:
                q[6 + index] = lo
                self.forward(q, mug_pose, tree_pose)
                if finger in self.contacts()[0]:
                    break
                hi = lo
            else:
                return None
            # The cup is hollow: contact is not globally monotonic in opening.
            # Refine only the first contact bracket reached from a free opening.
            for _ in range(13):
                q[6 + index] = (lo + hi) / 2
                self.forward(q, mug_pose, tree_pose)
                if finger in self.contacts()[0]:
                    lo = q[6 + index]
                else:
                    hi = q[6 + index]
            apertures.append(lo)
        q[-2:] = 0.044
        self.forward(q, mug_pose, tree_pose)
        frame = self.data.site_xmat[self.ik.site_id].reshape(3, 3)
        point = (
            self.data.site_xpos[self.ik.site_id] + 0.5 * (apertures[0] - apertures[1]) * frame[:, 1]
        )
        centred = self.ik.solve(
            point, quaternion(frame), q[:6], position_tolerance=1e-5, angle_tolerance=1e-4
        )
        if centred is None:
            return None
        q[:6] = centred
        q[-2:] = 0.044
        self.forward(q, mug_pose, tree_pose)
        if self.contacts()[0] & set(self.fingers):
            return None
        lo, hi = 0.002, 0.044
        for _ in range(13):
            q[-2:] = (lo + hi) / 2
            self.forward(q, mug_pose, tree_pose)
            if set(self.fingers) <= self.contacts()[0]:
                lo = q[-1]
            else:
                hi = q[-1]
        q[-2:] = max(0.002, lo - 0.0001)
        self.forward(q, mug_pose, tree_pose)
        touching, safe = self.contacts()
        return q.copy() if safe and set(self.fingers) <= touching else None

    def reachable(self, mug_pose, tree_pose):
        # Existence witnesses must not accidentally inherit a sampled mug's
        # grasp, nor miss a narrow real rim grasp with just a few random rolls.
        rolls = np.linspace(-math.pi, math.pi, self.cfg.witness_roll_samples)
        rolls = rolls[np.argsort(np.abs(rolls), kind="stable")]
        return any(self.grasp(mug_pose, tree_pose, roll) is not None for roll in rolls)

    def settled_poses_reachable(self, mug_poses, tree_poses):
        """Check saved object poses in order; this does not verify transfer paths."""
        mug_poses, tree_poses = np.asarray(mug_poses), np.asarray(tree_poses)
        if mug_poses.ndim != 2 or mug_poses.shape[1] != 7 or tree_poses.shape != mug_poses.shape:
            raise ValueError("Mug and tree poses must both have shape [N,7]")
        if not np.isfinite(mug_poses).all() or not np.isfinite(tree_poses).all():
            raise ValueError("Post-settling poses must be finite")
        if not len(mug_poses):
            return np.empty(0, dtype=bool)
        if self.pool is None:
            return np.array(
                [self.reachable(mug, tree) for mug, tree in zip(mug_poses, tree_poses, strict=True)]
            )
        chunks = np.array_split(np.arange(len(mug_poses)), min(self.workers, len(mug_poses)))
        return np.concatenate(
            self.pool.map(
                _worker_reachable,
                [(self.resting_poses, mug_poses[chunk], tree_poses[chunk]) for chunk in chunks],
            )
        )

    def thread_path_reachable(self, q, mug_pose, tree_pose):
        """Discrete static approach witnesses; actual threading still needs physics."""
        self.forward(q, mug_pose, tree_pose)
        start_point = self.data.site_xpos[self.ik.site_id].copy()
        frame = self.data.site_xmat[self.ik.site_id].reshape(3, 3)
        target_quat = quaternion(frame)
        previous = q.copy()
        for anchor_x in np.linspace(
            BRANCH_OUTER[0] + self.cfg.pre_thread_clearance,
            self.cfg.final_anchor_x[0],
            10,
        )[1:]:
            target_mug = threaded_mug_pose(tree_pose, anchor_x)
            point = start_point + target_mug[:3] - mug_pose[:3]
            solution = self.ik.solve(point, target_quat, previous[:6])
            if solution is None:
                return False
            previous[:6] = solution
            self.forward(previous, target_mug, tree_pose)
            if not self.contacts()[1]:
                return False
        return True

    def candidate(self, kind):
        category = RESET_TYPES.index(kind)
        tree_pose = np.r_[
            self.rng.uniform(*self.cfg.x_range),
            self.rng.uniform(*self.cfg.tree_y_range),
            TREE_INITIAL_Z,
            yaw_quaternion(self.rng.uniform(-math.pi, math.pi)),
        ]
        if category == 3:
            mug_pose = threaded_mug_pose(tree_pose, BRANCH_OUTER[0] + self.cfg.pre_thread_clearance)
        elif category == 1:
            if self.resting_poses is None:
                raise ValueError(
                    "Task1 requires this collection's accepted task0 resting mug poses"
                )
            mug_pose = self.resting_poses[self.rng.integers(len(self.resting_poses))].copy()
        else:
            mug_pose = np.r_[
                self.rng.uniform(*self.cfg.x_range),
                self.rng.uniform(*self.cfg.y_range),
                MUG_INITIAL_Z,
                yaw_quaternion(self.rng.uniform(-math.pi, math.pi)),
            ]
            if category == 2:
                mug_pose[2] = self.rng.uniform(*self.cfg.airborne_z)
                quat = self.rng.normal(size=4)
                mug_pose[3:] = quat / np.linalg.norm(quat)
        if category != 3 and np.linalg.norm(mug_pose[:2] - tree_pose[:2]) < 0.14:
            return None
        approach = threaded_mug_pose(tree_pose, BRANCH_OUTER[0] + self.cfg.pre_thread_clearance)
        for name, value in (("mug", mug_pose), ("tree", tree_pose), ("mug", approach)):
            points = self.workspace_corners[name] @ rotation(value[3:]).T + value[:3]
            if not contains_bounds(points.min(0), points.max(0)):
                return None
        grasp = self.grasp(mug_pose, tree_pose, self.rng.uniform(-math.pi, math.pi))
        if grasp is None:
            return None
        if category == 3 and not self.thread_path_reachable(grasp, mug_pose, tree_pose):
            return None
        # A separate reachable branch approach, not a proof of full transfer.
        if category != 3 and not self.reachable(approach, tree_pose):
            return None
        if category == 0:
            for _ in range(80):
                q = np.r_[self.rng.uniform(self.limits[:6, 0], self.limits[:6, 1]), 0.044, 0.044]
                self.forward(q, mug_pose, tree_pose)
                touching, safe = self.contacts()
                if (
                    safe
                    and not touching & self.robot
                    and self.data.site_xpos[self.ik.site_id, 2] > 0.08
                ):
                    grasp = q
                    break
            else:
                return None
        return {
            "joint_pos": grasp,
            "joint_vel": np.zeros(8),
            "mug_pose": mug_pose,
            "mug_vel": np.zeros(6),
            "tree_pose": tree_pose,
        }

    def _batch_arrays(self, kind, count, max_attempts):
        if count < 1 or max_attempts < count:
            raise ValueError("count must be positive and max_attempts >= count")
        rows = []
        for _attempt in range(max_attempts):
            row = self.candidate(kind)
            if row is not None:
                rows.append(row)
            if len(rows) == count:
                break
        # Exhausting a bounded search is an expected rejection outcome, not a
        # worker failure. Preserve its valid rows; programming/physics errors
        # still propagate. Empty fields retain the ordinary transport contract.
        arrays = {
            key: np.stack([row[key] for row in rows])
            if rows
            else np.empty((0, width), dtype=np.float64)
            for key, width in zip(FIELDS, WIDTHS, strict=True)
        }
        return arrays, _attempt + 1

    def batch(self, kind, count, max_attempts):
        if count < 1 or max_attempts < count:
            raise ValueError("count must be positive and max_attempts >= count")
        if self.pool is None:
            arrays, attempts = self._batch_arrays(kind, count, max_attempts)
            results = [(arrays, attempts)]
        else:
            results = self.pool.map(_worker_batch, self._parallel_jobs(kind, count, max_attempts))
        return self._finish_batch(kind, count, results)

    def _parallel_jobs(self, kind, count, max_attempts):
        sizes = [
            min(self.cfg.parallel_chunk_size, count - offset)
            for offset in range(0, count, self.cfg.parallel_chunk_size)
        ]
        bounds = np.cumsum([0, *sizes]) * max_attempts // count
        return [
            (self.resting_poses, kind, size, int(bounds[i + 1] - bounds[i]))
            for i, size in enumerate(sizes)
        ]

    def _finish_batch(self, kind, count, results):
        arrays = {key: np.concatenate([rows[key] for rows, _ in results]) for key in results[0][0]}
        attempts = sum(attempts for _, attempts in results)
        accepted = len(arrays["joint_pos"])
        print(
            f"{kind}: IK/static accepted {accepted}/{attempts} proposals "
            f"(requested={count}, shortage={count - accepted})",
            flush=True,
        )
        self.last_batch_attempts = attempts
        return {key: torch.tensor(value, dtype=torch.float32) for key, value in arrays.items()}

    def iter_batches(self, kind, count, max_attempts, *, num_batches, prefetch=True):
        """Small dynamic jobs, with bounded lookahead while the GPU consumes rows.

        Disable cross-batch prefetch when post-settling CPU witnesses need this
        same pool. They would otherwise queue behind future proposal batches.
        Unused prefetched candidates are discarded on close, never saved.
        """
        if count < 1 or max_attempts < count or num_batches < 1:
            raise ValueError("Positive batch counts and max_attempts >= count required")
        if self.pool is None or not prefetch:
            for _ in range(num_batches):
                yield self.batch(kind, count, max_attempts)
            return
        batch_jobs = self._parallel_jobs(kind, count, max_attempts)
        jobs = (job for _ in range(num_batches) for job in batch_jobs)
        results = self.pool.imap(_worker_batch, jobs, max_pending=2 * len(batch_jobs))
        try:
            for _ in range(num_batches):
                yield self._finish_batch(kind, count, [next(results) for _ in batch_jobs])
        finally:
            results.close()
