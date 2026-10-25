"""MugHang SuperMimic surrogate: mug hung on the bar by its handle, released.

Native success is geometric (tree-frame envelope, handle-on-bar engagement),
plus a translation-speed limit and no two-finger grasp. Imagination keeps the
geometry, loosens the speed limit and uses both fingers opened past
``release_opening`` as the release proxy. Native real rewards and evaluation
remain authoritative.
"""

import torch

from clab.controllers.torch_kinematics import quat_apply, quat_inverse, six_to_quat
from clab.tasks.supermimic import RigidObjectAdapter, RigidObjectMDP

from .props import (
    BRANCH_INNER,
    BRANCH_LENGTH,
    BRANCH_OUTER,
    HANDLE_ANCHOR,
    HANG_TARGET,
    MUG_GRASP_POINT,
)


def _apply_inverse(quaternion, vector):
    return quat_apply(quat_inverse(quaternion), vector)


class MugHangMDP(RigidObjectMDP):
    task = "mug_hang"
    object_name = "mug"
    target_name = "tree"
    actor_widths = (6, 7, 8, 6, 6, 6, 3)
    reward_mode = "imagined_endpoint_handle_geometry_speed_release_shaped"
    # Rim grasps hold each finger at 0.002-0.006 m; the wall is thin.
    held_width = 0.003
    release_opening = 0.015

    def _frames(self, s, c):
        mug = self.obj(s, "position")
        mug_quat = six_to_quat(self.obj(s, "rotation"))
        tree = self.target(c, "position")
        tree_quat = six_to_quat(self.target(c, "rotation"))
        return mug, mug_quat, tree, tree_quat

    def _point(self, position, quaternion, local):
        return position + quat_apply(quaternion, position.new_tensor(local).expand_as(position))

    def _handle_geometry(self, mug, mug_quat, tree, tree_quat):
        """Native finite-segment anchor distance, projection and handle alignment."""
        inner = _apply_inverse(mug_quat, self._point(tree, tree_quat, BRANCH_INNER) - mug)
        outer = _apply_inverse(mug_quat, self._point(tree, tree_quat, BRANCH_OUTER) - mug)
        direction = (outer - inner) / BRANCH_LENGTH
        anchor = mug.new_tensor(HANDLE_ANCHOR)
        projection = ((anchor - inner) * direction).sum(-1)
        closest = inner + projection.clamp(0, BRANCH_LENGTH)[:, None] * direction
        return (anchor - closest).norm(dim=-1), projection, direction[:, 0]

    def _hung(self, s, c):
        mug, mug_quat, tree, tree_quat = self._frames(s, c)
        relative = _apply_inverse(tree_quat, mug - tree)
        radial, projection, alignment = self._handle_geometry(mug, mug_quat, tree, tree_quat)
        strict = torch.stack(
            (0.085 - relative[:, 0].abs(), 0.05 - relative[:, 1].abs(), mug[:, 2] - 0.11), -1
        )
        closed = torch.stack(
            (
                0.018 - radial,
                projection + 0.005,
                BRANCH_LENGTH + 0.005 - projection,
                alignment - 0.75,
            ),
            -1,
        )
        hard = ((strict > 0).all(-1) & (closed >= 0).all(-1)).float()
        soft = torch.sigmoid(torch.cat((strict, closed), -1) / self.smooth_temperature).prod(-1)
        return hard + (soft - soft.detach())

    def _candidate(self, s, c):
        speed = self.obj(s, "velocity")[:, :3].norm(dim=-1)
        return self._hung(s, c) * self._gate(self.stable_linear_speed - speed) * self._released(s)

    def extra_observation(self, s, c):
        mug, mug_quat, tree, tree_quat = self._frames(s, c)
        handle = self._point(mug, mug_quat, HANDLE_ANCHOR)
        return (_apply_inverse(tree_quat, self._point(tree, tree_quat, HANG_TARGET) - handle),)

    def reward(self, s, a, s1, c):
        mug, mug_quat, tree, tree_quat = self._frames(s1, c)
        grasp_target = self._point(mug, mug_quat, MUG_GRASP_POINT)
        reaching = 1 - torch.tanh(10.0 * (self.physical_grasp(s1, c) - grasp_target).norm(dim=-1))
        _, _, alignment = self._handle_geometry(mug, mug_quat, tree, tree_quat)
        distance = (
            self._point(mug, mug_quat, HANDLE_ANCHOR) - self._point(tree, tree_quat, HANG_TARGET)
        ).norm(dim=-1)
        threading = torch.exp(-10.0 * distance) * alignment.clamp(0, 1)
        # Imagination-only shaping: once hung, reaching stops paying and finger
        # opening is rewarded instead.
        hung = self._hung(s1, c).detach()
        values = {
            "reaching": reaching * (1 - hung),
            "threading": threading,
            "success": self._success_term(s, s1, c),
            **self._costs(s, a, s1),
        }
        reward = sum(self.weights[name] * value for name, value in values.items())
        reward = (reward + self.release_weight * hung * self._release_progress(s1)) * self.dt
        done = 1 - (1 - values["success"]) * (1 - self._timeout(s1))
        return reward, done

    def task_contract(self):
        return {
            "imagined_success": "native tree-frame envelope and handle engagement, mug translation speed below stable_linear_speed (native 0.01 m/s), both fingers opened past release_opening; no contact test",
            "imagined_release_shaping": "when hung: reaching gated off and release_weight * finger opening fraction added; imagination only",
        }


class MugHangAdapter(RigidObjectAdapter):
    mdp_class = MugHangMDP
    object_entity = "mug"
    target_entity = "tree"
    object_geoms = tuple(f"mug/mug_collision_{index}" for index in range(32))
    target_geoms = (
        "tree/tree_base_collision",
        "tree/tree_post_collision",
        "tree/tree_branch_collision",
    )
    actor_terms = (
        "mug_in_tree_frame",
        "previous_actions",
        "joint_pos",
        "ee_in_base_frame",
        "mug_in_ee_frame",
        "tree_in_ee_frame",
        "handle_to_hang_target",
    )
    reward_terms = frozenset(
        {"action_magnitude", "action_rate", "joint_vel", "reaching", "threading", "success"}
    )

    @staticmethod
    def robot_binding():
        from .variants import DEFAULT_ROBOT

        return DEFAULT_ROBOT

    def task_bodies(self):
        return ["mug/mug", "tree/tree"]

    def native_candidate(self):
        # StableSuccess keeps only its counter; recompute the same predicate on
        # the captured POST state (terminations already ran, autoreset has not).
        from .mdp import success_candidate

        return success_candidate(self.env)
