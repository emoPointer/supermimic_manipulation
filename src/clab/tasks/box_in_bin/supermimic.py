"""BoxInBin SuperMimic surrogate: placement in the bin, release and rest.

Imagined success keeps the native placement geometry but uses looser speed
thresholds and an opened-finger release proxy; it has no contact or support-force
test. Native real rewards and evaluation remain authoritative.
"""

import torch

from clab.controllers.torch_kinematics import six_to_matrix
from clab.tasks.supermimic import RigidObjectAdapter, RigidObjectMDP

from .settings import BIN_INNER_HALF_EXTENTS, BOX_HALF_EXTENTS


class BoxInBinMDP(RigidObjectMDP):
    task = "box_in_bin"
    object_name = "box"
    target_name = "bin"
    reward_mode = "imagined_endpoint_geometry_velocity_release_shaped"
    held_width = BOX_HALF_EXTENTS[1]
    release_opening = 0.028

    def _placed(self, s, c):
        """Native placement geometry (box inside the bin, resting height)."""
        h = self.context.get
        box = self.obj(s, "position")
        matrix = six_to_matrix(self.obj(s, "rotation"))
        bin_pos = h(c, "bin_position")
        bin_matrix = six_to_matrix(h(c, "bin_rotation"))
        yaw = torch.atan2(bin_matrix[:, 1, 0], bin_matrix[:, 0, 0])
        cy, sy = yaw.cos(), yaw.sin()
        delta = box - bin_pos
        x = cy * delta[:, 0] + sy * delta[:, 1]
        y = -sy * delta[:, 0] + cy * delta[:, 1]
        half = s.new_tensor(BOX_HALF_EXTENTS)
        ext_x = ((cy[:, None] * matrix[:, 0] + sy[:, None] * matrix[:, 1]).abs() * half).sum(-1)
        ext_y = ((-sy[:, None] * matrix[:, 0] + cy[:, None] * matrix[:, 1]).abs() * half).sum(-1)
        ext_z = (matrix[:, 2].abs() * half).sum(-1)
        z = box[:, 2] + h(c, "origin_z").squeeze(-1)
        floor_error = box[:, 2] - ext_z - (bin_pos[:, 2] - 0.010)
        # Native lateral and height comparisons are strict; the hard branch is
        # computed separately to preserve equality boundaries exactly.
        margins = torch.stack(
            (
                BIN_INNER_HALF_EXTENTS[0] - x.abs() - ext_x,
                BIN_INNER_HALF_EXTENTS[1] - y.abs() - ext_y,
                0.005 - floor_error.abs(),
                z,
                0.100 - z,
            ),
            -1,
        )
        hard_geometry = (
            (margins[:, :2] > 0).all(-1) & (margins[:, 2] >= 0) & (margins[:, 3:] > 0).all(-1)
        ).float()
        soft_geometry = torch.sigmoid(margins / self.smooth_temperature).prod(-1)
        return hard_geometry + (soft_geometry - soft_geometry.detach())

    def _candidate(self, s, c):
        # No inferred contact or support force. Native success requires release;
        # without it imagination credited holding the box closed inside the bin.
        # Both fingers opened past release_opening approximate release; residual
        # finger contact remains possible. Real evaluation still checks both.
        # Speed thresholds are looser than native (0.01 m/s, 0.1 rad/s): the WM
        # predicts a resting box at ~0.02 m/s / 0.3 rad/s, never below native.
        return self._placed(s, c) * self._stable(s) * self._released(s)

    def reward(self, s, a, s1, c):
        box = self.obj(s1, "position")
        reaching = 1 - torch.tanh((self.physical_grasp(s1, c) - box).norm(dim=-1))
        goal = self.target(c, "position") + s.new_tensor((0.0, 0.0, 0.0405))
        placement = torch.exp(-(box - goal).norm(dim=-1))
        # Imagination-only shaping (not a native term): once the box is placed,
        # reward finger opening instead of grasp proximity, so holding stops
        # paying. The placement gate is detached; it only selects the term.
        placed = self._placed(s1, c).detach()
        release = placed * self._release_progress(s1)
        values = {
            "reaching": reaching * (1 - placed),
            "placement": placement,
            "success": self._success_term(s, s1, c),
            **self._costs(s, a, s1),
        }
        reward = sum(self.weights[name] * value for name, value in values.items())
        reward = (reward + self.release_weight * release) * self.dt
        success = values["success"]
        done = 1 - (1 - success) * (1 - self._timeout(s1))
        return reward, done

    def task_contract(self):
        return {
            "actor_history": "six terms of widths 6,7,8,6,6,6; each has five oldest-to-newest frames",
            "imagined_success": "native placement geometry, looser speed thresholds, both finger joints opened past release_opening; no contact or support-force condition",
            "imagined_release_shaping": "when placed: reaching gated off and release_weight * mean finger opening fraction from the held box half-width to release_opening added; imagination only",
            "success_counter_initialization": "native counter at replay start; subsequent imagined increments use the geometry/velocity surrogate",
        }


class BoxInBinAdapter(RigidObjectAdapter):
    mdp_class = BoxInBinMDP
    object_entity = "box"
    target_entity = "bin"
    object_geoms = ("box/box_collision",)
    target_geoms = tuple(
        f"bin/{name}"
        for name in (
            "floor",
            "wall_negative_y",
            "wall_negative_x",
            "wall_positive_y",
            "wall_positive_x",
        )
    )
    actor_terms = (
        "insertive_asset_in_receptive_asset_frame",
        "prev_actions",
        "joint_pos",
        "end_effector_pose",
        "insertive_asset_pose",
        "receptive_asset_pose",
    )
    reward_terms = frozenset(
        {"action_magnitude", "action_rate", "joint_vel", "reaching", "placement", "success"}
    )

    @staticmethod
    def robot_binding():
        from .variants import DEFAULT_ROBOT

        return DEFAULT_ROBOT

    def task_bodies(self):
        return ["box/box", "bin/bin"]
