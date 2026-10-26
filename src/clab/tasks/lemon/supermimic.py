"""Lemon SuperMimic surrogate: lemon resting on the plate, released.

Native success needs lemon-plate contact sampled across the control interval and
forbids a two-finger grasp. The reduced state has no contacts, so imagination
uses the native centre geometry, a resting-height band just above the plate
(``contact_margin`` over the native goal height) as the contact proxy, and both
fingers opened past ``release_opening`` as the release proxy. Like native success
it has no speed threshold. Native real rewards and evaluation remain authoritative.
"""

import torch

from clab.tasks.supermimic import RigidObjectAdapter, RigidObjectMDP
from clab.tensors import constant

from .settings import SUCCESS_MAX_HEIGHT, SUCCESS_MIN_HEIGHT, SUCCESS_XY_TOLERANCE


class LemonMDP(RigidObjectMDP):
    task = "lemon"
    object_name = "lemon"
    target_name = "plate"
    reward_mode = "imagined_endpoint_geometry_contact_height_release_shaped"
    # Narrow-axis half-width of the lemon; grasped reset rows hold 0.012-0.022 m.
    # Native success only forbids a two-finger grasp, which ends just past that.
    held_width = 0.0199
    release_opening = 0.023

    def __init__(self, *, goal_height, contact_margin=0.01, **kwargs):
        super().__init__(**kwargs)
        if not 0 < contact_margin < SUCCESS_MAX_HEIGHT:
            raise ValueError("contact_margin must be positive and below the success height")
        self.goal_height = float(goal_height)
        self.contact_margin = float(contact_margin)

    def _resting(self, s, c):
        """Native centre geometry plus the resting-height contact proxy."""
        delta = self.obj(s, "position") - self.target(c, "position")
        height = delta[:, 2]
        margins = torch.stack(
            (
                SUCCESS_XY_TOLERANCE - delta[:, :2].norm(dim=-1),
                height - SUCCESS_MIN_HEIGHT,
                SUCCESS_MAX_HEIGHT - height,
                self.goal_height + self.contact_margin - height,
            ),
            -1,
        )
        hard = ((margins[:, :3] > 0).all(-1) & (margins[:, 3] >= 0)).float()
        soft = torch.sigmoid(margins / self.smooth_temperature).prod(-1)
        return hard + (soft - soft.detach())

    def _candidate(self, s, c):
        return self._resting(s, c) * self._released(s)

    def reward(self, s, a, s1, c):
        lemon = self.obj(s1, "position")
        reaching = 1 - torch.tanh((self.physical_grasp(s1, c) - lemon).norm(dim=-1))
        goal = self.target(c, "position") + constant(s, (0.0, 0.0, self.goal_height))
        placement = torch.exp(-(lemon - goal).norm(dim=-1))
        # Imagination-only shaping, as in BoxInBin: once resting on the plate,
        # reaching stops paying and finger opening is rewarded instead.
        placed = self._resting(s1, c).detach()
        values = {
            "reaching": reaching * (1 - placed),
            "placement": placement,
            "success": self._success_term(s, s1, c),
            **self._costs(s, a, s1),
        }
        reward = sum(self.weights[name] * value for name, value in values.items())
        reward = (reward + self.release_weight * placed * self._release_progress(s1)) * self.dt
        done = 1 - (1 - values["success"]) * (1 - self._timeout(s1))
        return reward, done

    def task_contract(self):
        return {
            "imagined_success": "native centre XY and height thresholds, height at most goal_height + contact_margin above the plate (contact proxy), both fingers opened past release_opening; no contact, grasp or speed test",
            "imagined_release_shaping": "when resting: reaching gated off and release_weight * finger opening fraction added; imagination only",
            "goal_height": self.goal_height,
            "contact_margin": self.contact_margin,
        }


class LemonAdapter(RigidObjectAdapter):
    mdp_class = LemonMDP
    object_entity = "lemon"
    target_entity = "plate"
    object_geoms = ("lemon/lemon_collision_0", "lemon/lemon_collision_1")
    target_geoms = tuple(f"plate/plate_collision_{index}" for index in range(5))
    actor_terms = (
        "lemon_in_plate_frame",
        "previous_actions",
        "joint_pos",
        "ee_in_base_frame",
        "lemon_in_ee_frame",
        "plate_in_ee_frame",
    )
    reward_terms = frozenset(
        {"action_magnitude", "action_rate", "joint_vel", "reaching", "placement", "success"}
    )

    def __init__(self, env, **options):
        goal_height = env.cfg.rewards["placement"].params["goal_height"]
        super().__init__(env, goal_height=goal_height, **options)

    @staticmethod
    def robot_binding():
        from .variants import DEFAULT_ROBOT

        return DEFAULT_ROBOT

    def task_bodies(self):
        return ["lemon/lemon", "plate/plate"]
