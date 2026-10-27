"""Square SuperMimic surrogate: nut threaded, seated, released and at rest.

Native success needs a legal exposed-handle grasp observed earlier, physical
seating with base support and no penetration, low speed and full release; a
two-finger ring or mixed grasp ends the episode as invalid. Imagination carries
the native ``grasp_seen`` and ``invalid`` flags as analytic state and replaces
the contact predicates by grasp-point geometry in the nut frame plus a closed-
on-object finger band, and the release by the fingers' collision boxes clearing
the nut's. Seating keeps the native threaded/height/tilt geometry; support force
and penetration are not modelled. Native real rewards and evaluation remain
authoritative.
"""

import math

import torch

from clab.controllers.torch_kinematics import (
    quat_apply,
    quat_inverse,
    quat_mul,
    quat_to_matrix,
    six_to_quat,
)
from clab.robots.tool_geometry import finger_boxes
from clab.tasks.supermimic import RigidObjectAdapter, RigidObjectMDP
from clab.tensors import constant

from .mdp import threaded_geometry
from .props import NUT_BOXES, NUT_GRASP_POINT, NUT_HALF_HEIGHT, NUT_SEATED_OFFSET, PEG_HALF_HEIGHT
from .settings import AssemblyCriteria

_HANDLE_CENTRE, _HANDLE_HALF = NUT_BOXES[4]
_RING_OUTER = max(position[0] + size[0] for position, size in NUT_BOXES[:4])
# Peg-local nut centre held above the peg top, mid-range of the collector's
# above-peg clearance (0.015-0.040 m).
_ABOVE_PEG = PEG_HALF_HEIGHT + NUT_HALF_HEIGHT + 0.025
# Finger collision boxes: the pad within 2 cm behind the grasp point (the tips reach
# 1.3 cm past it) and the rest of the finger. One box for the whole finger was so
# loose (its base is wide) that it rarely read a natively released seated nut as
# released; the split recovers most of them, and finer bands add little.
_FINGER_DEPTHS = (0.02, 0.2)


def _apply_inverse(quaternion, vector):
    return quat_apply(quat_inverse(quaternion), vector)


class SquareMDP(RigidObjectMDP):
    task = "square"
    object_name = "nut"
    target_name = "peg"
    reward_mode = "imagined_endpoint_grasp_flags_seating_release"
    process_fields = ("grasp_seen", "invalid")
    # Handle grasps hold each finger at 0.011-0.018 m (grasped reset rows).
    held_width = 0.0148
    release_opening = 0.024

    def __init__(
        self,
        *,
        criteria,
        reaching_scale,
        far_reaching_scale=0.5,
        alignment_scale,
        tilt_scale,
        insertion_scale,
        grip_band=(0.008, 0.020),
        finger_boxes,
        release_clearance=0.001,
        carry_weight=0.2,
        early_release_penalty=2.0,
        release_radius=0.01,
        peg_hold_cost=2.0,
        action_bound_weight=0.05,
        **kwargs,
    ):
        super().__init__(action_bound_weight=action_bound_weight, **kwargs)
        for name, value in (
            ("carry_weight", carry_weight),
            ("early_release_penalty", early_release_penalty),
            ("release_radius", release_radius),
            ("peg_hold_cost", peg_hold_cost),
            ("release_clearance", release_clearance),
        ):
            if not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        self.carry_weight = float(carry_weight)
        self.early_release_penalty = float(early_release_penalty)
        self.release_radius = float(release_radius)
        self.peg_hold_cost = float(peg_hold_cost)
        self.release_clearance = float(release_clearance)
        self.criteria = criteria
        self.reaching_scale = float(reaching_scale)
        self.far_reaching_scale = float(far_reaching_scale)
        if not 0 < self.far_reaching_scale < float("inf"):
            raise ValueError("far_reaching_scale must be finite and positive")
        self.alignment_scale, self.tilt_scale = float(alignment_scale), float(tilt_scale)
        self.insertion_scale = float(insertion_scale)
        self.grip_band = tuple(float(value) for value in grip_band)
        if not 0 < self.grip_band[0] < self.grip_band[1] < self.release_opening:
            raise ValueError("grip_band must be ordered and below release_opening")
        self.finger_boxes = tuple(
            tuple(tuple(tuple(float(v) for v in part) for part in box) for box in bands)
            for bands in finger_boxes
        )
        boxes = [(finger, box) for finger, bands in enumerate(self.finger_boxes) for box in bands]
        if (
            len(self.finger_boxes) != 2
            or not all(self.finger_boxes)
            or any(
                len(box) != 3
                or any(len(part) != 3 or not all(math.isfinite(v) for v in part) for part in box)
                for _, box in boxes
            )
        ):
            raise ValueError(
                "finger_boxes must hold finite (lower, upper, direction) boxes per finger"
            )
        if any(
            lo >= hi for _, (lower, upper, _) in boxes for lo, hi in zip(lower, upper, strict=True)
        ):
            raise ValueError("finger boxes must have positive extent")
        self._box_finger = tuple(finger for finger, _ in boxes)
        self._box_centre = tuple(
            tuple((lo + hi) / 2 for lo, hi in zip(lower, upper, strict=True))
            for _, (lower, upper, _) in boxes
        )
        self._box_half = tuple(
            tuple((hi - lo) / 2 for lo, hi in zip(lower, upper, strict=True))
            for _, (lower, upper, _) in boxes
        )
        self._box_direction = tuple(direction for _, (_, _, direction) in boxes)

    def _frames(self, s, c):
        return (
            self.obj(s, "position"),
            six_to_quat(self.obj(s, "rotation")),
            self.target(c, "position"),
            six_to_quat(self.target(c, "rotation")),
        )

    def _grasp_local(self, s, c):
        nut, nut_quat, _, _ = self._frames(s, c)
        return _apply_inverse(nut_quat, self.physical_grasp(s, c) - nut)

    def _gripping(self, s):
        """Both fingers closed onto an object: inside the held band, not fully shut.

        Straight-through (hard forward, sigmoid band edges backward), so the rewards
        that need a grasp show the actor that opening the gripper loses them.
        """
        fingers = self.state.get(s, "robot_q")[:, 6:8]
        low, high = self.grip_band
        return self._band(fingers, low, high, self.release_temperature).prod(-1)

    def _open(self, s):
        """Both fingers opened past grip_band, ready to close around the handle."""
        fingers = self.state.get(s, "robot_q")[:, 6:8]
        return self._gate(fingers - self.grip_band[1], self.release_temperature).prod(-1)

    def _finger_gap(self, s, c):
        """Smallest separation [N] between a finger collision box and a nut box.

        Each finger box (pad and rest of finger), posed in the nut frame by the arm
        joints and its finger's opening, is bounded by an axis-aligned box; a pair's
        separation is its largest per-axis gap (<= 0 where the bounds overlap). Wrist yaw only enlarges the
        bound, so a skewed hand reads as closer, never farther.
        """
        nut, nut_quat, _, _ = self._frames(s, c)
        grasp, tool_quat = self.physical_tool(s, c)
        rotation = quat_to_matrix(quat_mul(quat_inverse(nut_quat), tool_quat))
        local = _apply_inverse(nut_quat, grasp - nut)
        opening = self.state.get(s, "robot_q")[:, 6:8][:, list(self._box_finger)]
        tool_centre = s.new_tensor(self._box_centre) + opening[..., None] * s.new_tensor(
            self._box_direction
        )
        centre = local[:, None] + torch.einsum("nij,nkj->nki", rotation, tool_centre)
        half = torch.einsum("nij,kj->nki", rotation.abs(), s.new_tensor(self._box_half))
        boxes = s.new_tensor(NUT_BOXES)  # [box, (centre, half), 3]
        gap = (centre[:, :, None] - boxes[:, 0]).abs() - (half[:, :, None] + boxes[:, 1])
        return gap.amax(-1).amin(-1).amin(-1)

    def _hands_off(self, s, c):
        """Both fingers clear of the nut by release_clearance, the proxy for no nut-robot contact.

        Native release is the end of every nut-robot contact. Its previous proxy, not
        both fingers inside grip_band, counted an opened hand as released, and most
        stalled seated nuts still had one open finger pressing on the ring (about
        1.5 N), so imagination counted successes the native test never did.
        Straight-through on the smallest finger-nut separation.
        """
        return self._gate(self._finger_gap(s, c) - self.release_clearance, self.release_temperature)

    def grasp_status(self, s, c):
        """Proxies for native (legal handle grasp, wrong ring or mixed grasp).

        The legal grasp is straight-through in the finger band; the wrong grasp is hard.
        """
        local = self._grasp_local(s, c)
        gripping = self._gripping(s)
        near_slab = local[:, 2].abs() <= 2 * NUT_BOXES[0][1][2]
        handle = (
            (local[:, 0] >= _RING_OUTER + self.criteria.handle_clearance)
            & (local[:, 0] <= _HANDLE_CENTRE[0] + _HANDLE_HALF[0])
            & (local[:, 1].abs() <= _HANDLE_HALF[1] + self.criteria.contact_tolerance)
        )
        ring = (local[:, 0] < _RING_OUTER) & (local[:, 1].abs() <= _RING_OUTER)
        legal = gripping * (near_slab & handle).to(gripping.dtype)
        return legal, (gripping.detach() > 0.5) & near_slab & ring

    def _tilt(self, s, c):
        _, nut_quat, _, peg_quat = self._frames(s, c)
        normal = constant(s, (0.0, 0.0, 1.0)).expand(len(s), -1)
        return _apply_inverse(peg_quat, quat_apply(nut_quat, normal))[:, 2].abs().clamp(0, 1)

    def _seated(self, s, c):
        """Native threaded geometry (hard) with differentiable height and tilt margins."""
        nut, nut_quat, peg, peg_quat = self._frames(s, c)
        threaded = threaded_geometry(
            nut, nut_quat, peg, peg_quat, axial_tolerance=self.criteria.max_penetration
        ).to(s.dtype)
        relative = _apply_inverse(peg_quat, nut - peg)
        margins = torch.stack(
            (
                self.criteria.seated_height_tolerance - (relative[:, 2] - NUT_SEATED_OFFSET).abs(),
                self._tilt(s, c) - math.cos(math.radians(self.criteria.max_tilt_degrees)),
            ),
            -1,
        )
        hard = (margins >= 0).all(-1).to(s.dtype)
        soft = torch.sigmoid(margins / self.smooth_temperature).prod(-1)
        return threaded * (hard + (soft - soft.detach()))

    def _flag(self, s, name):
        return self.state.get(s, name).squeeze(-1)

    def _candidate(self, s, c):
        valid = self._flag(s, "grasp_seen") * (1 - self._flag(s, "invalid"))
        return self._seated(s, c) * self._stable(s) * self._hands_off(s, c) * valid

    def advance_task_fields(self, fields, s, c):
        provisional = self.state.pack(fields)
        grasped, wrong = self.grasp_status(provisional, c)
        invalid = torch.maximum(self.state.get(s, "invalid"), wrong.to(s.dtype)[:, None])
        legal = grasped.detach()[:, None] * (1 - invalid)
        fields["invalid"] = invalid
        fields["grasp_seen"] = torch.maximum(self.state.get(s, "grasp_seen"), legal)
        super().advance_task_fields(fields, s, c)

    def reward(self, s, a, s1, c):
        nut, nut_quat, peg, peg_quat = self._frames(s1, c)
        grasped, _ = self.grasp_status(s1, c)
        grasped = grasped.to(s1.dtype)
        invalid, grasp_seen = self._flag(s1, "invalid"), self._flag(s1, "grasp_seen")
        valid = 1 - invalid
        seated = self._seated(s1, c).detach()
        handle = nut + quat_apply(nut_quat, constant(nut, NUT_GRASP_POINT).expand_as(nut))
        # Imagination-only: the native reaching term stops while the handle is
        # grasped, so an open gripper hovering at the handle (0.2) paid more than
        # holding it (handle_grasp 0.1) and the policy released at once. Keep
        # paying reaching while grasped; it still stops once the nut is seated.
        approach = valid * (1 - seated * grasp_seen)
        # Imagination-only: reaching pays only an open hand or a legal grasp. The
        # full-task policy shut its gripper at once and a closed hand beside the
        # handle (which cannot grasp it) was paid like an open one.
        ready = torch.maximum(self._open(s1), grasped)
        distance = (self.physical_grasp(s1, c) - handle).norm(dim=-1)
        relative = _apply_inverse(peg_quat, nut - peg)
        # acos has an infinite derivative at 1 (a level nut); stay just inside.
        tilt = torch.acos(self._tilt(s1, c).clamp_max(1 - 1e-6))
        alignment = torch.exp(
            -(relative[:, :2].norm(dim=-1) / self.alignment_scale).square()
            - (tilt / self.tilt_scale).square()
        )
        seated_target = peg + quat_apply(
            peg_quat, constant(peg, (0.0, 0.0, NUT_SEATED_OFFSET)).expand_as(peg)
        )
        threaded = threaded_geometry(
            nut, nut_quat, peg, peg_quat, axial_tolerance=self.criteria.max_penetration
        ).to(s1.dtype)
        insertion = torch.exp(-((nut - seated_target).norm(dim=-1) / self.insertion_scale).square())
        # Imagination-only shaping. The native alignment and insertion terms are
        # 3 cm Gaussians, flat at the ~12 cm start of a carry, so pull the nut
        # toward the nearer of the above-peg and seated poses (descending onto
        # the peg never lowers it) while it is held by the handle or threaded,
        # not when shoved across the table.
        above_target = peg + quat_apply(
            peg_quat, constant(peg, (0.0, 0.0, _ABOVE_PEG)).expand_as(peg)
        )
        carry = torch.exp(
            -torch.minimum((nut - above_target).norm(dim=-1), (nut - seated_target).norm(dim=-1))
        )
        candidate = self._candidate(s1, c)
        values = {
            # Imagination-only far term: the native 5 cm tanh is flat (derivative
            # ~1e-9) at the ~0.5 m full-task start, so nothing pulled the hand in.
            "reaching": (
                2
                - torch.tanh(distance / self.reaching_scale)
                - torch.tanh(distance / self.far_reaching_scale)
            )
            * approach
            * ready,
            # Imagination-only: holding stops paying once the nut is on the peg
            # (natively once seated). Pushing a threaded nut down by its handle
            # tilts and jams it; released, it slides down and seats.
            "handle_grasp": grasped * valid * (1 - threaded),
            "alignment": alignment * grasped * valid * (1 - threaded),
            "insertion": insertion * threaded * grasp_seen * valid,
            "release": candidate,
            "invalid_grasp": invalid,
            "success": self._success_term(s, s1, c),
            **self._costs(s, a, s1),
        }
        reward = sum(self.weights[name] * value for name, value in values.items())
        # A threaded nut counts as carried; otherwise the grasp's derivative applies.
        carried = threaded + (1 - threaded) * grasped
        reward = (reward + self.carry_weight * carry * carried * valid) * self.dt
        # Imagination-only release shaping on the raw gripper command, whose
        # straight-through derivative acts directly (through the world model's
        # finger prediction, trained only at the two binary targets, it was too
        # weak to change a decision). The grasped-start categories all taught
        # "open at once": a nut over the peg drops and self-centres, and nothing
        # showed that the same reflex drops it on the table elsewhere. Opening a
        # held nut away from the peg (not threaded, centre beyond release_radius
        # of the axis) costs early_release_penalty; keeping the gripper closed
        # while the nut is threaded costs peg_hold_cost per second.
        held, _ = self.grasp_status(s, c)
        nut0, nut_quat0, _, _ = self._frames(s, c)
        on_peg = threaded_geometry(
            nut0, nut_quat0, peg, peg_quat, axial_tolerance=self.criteria.max_penetration
        ).to(s.dtype)
        axis = _apply_inverse(peg_quat, nut0 - peg)[:, :2].norm(dim=-1)
        over_peg = (axis < self.release_radius).to(s.dtype)
        early = held.detach() * (1 - on_peg) * (1 - over_peg) * self._commands_open(a)
        reward = reward - self.early_release_penalty * early
        reward = reward - self.peg_hold_cost * on_peg * self._commands_closed(a) * self.dt
        done = 1 - (1 - values["success"]) * (1 - invalid) * (1 - self._timeout(s1))
        return reward, done

    def task_contract(self):
        return {
            "imagined_success": "native threaded geometry with seated height and tilt, nut speeds below stable thresholds, both fingers' collision boxes (finger_boxes: per finger the pad within 2 cm of the grasp point and the rest, link6 axes, posed by the arm joints and finger openings, bounded axis-aligned in the nut frame) at least release_clearance from every nut box (release proxy, straight-through), grasp_seen and not invalid; no support-force, penetration or contact test",
            "finger_boxes": [
                [list(map(list, box)) for box in bands] for bands in self.finger_boxes
            ],
            "release_clearance": self.release_clearance,
            "imagined_grasp": "grasp point in the nut frame inside the exposed handle (legal) or over the ring (wrong), both fingers within grip_band (the legal grasp straight-through in the band); replaces force-bearing finger contact classification",
            "process_flags": "grasp_seen and invalid captured from the native success term, then latched analytically",
            "imagined_shaping": "carry_weight * exp(-distance to the nearer of the above-peg and seated poses / 1 m) added while the handle is grasped or the nut threaded; reaching also pays while grasped, pays only an open hand (both fingers past grip_band) or a legal grasp, and adds a far term 1 - tanh(distance / far_reaching_scale); handle_grasp and alignment gated off once threaded; imagination only",
            "carry_weight": self.carry_weight,
            "imagined_early_release": "early_release_penalty for commanding open while the handle is grasped, the nut is not threaded and its centre is beyond release_radius of the peg axis; native decision forward, hinge derivative backward on the raw gripper command; imagination only",
            "early_release_penalty": self.early_release_penalty,
            "release_radius": self.release_radius,
            "imagined_peg_release": "peg_hold_cost per second for commanding closed while the nut is threaded; native decision forward, hinge derivative backward on the raw gripper command; imagination only",
            "peg_hold_cost": self.peg_hold_cost,
            "grip_band": list(self.grip_band),
            "criteria": vars(self.criteria),
            "reward_scales": {
                "reaching": self.reaching_scale,
                "far_reaching": self.far_reaching_scale,
                "alignment": self.alignment_scale,
                "tilt": self.tilt_scale,
                "insertion": self.insertion_scale,
            },
        }


class SquareAdapter(RigidObjectAdapter):
    mdp_class = SquareMDP
    object_entity = "nut"
    target_entity = "peg"
    object_geoms = tuple(f"nut/nut_collision_{index}" for index in range(5))
    target_geoms = ("peg/peg_collision_0", "peg/peg_collision_1")
    actor_terms = (
        "nut_in_peg_frame",
        "previous_actions",
        "joint_pos",
        "ee_in_base_frame",
        "nut_in_ee_frame",
        "peg_in_ee_frame",
    )
    reward_terms = frozenset(
        {
            "action_magnitude",
            "action_rate",
            "joint_vel",
            "reaching",
            "handle_grasp",
            "alignment",
            "insertion",
            "release",
            "invalid_grasp",
            "success",
        }
    )

    def __init__(self, env, **options):
        rewards = env.cfg.rewards
        criteria = env.cfg.terminations["success"].params["criteria"]
        super().__init__(
            env,
            criteria=AssemblyCriteria(**criteria) if isinstance(criteria, dict) else criteria,
            finger_boxes=finger_boxes(
                env.sim.mj_model, robot=self.robot_binding(), depths=_FINGER_DEPTHS
            ),
            reaching_scale=rewards["reaching"].params["distance_scale"],
            alignment_scale=rewards["alignment"].params["distance_scale"],
            tilt_scale=rewards["alignment"].params["tilt_scale"],
            insertion_scale=rewards["insertion"].params["distance_scale"],
            **options,
        )

    @staticmethod
    def robot_binding():
        from .variants import DEFAULT_ROBOT

        return DEFAULT_ROBOT

    def task_bodies(self):
        return ["nut/nut", "peg/peg"]

    def process_state(self):
        return (("grasp_seen", 1), ("invalid", 1))

    def capture_extra(self):
        return {
            "grasp_seen": self.success.grasp_seen.float()[:, None],
            "invalid": self.success.invalid.float()[:, None],
        }
