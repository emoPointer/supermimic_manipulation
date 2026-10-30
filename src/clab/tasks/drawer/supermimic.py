"""Drawer SuperMimic surrogate: grasp/lift, release inside, push closed, hold.

The world model predicts the arm, the pod and the passive slide. The native
``PhysicalSequence`` milestones (stage, prefix, accomplishments, counters) are
carried as analytic state and advanced with the native update rules, which never
fail. Its two contact predicates are approximated geometrically: a two-finger
pod grasp is the physical grasp point near the pod centre with both fingers in
the closed-on-pod band, and drawer contact is the grasp point at the outside of
the front panel. Native real rewards and evaluation remain authoritative.
"""

import torch

from clab.controllers.torch_kinematics import quat_apply, quat_inverse, six_to_quat
from clab.tasks.supermimic import RigidObjectAdapter, RigidObjectMDP

from .process import PhysicalSequence, final_candidate, pod_inside
from .settings import GRASP_SECONDS, MIN_CLOSE_DELTA, MIN_POD_LIFT, RELEASE_SECONDS

# Front panel outer face in drawer-local x, its half-width and height span.
_PANEL_X, _PANEL_HALF_Y, _PANEL_Z = 0.1, 0.093, (0.0, 0.08)
_PROCESS = (
    ("stage", 1),
    ("prefix", 2),
    ("accomplished", 3),
    ("initial_q", 1),
    ("initial_pod_z", 1),
    ("close_q", 1),
    ("grasp_count", 1),
    ("release_count", 1),
    ("grasp_confirmed", 1),
    ("push_contact", 1),
)


def _apply_inverse(quaternion, vector):
    return quat_apply(quat_inverse(quaternion), vector)


class DrawerMDP(RigidObjectMDP):
    task = "drawer"
    object_name = "pod"
    target_name = "housing"
    actor_widths = (6, 7, 8, 6, 6, 6, 2, PhysicalSequence.observation_dim)
    reward_mode = "imagined_physical_sequence_with_geometric_contact_proxies"
    process_fields = tuple(name for name, _ in _PROCESS)
    # Pod grasps hold each finger at ~0.024 m (pod diameter ~48 mm).
    held_width = 0.024
    release_opening = 0.034

    def __init__(
        self,
        *,
        grip_band=(0.018, 0.030),
        grasp_radius=0.03,
        panel_reach=(-0.005, 0.03),
        outside_release_penalty=2.0,
        inside_hold_cost=2.0,
        far_reaching_scale=0.5,
        action_bound_weight=0.05,
        **kwargs,
    ):
        super().__init__(action_bound_weight=action_bound_weight, **kwargs)
        process = [self.state.slices[name] for name, _ in _PROCESS]
        if process[-1].stop != self.state_dim or any(
            a.stop != b.start for a, b in zip(process, process[1:], strict=False)
        ):
            raise ValueError("Drawer process fields must end the rollout state")
        self._process_start = process[0].start
        self.critic_dim -= self.state_dim - self._process_start
        for name, value in (
            ("outside_release_penalty", outside_release_penalty),
            ("inside_hold_cost", inside_hold_cost),
        ):
            if not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        self.outside_release_penalty = float(outside_release_penalty)
        self.inside_hold_cost = float(inside_hold_cost)
        self.far_reaching_scale = float(far_reaching_scale)
        if not 0 < self.far_reaching_scale < float("inf"):
            raise ValueError("far_reaching_scale must be finite and positive")
        self.grip_band = tuple(float(v) for v in grip_band)
        self.grasp_radius = float(grasp_radius)
        self.panel_reach = tuple(float(v) for v in panel_reach)
        steps = lambda seconds: round(seconds / self.dt)  # noqa: E731
        self.grasp_steps, self.release_steps = steps(GRASP_SECONDS), steps(RELEASE_SECONDS)

    def _field(self, s, name):
        return self.state.get(s, name).squeeze(-1)

    def _geometry(self, s, c):
        """Native process geometry: pod in the housing frame, height and slide."""
        pod = self.obj(s, "position")
        housing_quat = six_to_quat(self.target(c, "rotation"))
        local = _apply_inverse(housing_quat, pod - self.target(c, "position"))
        return local, pod[:, 2], self._field(s, "slide_q"), housing_quat

    def contacts(self, s, c):
        """Proxies for native (two-finger pod grasp, finger-drawer contact).

        The grasp is straight-through in the finger band; drawer contact is hard.
        """
        pod = self.obj(s, "position")
        grip = self.physical_grasp(s, c)
        pod_local = _apply_inverse(six_to_quat(self.obj(s, "rotation")), grip - pod)
        fingers = self.state.get(s, "robot_q")[:, 6:8]
        low, high = self.grip_band
        closed = self._band(fingers, low, high, self.release_temperature).prod(-1)
        grasped = closed * (pod_local.norm(dim=-1) <= self.grasp_radius).to(closed.dtype)
        _, _, slide, housing_quat = self._geometry(s, c)
        local = _apply_inverse(housing_quat, grip - self.target(c, "position"))
        front = local[:, 0] - slide - _PANEL_X
        drawer = (
            (front >= self.panel_reach[0])
            & (front <= self.panel_reach[1])
            & (local[:, 1].abs() <= _PANEL_HALF_Y)
            & (local[:, 2] >= _PANEL_Z[0])
            & (local[:, 2] <= _PANEL_Z[1])
        )
        return grasped, drawer

    def _final(self, s, c):
        """Native final predicate with differentiable slide and placement margins."""
        local, height, slide, _ = self._geometry(s, c)
        grasped, _ = self.contacts(s, c)
        hard = final_candidate(local, height, slide, grasped.detach() > 0.5).to(s.dtype)
        margins = torch.stack(
            (
                0.03 - slide,
                0.08 - (local[:, 0] - slide).abs(),
                0.08 - local[:, 1].abs(),
                0.06 - height,
            ),
            -1,
        )
        soft = torch.sigmoid(margins / self.smooth_temperature).prod(-1)
        return hard + (soft - soft.detach())

    def _candidate(self, s, c):
        return (self._field(s, "stage") == 3).to(s.dtype) * self._final(s, c)

    def advance_task_fields(self, fields, s, c):
        """Native PhysicalSequence.update on the provisional (predicted) state."""
        provisional = self.state.pack(fields)
        local, pod_z, slide, _ = self._geometry(provisional, c)
        grasped, drawer = self.contacts(provisional, c)
        grasped = grasped.detach() > 0.5
        inside = pod_inside(local, pod_z, slide)
        get = lambda name: self._field(s, name)  # noqa: E731
        stage = get("stage").round().long()
        push_contact = get("push_contact") > 0.5
        accomplished = self.state.get(s, "accomplished") > 0.5
        grasp_count, release_count = get("grasp_count"), get("release_count")
        close_q = get("close_q")
        lifting = stage == 0
        grasp_count = torch.where(
            lifting & grasped, grasp_count + 1, torch.where(lifting, 0, grasp_count)
        )
        confirmed = torch.where(
            lifting, grasp_count >= self.grasp_steps, get("grasp_confirmed") > 0.5
        )
        lifted = lifting & confirmed & grasped & ((pod_z - get("initial_pod_z")) >= MIN_POD_LIFT)
        releasing = stage == 1
        release_count = torch.where(
            releasing & ~grasped & inside,
            release_count + 1,
            torch.where(releasing, 0, release_count),
        )
        released = releasing & (release_count >= self.release_steps)
        closing = stage == 2
        escaped = closing & (grasped | ~inside)
        closing = closing & ~escaped
        release_count = torch.where(escaped, 0, release_count)
        push_contact = ~escaped & (push_contact | (closing & drawer))
        close_q = torch.where(released, slide, close_q)
        closed = closing & push_contact & ((close_q - slide) >= MIN_CLOSE_DELTA) & (slide <= 0.03)
        stage = torch.where(
            lifted | escaped,
            1,
            torch.where(released, 2, torch.where(closed, 3, stage)),
        )
        accomplished = accomplished | torch.stack((lifted, released, closed), -1)
        dtype = provisional.dtype
        fields.update(
            stage=stage.to(dtype)[:, None],
            accomplished=accomplished.to(dtype),
            close_q=close_q[:, None],
            grasp_count=grasp_count[:, None],
            release_count=release_count[:, None],
            grasp_confirmed=confirmed.to(dtype)[:, None],
            push_contact=push_contact.to(dtype)[:, None],
        )
        super().advance_task_fields(fields, s, c)

    def extra_observation(self, s, c):
        """Slide state and the native stage-masked process observation."""
        g = self.state.get
        stage = self._field(s, "stage").round().long().clamp(0, 3)
        count = self._field(s, "success_count")
        onehot = torch.nn.functional.one_hot(stage, 4).to(s.dtype)
        lift, release, push, hold = onehot[:, 0:1], onehot[:, 1:2], onehot[:, 2:3], onehot[:, 3:4]
        process = torch.cat(
            (
                onehot,
                g(s, "initial_pod_z") * lift,
                g(s, "grasp_count") / self.grasp_steps * lift,
                g(s, "grasp_confirmed") * lift,
                g(s, "release_count") / self.release_steps * release,
                g(s, "close_q") * push,
                g(s, "push_contact") * push,
                count[:, None] / self.required_steps * hold,
                (count >= self.required_steps).to(s.dtype)[:, None],
            ),
            -1,
        )
        return torch.cat((g(s, "slide_q"), g(s, "slide_dq")), -1), process

    def critic_features(self, s, c, obs):
        """Rollout state without the raw process fields, context and actor history.

        The process reaches the critic only through the stage-masked observation
        (PhysicalSequence.observation), so equal futures get equal inputs whichever
        reset category the episode started from.
        """
        return torch.cat((s[..., : self._process_start], c, obs), -1)

    def reward(self, s, a, s1, c):
        stage = self._field(s1, "stage")
        local, height, slide, housing_quat = self._geometry(s1, c)
        grip = self.physical_grasp(s1, c)
        # Imagination-only: the native closing target sits on the panel's outer
        # face, where a hand left inside the drawer scored almost as well as one
        # pushing from the front, then jammed against the housing at ~0.08 m.
        # Aim 2 cm in front of the face so pushing from outside pays more.
        panel_local = torch.stack(
            (slide + _PANEL_X + 0.02, torch.zeros_like(slide), torch.full_like(slide, 0.04)), -1
        )
        panel = self.target(c, "position") + quat_apply(housing_quat, panel_local)
        target = torch.where((stage < 2)[:, None], self.obj(s1, "position"), panel)
        distance = (grip - target).norm(dim=-1)
        reaching = 1 - torch.tanh(10 * distance)
        # Imagination-only: before the pod is released, reaching is averaged with a
        # far term (the native 10 cm tanh is flat at the ~0.5 m full-task start) and,
        # until the grasp is confirmed, pays only an open hand or a grasp: the
        # full-task policy shut its gripper at once and a closed hand beside the pod
        # (which cannot grasp it) was paid like an open one.
        grasped1, _ = self.contacts(s1, c)
        fingers = self.state.get(s1, "robot_q")[:, 6:8]
        opened = self._gate(fingers - self.grip_band[1], self.release_temperature).prod(-1)
        seen = torch.maximum(self._field(s1, "grasp_confirmed"), (stage >= 1).to(s1.dtype))
        ready = seen + (1 - seen) * torch.maximum(opened, grasped1)
        far = 1 - torch.tanh(distance / self.far_reaching_scale)
        # Imagination-only stage credit, as for progress: the release ends reaching
        # for the held pod (~1) and the closing stage's reach starts ~0.1 from the
        # panel, so releasing inside the drawer lowered the reward and the held pod
        # was carried away from it. Count the completed reach.
        reaching = torch.where(stage < 2, (reaching + far) / 2 * ready, 1 + reaching)
        lift = ((height - self._field(s1, "initial_pod_z")) / 0.04).clamp(0, 1)
        place_target = torch.stack(
            (slide, torch.zeros_like(slide), torch.full_like(slide, 0.03)), -1
        )
        place = torch.exp(-10 * (local - place_target).norm(dim=-1))
        close = 1 - torch.tanh(10 * slide.abs())
        progress = torch.where(stage == 0, lift, torch.where(stage == 1, place, close))
        # Imagination-only shaping: the native progress restarts at every stage
        # (lift ~1 becomes place ~0.05; place ~1 becomes close ~0.14), so
        # completing a stage lowered the reward and stopping just short of it
        # paid more. Count completed stages so progress never drops at a stage.
        progress = stage.clamp(0, 3) + progress
        values = {
            "reaching": reaching,
            "progress": progress,
            "success": self._success_term(s, s1, c),
            **self._costs(s, a, s1),
        }
        reward = sum(self.weights[name] * value for name, value in values.items()) * self.dt
        # Imagination-only release shaping on the raw gripper command, whose
        # straight-through derivative acts directly; through the world model's
        # finger prediction (trained at the two binary targets) the derivative was
        # too weak to change a decision. Opening a held pod costs
        # outside_release_penalty in the lift stage (the grasped-on-table starts
        # learned to open at once and dropped it) and, in the release stage,
        # unless the pod is over the drawer opening: released there it drops into
        # the drawer (every one of 256 from the PodHeldAboveOpenDrawer pose, 12 cm
        # above the bottom), elsewhere onto the table. Keeping the gripper closed
        # once the pod is inside costs inside_hold_cost per second (natively
        # nothing pays for letting go until the stage advances).
        stage0 = self._field(s, "stage").round()
        releasing = (stage0 == 1).to(s.dtype)
        lifting = (stage0 == 0).to(s.dtype)
        held, _ = self.contacts(s, c)
        local0, height0, slide0, _ = self._geometry(s, c)
        inside0 = pod_inside(local0, height0, slide0).to(s.dtype)
        over0 = (((local0[:, 0] - slide0).abs() < 0.08) & (local0[:, 1].abs() < 0.08)).to(s.dtype)
        outside = (lifting + releasing * (1 - over0)) * held.detach() * self._commands_open(a)
        reward = reward - self.outside_release_penalty * outside
        hold = releasing * inside0 * self._commands_closed(a)
        reward = reward - self.inside_hold_cost * hold * self.dt
        done = 1 - (1 - values["success"]) * (1 - self._timeout(s1))
        return reward, done

    def task_contract(self):
        return {
            "imagined_success": "native PhysicalSequence update (no failure: a lost grasp before the lift restarts the grasp, a pod grasped or out of the drawer while closing returns to the release stage) and final predicate on predicted states; pod grasp and drawer contact replaced by geometric proxies; done only on success or the time limit",
            "imagined_grasp": "physical grasp point within grasp_radius of the pod centre and both fingers within grip_band (straight-through in the band)",
            "imagined_release": "outside_release_penalty for commanding open while the pod is held, in the lift stage and in the release stage unless the pod is over the drawer opening (the inside footprint at any height); inside_hold_cost per second for commanding closed while the pod is inside in the release stage; native decision forward, hinge derivative backward on the raw gripper command; imagination only",
            "outside_release_penalty": self.outside_release_penalty,
            "inside_hold_cost": self.inside_hold_cost,
            "imagined_drawer_contact": "physical grasp point within panel_reach of the front panel outer face, inside its width and height",
            "grip_band": list(self.grip_band),
            "grasp_radius": self.grasp_radius,
            "panel_reach": list(self.panel_reach),
            "process_fields": "stage, prefix, accomplished, references, counters and contact flags captured from the native sequence, then advanced analytically",
            "imagined_progress_shaping": "completed stages plus the native stage progress, so progress never drops when a stage completes; imagination only",
            "imagined_closing_reach": "closing-stage reaching target 2 cm in front of the panel's outer face (natively on the face); imagination only",
            "imagined_ready_reach": "before the closing stage, reaching averages the native term with 1 - tanh(distance / far_reaching_scale) and, until grasp_confirmed or the lift stage completes, pays only an open hand (both fingers past grip_band) or a pod grasp; in the closing and final stages reaching is 1 plus the native term (the completed reach counted, like progress); imagination only",
            "far_reaching_scale": self.far_reaching_scale,
        }


class DrawerAdapter(RigidObjectAdapter):
    mdp_class = DrawerMDP
    object_entity = "pod"
    target_entity = "fixture"
    object_geoms = ("pod/pod_collision",)
    target_geoms = tuple(f"fixture/housing_collision_{index}" for index in range(6)) + tuple(
        f"fixture/drawer_collision_{index}" for index in range(4)
    )
    actor_terms = (
        "pod_in_housing",
        "previous_actions",
        "joint_pos",
        "ee_in_base",
        "pod_in_ee",
        "housing_in_ee",
        "slide_state",
        "process_state",
    )
    reward_terms = frozenset(
        {"reaching", "progress", "action_magnitude", "action_rate", "joint_vel", "success"}
    )

    @staticmethod
    def robot_binding():
        from .variants import DEFAULT_ROBOT

        return DEFAULT_ROBOT

    def task_bodies(self):
        return ["pod/pod", "fixture/housing", "fixture/drawer"]

    def extra_physical_state(self):
        return (("slide_q", 1), ("slide_dq", 1))

    def process_state(self):
        return _PROCESS

    def required_steps(self):
        return self.success.state.success_steps

    def native_success_count(self):
        return self.success.state.final_count.float()[:, None]

    def native_candidate(self):
        from .mdp import geometry

        sequence = self.success.state
        local, height, slide, grasped = geometry(self.env)
        final = final_candidate(local, height, slide, grasped)
        return (sequence.stage == 3) & ~sequence.failed & final

    def capture_extra(self):
        fixture = self.env.scene["fixture"].data
        sequence = self.success.state
        return {
            "slide_q": fixture.joint_pos[:, :1],
            "slide_dq": fixture.joint_vel[:, :1],
            "stage": sequence.stage.float()[:, None],
            "prefix": sequence.prefix.float(),
            "accomplished": sequence.accomplished.float(),
            "initial_q": sequence.initial_q[:, None],
            "initial_pod_z": sequence.initial_pod_z[:, None],
            "close_q": sequence.close_q[:, None],
            "grasp_count": sequence.grasp_count.float()[:, None],
            "release_count": sequence.release_count.float()[:, None],
            "grasp_confirmed": sequence.grasp_confirmed.float()[:, None],
            "push_contact": sequence.push_contact.float()[:, None],
        }
