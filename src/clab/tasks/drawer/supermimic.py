"""Drawer SuperMimic surrogate: grasp/lift, release inside, push closed, hold.

The world model predicts the arm, the pod and the passive slide. The native
``PhysicalSequence`` milestones (stage, prefix, accomplishments, counters,
failure) are carried as analytic state and advanced with the native update
rules. Its two contact predicates are approximated geometrically: a two-finger
pod grasp is the physical grasp point near the pod centre with both fingers in
the closed-on-pod band, and drawer contact is the grasp point at the outside of
the front panel. Native real rewards and evaluation remain authoritative.
"""

import torch

from clab.controllers.torch_kinematics import quat_apply, quat_inverse, six_to_quat
from clab.tasks.supermimic import RigidObjectAdapter, RigidObjectMDP

from .process import final_candidate, pod_inside
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
    ("failed", 1),
)


def _apply_inverse(quaternion, vector):
    return quat_apply(quat_inverse(quaternion), vector)


class DrawerMDP(RigidObjectMDP):
    task = "drawer"
    object_name = "pod"
    target_name = "housing"
    actor_widths = (6, 7, 8, 6, 6, 6, 2, 19)
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        """Proxies for native (two-finger pod grasp, finger-drawer contact)."""
        pod = self.obj(s, "position")
        grip = self.physical_grasp(s, c)
        pod_local = _apply_inverse(six_to_quat(self.obj(s, "rotation")), grip - pod)
        fingers = self.state.get(s, "robot_q")[:, 6:8]
        low, high = self.grip_band
        closed = ((fingers >= low) & (fingers <= high)).all(-1)
        grasped = closed & (pod_local.norm(dim=-1) <= self.grasp_radius)
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
        hard = final_candidate(local, height, slide, grasped).to(s.dtype)
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
        at_hold = (self._field(s, "stage") == 3).to(s.dtype) * (1 - self._field(s, "failed"))
        return at_hold * self._final(s, c)

    def advance_task_fields(self, fields, s, c):
        """Native PhysicalSequence.update on the provisional (predicted) state."""
        provisional = self.state.pack(fields)
        local, pod_z, slide, _ = self._geometry(provisional, c)
        grasped, drawer = self.contacts(provisional, c)
        inside = pod_inside(local, pod_z, slide)
        get = lambda name: self._field(s, name)  # noqa: E731
        stage = get("stage").round().long()
        failed, confirmed = get("failed") > 0.5, get("grasp_confirmed") > 0.5
        push_contact = get("push_contact") > 0.5
        accomplished = self.state.get(s, "accomplished") > 0.5
        grasp_count, release_count = get("grasp_count"), get("release_count")
        close_q = get("close_q")
        alive = ~failed
        lifting = (stage == 0) & alive
        grasp_count = torch.where(
            lifting & grasped, grasp_count + 1, torch.where(lifting, 0, grasp_count)
        )
        failed = failed | (lifting & confirmed & ~grasped)
        confirmed = confirmed | (lifting & (grasp_count >= self.grasp_steps))
        lifted = (
            lifting
            & ~failed
            & confirmed
            & grasped
            & ((pod_z - get("initial_pod_z")) >= MIN_POD_LIFT)
        )
        releasing = (stage == 1) & alive
        failed = failed | (releasing & ~grasped & ~inside)
        release_count = torch.where(
            releasing & ~grasped & inside,
            release_count + 1,
            torch.where(releasing, 0, release_count),
        )
        released = releasing & ~failed & (release_count >= self.release_steps)
        closing = (stage == 2) & alive
        failed = failed | (closing & (grasped | ~inside))
        push_contact = push_contact | (closing & drawer)
        close_q = torch.where(released, slide, close_q)
        closed = (
            closing
            & ~failed
            & push_contact
            & ((close_q - slide) >= MIN_CLOSE_DELTA)
            & (slide <= 0.03)
        )
        stage = torch.where(lifted, 1, torch.where(released, 2, torch.where(closed, 3, stage)))
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
            failed=failed.to(dtype)[:, None],
        )
        super().advance_task_fields(fields, s, c)

    def extra_observation(self, s, c):
        g = self.state.get
        stage = self._field(s, "stage").round().long().clamp(0, 3)
        count = self._field(s, "success_count")
        process = torch.cat(
            (
                torch.nn.functional.one_hot(stage, 4).to(s.dtype),
                g(s, "prefix"),
                g(s, "accomplished"),
                g(s, "initial_q"),
                g(s, "initial_pod_z"),
                g(s, "close_q"),
                g(s, "grasp_count") / self.grasp_steps,
                g(s, "release_count") / self.release_steps,
                count[:, None] / self.required_steps,
                g(s, "grasp_confirmed"),
                g(s, "push_contact"),
                g(s, "failed"),
                (count >= self.required_steps).to(s.dtype)[:, None],
            ),
            -1,
        )
        return torch.cat((g(s, "slide_q"), g(s, "slide_dq")), -1), process

    def reward(self, s, a, s1, c):
        stage = self._field(s1, "stage")
        local, height, slide, housing_quat = self._geometry(s1, c)
        grip = self.physical_grasp(s1, c)
        panel_local = torch.stack(
            (slide + 0.099, torch.zeros_like(slide), torch.full_like(slide, 0.04)), -1
        )
        panel = self.target(c, "position") + quat_apply(housing_quat, panel_local)
        target = torch.where((stage < 2)[:, None], self.obj(s1, "position"), panel)
        reaching = 1 - torch.tanh(10 * (grip - target).norm(dim=-1))
        lift = ((height - self._field(s1, "initial_pod_z")) / 0.04).clamp(0, 1)
        place_target = torch.stack(
            (slide, torch.zeros_like(slide), torch.full_like(slide, 0.03)), -1
        )
        place = torch.exp(-10 * (local - place_target).norm(dim=-1))
        close = 1 - torch.tanh(10 * slide.abs())
        progress = torch.where(stage == 0, lift, torch.where(stage == 1, place, close))
        values = {
            "reaching": reaching,
            "progress": progress,
            "success": self._success_term(s, s1, c),
            **self._costs(s, a, s1),
        }
        reward = sum(self.weights[name] * value for name, value in values.items()) * self.dt
        failed = self._field(s1, "failed")
        done = 1 - (1 - values["success"]) * (1 - failed) * (1 - self._timeout(s1))
        return reward, done

    def task_contract(self):
        return {
            "imagined_success": "native PhysicalSequence update and final predicate on predicted states; pod grasp and drawer contact replaced by geometric proxies",
            "imagined_grasp": "physical grasp point within grasp_radius of the pod centre and both fingers within grip_band",
            "imagined_drawer_contact": "physical grasp point within panel_reach of the front panel outer face, inside its width and height",
            "grip_band": list(self.grip_band),
            "grasp_radius": self.grasp_radius,
            "panel_reach": list(self.panel_reach),
            "process_fields": "stage, prefix, accomplished, references, counters and flags captured from the native sequence, then advanced analytically",
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
            "failed": sequence.failed.float()[:, None],
        }
