"""Policy-independent physical process witnesses for Drawer.

A scripted source validates at known expert phase boundaries. RL has no such phase
oracle. This implementation advances through the same ordered physical milestones
using observed geometry/contact rather than planner labels. That is an explicit
task adaptation, not a literal replay of the expert's timing. The drawer starts
open: insertion/opening is neither a milestone nor an inherited prefix. Nothing
ends the task's sequence early: a grasp lost before the lift starts the grasp
again, a pod released anywhere may still come to rest in the drawer, and a pod
grasped again or out of the drawer while closing returns to the release stage.
Late reset prefixes are kept separate from this episode's accomplishments.
"""

import math

import torch

from .settings import (
    GRASP_SECONDS,
    MIN_CLOSE_DELTA,
    MIN_POD_LIFT,
    RELEASE_SECONDS,
    SUCCESS_SECONDS,
)


def pod_inside(pod_local, pod_height, slide):
    """Source body-origin predicate; no invented lower-height/contact threshold."""
    return (
        ((pod_local[:, 0] - slide).abs() < 0.08)
        & (pod_local[:, 1].abs() < 0.08)
        & (pod_height < 0.06)
    )


def final_candidate(pod_local, pod_height, slide, grasped):
    return (slide < 0.03) & pod_inside(pod_local, pod_height, slide) & ~grasped


class PhysicalSequence:
    """Vectorized ordered milestones; all history affecting success is observed.

    Stage 0 grasp/lift, 1 release, 2 push, 3 final hold.
    ``prefix`` only credits verified offline history. ``accomplished`` reports
    solely events witnessed since this reset. Counts use physical durations.
    ``strict`` (reset collection, whose rows must come from one unbroken process)
    instead latches ``failed`` on a grasp lost before the lift, a release outside the
    drawer and a re-grasp or escape while closing, and stops the sequence there.
    """

    observation_dim = 12

    def __init__(self, count, device, control_dt, strict=False):
        self.strict = strict
        self.stage = torch.zeros(count, dtype=torch.long, device=device)
        self.prefix = torch.zeros((count, 2), dtype=torch.bool, device=device)
        self.accomplished = torch.zeros((count, 3), dtype=torch.bool, device=device)
        self.initial_q = torch.zeros(count, device=device)
        self.initial_pod_z = torch.zeros(count, device=device)
        self.close_q = torch.zeros(count, device=device)
        self.grasp_count = torch.zeros(count, dtype=torch.long, device=device)
        self.release_count = torch.zeros_like(self.grasp_count)
        self.final_count = torch.zeros_like(self.grasp_count)
        self.grasp_confirmed = torch.zeros(count, dtype=torch.bool, device=device)
        self.push_contact = torch.zeros_like(self.grasp_confirmed)
        self.failed = torch.zeros_like(self.grasp_confirmed)
        self.success = torch.zeros_like(self.grasp_confirmed)
        self.grasp_steps = math.ceil(GRASP_SECONDS / control_dt - 1e-9)
        self.release_steps = math.ceil(RELEASE_SECONDS / control_dt - 1e-9)
        self.success_steps = math.ceil(SUCCESS_SECONDS / control_dt - 1e-9)

    def reset(self, indices, prefix, initial_q, initial_pod_z, close_q):
        if prefix.shape != (len(indices), 2) or not ((prefix == 0) | (prefix == 1)).all():
            raise ValueError("Drawer process prefix must be binary [N,2]")
        if (prefix[:, 1] > prefix[:, 0]).any():
            raise ValueError("Drawer release prefix requires the preceding grasp/lift witness")
        self.prefix[indices] = prefix.bool()
        self.stage[indices] = prefix.sum(-1).long()
        self.accomplished[indices] = False
        self.initial_q[indices] = initial_q
        self.initial_pod_z[indices] = initial_pod_z
        self.close_q[indices] = close_q
        for name in (
            "grasp_count",
            "release_count",
            "final_count",
            "grasp_confirmed",
            "push_contact",
            "failed",
            "success",
        ):
            getattr(self, name)[indices] = 0

    def update(
        self,
        *,
        slide,
        pod_z,
        inside,
        grasped,
        drawer_contact,
        final,
    ):
        prior = self.stage.clone()
        alive = ~self.failed
        lifting = (prior == 0) & alive
        self.grasp_count = torch.where(
            lifting & grasped,
            self.grasp_count + 1,
            torch.where(lifting, 0, self.grasp_count),
        )
        if self.strict:
            self.failed |= lifting & self.grasp_confirmed & ~grasped
        # Confirmation needs the current grasp unbroken for grasp_steps; a lost grasp
        # starts it again.
        self.grasp_confirmed = torch.where(
            lifting, self.grasp_count >= self.grasp_steps, self.grasp_confirmed
        )
        lifted = (
            lifting
            & ~self.failed
            & self.grasp_confirmed
            & grasped
            & ((pod_z - self.initial_pod_z) >= MIN_POD_LIFT)
        )
        self.stage[lifted] = 1
        self.accomplished[lifted, 0] = True

        releasing = (prior == 1) & alive
        if self.strict:
            self.failed |= releasing & ~grasped & ~inside
        self.release_count = torch.where(
            releasing & ~grasped & inside,
            self.release_count + 1,
            torch.where(releasing, 0, self.release_count),
        )
        released = releasing & ~self.failed & (self.release_count >= self.release_steps)
        self.stage[released] = 2
        self.close_q[released] = slide[released]
        self.accomplished[released, 1] = True

        closing = (prior == 2) & alive
        escaped = closing & (grasped | ~inside)
        if self.strict:
            self.failed |= escaped
        else:
            # Grasped again or out of the drawer: it has to be released inside again.
            self.stage[escaped] = 1
            self.release_count[escaped] = 0
            self.push_contact[escaped] = False
        closing &= ~escaped
        self.push_contact |= closing & drawer_contact
        closed = (
            closing
            & self.push_contact
            & ((self.close_q - slide) >= MIN_CLOSE_DELTA)
            & (slide <= 0.03)
        )
        self.stage[closed] = 3
        self.accomplished[closed, 2] = True
        candidate = (self.stage == 3) & ~self.failed & final
        self.final_count = torch.where(candidate, self.final_count + 1, 0)
        self.success[:] = self.final_count >= self.success_steps
        return self.success

    def observation(self):
        """The stage and the history the rest of the episode still depends on.

        Each stage's references and counters are observed only in that stage (zero
        elsewhere); the reset prefix, this episode's accomplishments and the initial
        slide, which never change the outcome, are not observed. A pod released in
        the drawer then looks the same whether the release happened in this episode
        or before a PodReleasedInOpenDrawer reset. With them observed, the critic
        valued a pod a PodHeldAboveOpenDrawer episode had just released at -1.8 and
        the same pod at that reset at +6, so placing it was never learned.
        """
        lift, release, push, hold = (self.stage == k for k in range(4))
        columns = (
            torch.nn.functional.one_hot(self.stage, 4).float(),
            torch.stack(
                (
                    self.initial_pod_z * lift,
                    self.grasp_count / self.grasp_steps * lift,
                    self.grasp_confirmed.float() * lift,
                    self.release_count / self.release_steps * release,
                    self.close_q * push,
                    self.push_contact.float() * push,
                    self.final_count / self.success_steps * hold,
                    self.success.float(),
                ),
                -1,
            ),
        )
        return torch.cat(columns, -1)
