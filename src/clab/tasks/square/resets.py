"""Apply Square reset banks through the shared rigid-prop reset implementation."""

from clab.tasks import rigid_resets

from .reset_data import SPEC


class ResetFromBank(rigid_resets.RigidResetFromBank):
    spec = SPEC

    def __call__(self, env, env_ids, path: str | None, reaching_only: bool = False):
        # Snapshot task-owned episode flags before the event writes a new reset
        # and before TerminationManager.reset clears them. Manual resets do not
        # count as completed episodes. No task-specific writer is introduced.
        if env.common_step_counter > 0:
            indices = rigid_resets._indices(env, env_ids)
            completed = indices[env.reset_buf[indices]]
            if len(completed):
                outcome = env.termination_manager.get_term_cfg("success").func
                for name, value in (
                    ("handle_grasp", outcome.grasp_seen),
                    ("threaded", outcome.threaded_seen),
                    ("invalid_grasp", outcome.invalid),
                ):
                    prefix = f"Episode_Diagnostics/{name}"
                    env.extras["log"][f"{prefix}_episodes"] = len(completed)
                    env.extras["log"][f"{prefix}_count"] = value[completed].sum()
        super().__call__(env, env_ids, path, reaching_only)


def snapshot(env, env_ids=None):
    return rigid_resets.snapshot(env, SPEC, env_ids)


def write_states(env, rows, env_ids=None):
    rigid_resets.write_states(env, rows, SPEC, env_ids)
