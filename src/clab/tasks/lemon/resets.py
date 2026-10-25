"""Lemon reset replay in pad-aligned environment-world coordinates."""

from clab.tasks import rigid_resets

from .reset_data import SPEC


class ResetFromBank(rigid_resets.RigidResetFromBank):
    spec = SPEC


def snapshot(env, env_ids=None):
    return rigid_resets.snapshot(env, SPEC, env_ids)


def write_states(env, rows, env_ids=None):
    rigid_resets.write_states(env, rows, SPEC, env_ids)
