"""Open-start Drawer bank resets with physically inherited process history."""

from pathlib import Path

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.manager_base import ManagerTermBase

from clab.tasks.rigid_resets import record_finished_episodes

from .mdp import sequence
from .reset_data import FIELDS, WIDTHS, bank_metadata, validate_bank
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


def write_states(env, rows, env_ids=None):
    indices = resolve_env_ids(env, env_ids)
    if tuple(rows) != FIELDS:
        raise ValueError(f"Drawer rows must follow {FIELDS}")
    for key, width in zip(FIELDS, WIDTHS, strict=True):
        if rows[key].shape != (len(indices), width):
            raise ValueError(f"Incorrect {key} reset shape: {rows[key].shape}")
    env.scene["robot"].write_joint_state_to_sim(
        rows["joint_pos"], rows["joint_vel"], env_ids=indices
    )
    pod_pose, housing_pose = rows["pod_pose"].clone(), rows["housing_pose"].clone()
    pod_pose[:, :3] += env.scene.env_origins[indices]
    housing_pose[:, :3] += env.scene.env_origins[indices]
    env.scene["pod"].write_root_state_to_sim(
        torch.cat((pod_pose, rows["pod_vel"]), -1), env_ids=indices
    )
    env.scene["fixture"].write_mocap_pose_to_sim(housing_pose, env_ids=indices)
    env.scene["fixture"].write_joint_state_to_sim(
        rows["slide_pos"], rows["slide_vel"], env_ids=indices
    )
    env.scene["robot"].set_joint_velocity_target(
        torch.zeros_like(rows["joint_vel"]), env_ids=indices
    )
    sequence(env).reset(
        indices,
        rows["sequence_prefix"],
        rows["initial_drawer_qpos"][:, 0],
        rows["initial_pod_height"][:, 0],
        rows["close_start_qpos"][:, 0],
    )


class ResetFromBank(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        path = cfg.params.get("path")
        if not path:
            raise ValueError("Provide a validated Drawer reset bank")
        bank = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        states = validate_bank(bank, env.sim.mj_model)
        self.metadata = bank_metadata(bank)
        self.states = {
            kind: {key: value.to(env.device) for key, value in rows.items()}
            for kind, rows in states.items()
        }
        self.task_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.type_override = None
        self._validated = False

    def __call__(self, env, env_ids, path, reaching_only=False):
        del path
        if not self._validated:
            DEFAULT_ROBOT.validate_env(env)
            self._validated = True
        indices = resolve_env_ids(env, env_ids)
        record_finished_episodes(env, indices, self.task_id, len(RESET_TYPES))
        if self.type_override is not None and self.type_override not in range(len(RESET_TYPES)):
            raise ValueError(f"Invalid Drawer reset type {self.type_override}")
        selected_type = (
            self.type_override if self.type_override is not None else (0 if reaching_only else None)
        )
        kinds = (
            torch.full((len(indices),), selected_type, dtype=torch.long, device=env.device)
            if selected_type is not None
            else torch.randint(len(RESET_TYPES), (len(indices),), device=env.device)
        )
        self.task_id[indices] = kinds
        for category, name in enumerate(RESET_TYPES):
            selected = indices[kinds == category]
            if not len(selected):
                continue
            data = self.states[name]
            rows = torch.randint(len(data["joint_pos"]), (len(selected),), device=env.device)
            write_states(env, {key: value[rows] for key, value in data.items()}, selected)
