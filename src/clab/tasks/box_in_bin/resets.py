"""Four-way BoxInBin reset replay in robot-base-local coordinates."""

from __future__ import annotations

from pathlib import Path

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from clab.tasks.rigid_resets import record_finished_episodes

from .reset_data import bank_metadata, validate_bank
from .settings import RESET_TYPES
from .variants import DEFAULT_ROBOT


def world_pose(pose: torch.Tensor, position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Map the bank's robot-base-local xyz/wxyz pose into the current world."""
    rotation = rotation.expand(*pose.shape[:-1], 4)
    return torch.cat(
        (position + quat_apply(rotation, pose[..., :3]), quat_mul(rotation, pose[..., 3:])), dim=-1
    )


class ResetFromBank(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(env)
        path = cfg.params.get("path")
        if not path:
            raise ValueError("Provide a BoxInBin reset bank path")
        bank_path = Path(path).expanduser()
        if not bank_path.is_file():
            raise FileNotFoundError(f"BoxInBin reset bank not found: {bank_path}")
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        validate_bank(bank, model=env.sim.mj_model)
        self.metadata = bank_metadata(bank)
        self.states = {
            kind: {key: value.to(env.device) for key, value in rows.items()}
            for kind, rows in bank["states"].items()
        }
        self._validated = False
        self.task_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.type_override: int | None = None

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        path: str | None,
        reaching_only: bool = False,
    ) -> None:
        del path
        if not self._validated:
            DEFAULT_ROBOT.validate_env(env)
            self._validated = True
        env_ids = resolve_env_ids(env, env_ids)
        record_finished_episodes(env, env_ids, self.task_id, len(RESET_TYPES))
        count = len(env_ids)
        if self.type_override is not None:
            kinds = torch.full((count,), self.type_override, dtype=torch.long, device=env.device)
        elif reaching_only:
            kinds = torch.zeros(count, dtype=torch.long, device=env.device)
        else:
            kinds = torch.randint(len(RESET_TYPES), (count,), device=env.device)
        self.task_id[env_ids] = kinds
        robot = env.scene["robot"]
        box = env.scene["box"]
        bin_asset = env.scene["bin"]
        for type_id, kind in enumerate(RESET_TYPES):
            selected = kinds == type_id
            if not selected.any():
                continue
            indices = env_ids[selected]
            rows = self.states[kind]
            sample = torch.randint(rows["joint_pos"].shape[0], (len(indices),), device=env.device)
            robot.write_joint_state_to_sim(
                rows["joint_pos"][sample], rows["joint_vel"][sample], env_ids=indices
            )
            position = robot.data.root_link_pos_w[indices]
            rotation = robot.data.root_link_quat_w[indices]
            box_pose = world_pose(rows["box_pose"][sample], position, rotation)
            bin_pose = world_pose(rows["bin_pose"][sample], position, rotation)
            velocity = rows["box_vel"][sample]
            velocity = torch.cat(
                (quat_apply(rotation, velocity[:, :3]), quat_apply(rotation, velocity[:, 3:])),
                dim=-1,
            )
            box.write_root_state_to_sim(torch.cat((box_pose, velocity), dim=-1), env_ids=indices)
            bin_asset.write_mocap_pose_to_sim(bin_pose, env_ids=indices)
        robot.set_joint_velocity_target(
            torch.zeros((count, len(DEFAULT_ROBOT.joints)), device=env.device), env_ids=env_ids
        )
