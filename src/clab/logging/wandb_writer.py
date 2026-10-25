"""W&B writer compatible with the uv-pinned RSL-RL and W&B APIs."""

from __future__ import annotations

import os
from pathlib import Path

import wandb
from rsl_rl.utils.wandb_log_writer import WandbLogWriter
from torch.utils.tensorboard import SummaryWriter


class ClabWandbLogWriter(WandbLogWriter):
    """Use the upstream writer methods without its removed W&B setting."""

    def __init__(self, log_dir: str, project_name: str) -> None:
        SummaryWriter.__init__(self, log_dir, flush_secs=10)
        run = wandb.init(
            project=project_name,
            entity=os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME"),
            name=Path(log_dir).name,
            dir=log_dir,
            config={"log_dir": log_dir},
            settings=wandb.Settings(init_timeout=120),
        )
        if run is None:
            raise RuntimeError("W&B initialization returned no run")
        print(f"W&B run: {run.url}", flush=True)
        self.logged_videos: set[str] = set()
        self._pending_step: int | None = None
        self._pending_metrics: dict[str, float] = {}

    def _flush_metrics(self) -> None:
        if self._pending_metrics:
            wandb.log(self._pending_metrics, step=self._pending_step)
            self._pending_metrics = {}

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Commit all metrics for one PPO iteration in a single W&B step."""
        SummaryWriter.add_scalar(
            self,
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        # RSL-RL also emits TensorBoard-only */time series using elapsed
        # seconds as their step. They cannot share W&B's iteration axis.
        if tag.endswith("/time"):
            return
        if global_step is None:
            raise ValueError("PPO W&B metrics require an explicit iteration step")
        if self._pending_step is not None and global_step != self._pending_step:
            if global_step < self._pending_step:
                raise ValueError("PPO W&B iteration steps must be nondecreasing")
            self._flush_metrics()
        self._pending_step = global_step
        self._pending_metrics[tag] = float(scalar_value)

    def save_model(self, model_path: str, it: int) -> None:
        """Flush metrics without uploading checkpoints, including legacy callers."""
        self._flush_metrics()

    def stop(self) -> None:
        self._flush_metrics()
        wandb.finish()
        self.close()
