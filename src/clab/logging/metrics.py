"""Scalar logging to TensorBoard, W&B, both, or neither."""

from pathlib import Path

import wandb
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, directory: Path, backend: str) -> None:
        if backend not in ("wandb", "tensorboard", "both", "none"):
            raise ValueError(f"Unknown logger backend: {backend}")
        self.use_wandb = backend in ("wandb", "both")
        self.writer = SummaryWriter(str(directory)) if backend in ("tensorboard", "both") else None

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        if self.use_wandb:
            wandb.log(metrics, step=step)
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
