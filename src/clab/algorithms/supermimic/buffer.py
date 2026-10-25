"""Time-by-environment replay preserving native terminal POST states.

Adapted from zita-ch/supermimic_vibed; see REFERENCE_LICENSE.
Every add is one consecutive vectorized simulator step. The caller must capture
POST before automatic reset, not reconstruct terminal states from reset output.
Optionally, ``pin`` keeps the current rows as a fixed segment and continues
collection in a new streaming ring; sampling is uniform over both segments and
never forms a window across them.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity_steps,
        num_envs,
        state_dim,
        action_dim,
        context_dim,
        obs_dim,
        device,
    ):
        for name, value in {
            "capacity_steps": capacity_steps,
            "num_envs": num_envs,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "obs_dim": obs_dim,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"Replay {name} must be a positive integer")
        if type(context_dim) is not int or context_dim < 0:
            raise ValueError("Replay context_dim must be a nonnegative integer")
        self.capacity_steps, self.num_envs = capacity_steps, num_envs
        self.device = torch.device(device)
        self.ptr = self.size = 0
        self.data = {}
        widths = {
            "s": state_dim,
            "s1": state_dim,
            "a": action_dim,
            "context": context_dim,
            "obs": obs_dim,
        }
        for key, width in widths.items():
            self.data[key] = torch.zeros(capacity_steps, num_envs, width, device=self.device)
        for key, dtype in {
            "reward": torch.float32,
            "terminated": torch.bool,
            "truncated": torch.bool,
            "valid": torch.bool,
            "episode_step": torch.long,
        }.items():
            self.data[key] = torch.zeros(capacity_steps, num_envs, dtype=dtype, device=self.device)
        self._episode_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.pinned = None  # Fixed segment after pin(): data, ptr, size, capacity_steps.

    @property
    def allocated_nbytes(self) -> int:
        stores = [self.data] + ([self.pinned["data"]] if self.pinned is not None else [])
        return sum(value.numel() * value.element_size() for d in stores for value in d.values())

    @property
    def pinned_size(self) -> int:
        return 0 if self.pinned is None else self.pinned["size"]

    def pin(self, streaming_steps: int) -> None:
        """Keep every current row as a fixed segment; continue in a new streaming ring.

        Pinned rows are never overwritten or copied. Episode-step tracking
        continues, so the first streaming rows extend episodes in progress.
        """
        if self.pinned is not None:
            raise ValueError("Replay is already pinned")
        if type(streaming_steps) is not int or streaming_steps < 1:
            raise ValueError("Replay streaming_steps must be a positive integer")
        if not self.size:
            raise ValueError("Cannot pin an empty replay")
        self.pinned = {
            "data": self.data,
            "ptr": self.ptr,
            "size": self.size,
            "capacity_steps": self.capacity_steps,
        }
        self.data = {
            key: torch.zeros(
                (streaming_steps, *value.shape[1:]), dtype=value.dtype, device=self.device
            )
            for key, value in self.pinned["data"].items()
        }
        self.capacity_steps = streaming_steps
        self.ptr = self.size = 0

    @torch.no_grad()
    def add(self, transition: dict) -> None:
        # Validate every field before mutating the ring so a rejected row is atomic.
        for key, storage in self.data.items():
            if key in ("episode_step", "valid") and key not in transition:
                continue
            if key not in transition:
                raise ValueError(f"Replay transition is missing {key!r}")
            value = transition[key]
            if not isinstance(value, torch.Tensor) or value.shape != storage.shape[1:]:
                raise ValueError(f"Replay {key!r} must have shape {tuple(storage.shape[1:])}")
            if value.device != storage.device:
                raise ValueError(f"Replay {key!r} is on {value.device}, expected {storage.device}")
            if key in ("terminated", "truncated", "valid"):
                if value.dtype != torch.bool:
                    raise ValueError(f"Replay {key!r} must be boolean")
            elif key == "episode_step":
                if value.is_floating_point() or value.dtype == torch.bool or (value < 0).any():
                    raise ValueError("Replay episode_step must be nonnegative integer")
            elif not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"Replay {key!r} must contain finite floating-point values")
            elif not torch.isfinite(value.to(dtype=storage.dtype)).all():
                raise ValueError(f"Replay {key!r} cannot be represented in {storage.dtype}")
        defaults = {"episode_step": self._episode_step, "valid": True}
        for key, storage in self.data.items():
            value = transition[key] if key in transition else defaults[key]
            storage[self.ptr].copy_(value.detach())
        done = transition["terminated"] | transition["truncated"]
        step = transition.get("episode_step", self._episode_step)
        self._episode_step.copy_(torch.where(done, 0, step + 1))
        self.ptr = (self.ptr + 1) % self.capacity_steps
        self.size = min(self.size + 1, self.capacity_steps)

    def _row(self, age):
        return (self.ptr - self.size + age) % self.capacity_steps

    def _segments(self, segment=None):
        """(data, ptr, size, capacity) per segment, pinned first."""
        streaming = (self.data, self.ptr, self.size, self.capacity_steps)
        if self.pinned is None:
            if segment == "pinned":
                raise ValueError("Replay has no pinned segment")
            return [streaming]
        pinned = tuple(self.pinned[k] for k in ("data", "ptr", "size", "capacity_steps"))
        return {"pinned": [pinned], "streaming": [streaming], None: [pinned, streaming]}[segment]

    @staticmethod
    def _split(batch_size, weights):
        """Deterministic proportional counts keep the concatenated batch shape fixed."""
        if len(weights) == 1:
            return [batch_size]
        first = round(batch_size * weights[0] / sum(weights))
        return [first, batch_size - first]

    def sample_windows(
        self, batch_size: int, window: int, segment=None, env_range=None, newest=None
    ) -> dict:
        """Uniform over valid window starts of all (or one named) segment(s).

        ``env_range`` restricts worlds to ``[low, high)``. ``newest`` keeps only
        windows lying inside the newest rows of the streaming ring.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Replay batch_size must be a positive integer")
        low, high = (0, self.num_envs) if env_range is None else env_range
        if not (type(low) is int and type(high) is int and 0 <= low < high <= self.num_envs):
            raise ValueError("Replay env_range must be integer bounds within the world count")
        if newest is not None:
            if type(newest) is not int or newest < 1 or segment == "pinned":
                raise ValueError("Replay newest must be a positive integer for the streaming ring")
            segment = "streaming" if self.pinned is not None else None
        segments = self._segments(segment)
        if newest is not None:
            segments = [(data, ptr, min(size, newest), cap) for data, ptr, size, cap in segments]
        segments = [s for s in segments if type(window) is int and s[2] >= window]
        if type(window) is not int or window < 1 or not segments:
            size = self.size + self.pinned_size
            raise ValueError(f"Replay has {size} rows, cannot sample window={window}")
        counts = self._split(batch_size, [size - window + 1 for _, _, size, _ in segments])
        parts = []
        for (data, ptr, size, capacity), count in zip(segments, counts, strict=True):
            if not count:
                continue
            start = torch.randint(size - window + 1, (count,), device=self.device)
            if env_range is None:
                env = torch.randint(self.num_envs, (count,), device=self.device)
            else:
                env = torch.randint(low, high, (count,), device=self.device)
            ages = start[:, None] + torch.arange(window, device=self.device)
            rows = (ptr - size + ages) % capacity
            parts.append({key: values[rows, env[:, None]] for key, values in data.items()})
        out = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
        done = out["terminated"] | out["truncated"]
        # The terminal transition remains valid. Only later rows cross a reset.
        # An invalid row (diverged physics) and everything after it are masked.
        before = torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), dim=1)
        out["mask"] = torch.cumprod((before & out["valid"]).float(), dim=1)
        out["done"] = done
        return out

    def sample_starts(self, batch_size: int) -> dict:
        segments = [s for s in self._segments() if s[2]]
        if type(batch_size) is not int or batch_size < 1 or not segments:
            raise ValueError(
                "Replay imagination starts require positive batch_size and nonempty data"
            )
        counts = self._split(batch_size, [size for _, _, size, _ in segments])
        parts = []
        for (data, ptr, size, capacity), count in zip(segments, counts, strict=True):
            if not count:
                continue
            age = torch.randint(size, (count,), device=self.device)
            env = torch.randint(self.num_envs, (count,), device=self.device)
            rows = (ptr - size + age) % capacity
            # The task observation already includes its validated, episode-local history.
            parts.append({key: values[rows, env] for key, values in data.items()})
        return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
