"""Episode-counted success metrics over RSL-RL's unchanged training logger."""

from __future__ import annotations

import re

import torch
from rsl_rl.utils.logger import Logger

_COUNTS = re.compile(r"Episode_Metrics/type_(\d+)_(episodes|successes|success_rate)$")
_DIAGNOSTICS = re.compile(r"Episode_Diagnostics/([a-z][a-z0-9_]*)_(episodes|count|rate)$")
# Overall and per-category success; tasks have different category counts.
_PUBLISHED_SUCCESS_METRICS = re.compile(r"Episode_Metrics/(success_rate|type_\d+_success_rate)")


def _count_pair(episodes, count) -> torch.Tensor:
    """Accept Python denominators without mixing CPU and CUDA scalar tensors."""
    numerator = torch.as_tensor(count).reshape(())
    denominator = torch.as_tensor(episodes, device=numerator.device).reshape(())
    return torch.stack((denominator, numerator))


def _sum_counts(batches: list[torch.Tensor], label: str) -> tuple[int, int]:
    counts = torch.stack(batches).detach().cpu().to(torch.float64)
    if (
        not bool(torch.isfinite(counts).all())
        or not bool((counts == counts.round()).all())
        or not bool((counts[:, 0] > 0).all())
        or not bool(((counts[:, 1] >= 0) & (counts[:, 1] <= counts[:, 0])).all())
    ):
        raise ValueError(f"Invalid completed-episode counts for {label}")
    return tuple(int(value) for value in counts.sum(0).tolist())


def aggregate_episode_metrics(records: list[dict]) -> list[dict]:
    """Sum completed outcomes per iteration, not means of reset-batch rates.

    Other metrics retain upstream reduction. Missing categories are omitted,
    never reported as zero success; inputs and their tensors are not mutated.
    Counts cover this process only, as does upstream's rank-zero logging.
    """
    groups: dict[str, list[torch.Tensor]] = {}
    diagnostics: dict[str, list[torch.Tensor]] = {}
    cleaned = []
    for record in records:
        names = {match[1] for key in record if (match := _DIAGNOSTICS.fullmatch(key))}
        for name in names:
            prefix = f"Episode_Diagnostics/{name}"
            keys = (f"{prefix}_episodes", f"{prefix}_count")
            if any(key not in record for key in keys):
                raise ValueError(f"{prefix}: both episode and diagnostic counts are required")
            diagnostics.setdefault(name, []).append(_count_pair(*(record[key] for key in keys)))
        kinds = {match[1] for key in record if (match := _COUNTS.fullmatch(key))}
        if "Episode_Metrics/success_rate" in record and not kinds:
            raise ValueError("Episode success metrics require per-category integer counts")
        for kind in kinds:
            prefix = f"Episode_Metrics/type_{kind}"
            keys = (f"{prefix}_episodes", f"{prefix}_successes")
            if any(key not in record for key in keys):
                raise ValueError(f"{prefix}: both episode and success counts are required")
            groups.setdefault(kind, []).append(_count_pair(*(record[key] for key in keys)))
        cleaned.append(
            {
                key: value
                for key, value in record.items()
                if not _COUNTS.fullmatch(key)
                and not _DIAGNOSTICS.fullmatch(key)
                and key != "Episode_Metrics/success_rate"
            }
        )
    summary = {}
    episodes = successes = 0
    for kind, batches in groups.items():
        count, success = _sum_counts(batches, f"category {kind}")
        prefix = f"Episode_Metrics/type_{kind}"
        summary[f"{prefix}_episodes"] = count
        summary[f"{prefix}_successes"] = success
        summary[f"{prefix}_success_rate"] = success / count
        episodes += count
        successes += success
    if episodes:
        summary.update(
            {
                "Episode_Metrics/episodes": episodes,
                "Episode_Metrics/successes": successes,
                "Episode_Metrics/success_rate": successes / episodes,
            }
        )
    for name, batches in diagnostics.items():
        episodes, count = _sum_counts(batches, f"diagnostic {name}")
        prefix = f"Episode_Diagnostics/{name}"
        summary.update(
            {
                f"{prefix}_episodes": episodes,
                f"{prefix}_count": count,
                f"{prefix}_rate": count / episodes,
            }
        )
    if summary:
        cleaned.append(summary)
    return cleaned


def aggregate_success_rates(records: list[dict]) -> dict[str, float]:
    """Expose the same episode-weighted success rates to non-RSL training loops."""
    return {
        key: float(value)
        for record in aggregate_episode_metrics(records)
        for key, value in record.items()
        if _PUBLISHED_SUCCESS_METRICS.fullmatch(key)
    }


class EpisodeLogger(Logger):
    """Aggregate counts internally; publish only the success-rate metrics.

    Filter before the shared upstream logger so console, TensorBoard and W&B
    receive the same rates. Other namespaces, including diagnostics, are unchanged.
    A category with no completed episodes is omitted, not assigned a fake zero.
    """

    def log(self, *args, **kwargs) -> None:
        if self.writer is not None:
            self.ep_extras = [
                {
                    key: value
                    for key, value in record.items()
                    if not key.startswith("Episode_Metrics/")
                    or _PUBLISHED_SUCCESS_METRICS.fullmatch(key)
                }
                for record in aggregate_episode_metrics(self.ep_extras)
            ]
        super().log(*args, **kwargs)
