"""Device-resident constants for code that runs every simulation or control step."""

from functools import lru_cache

import torch


def constant(like: torch.Tensor, values) -> torch.Tensor:
    """``like.new_tensor(values)``, created once per device and dtype.

    Each ``new_tensor`` of a Python constant is a host-to-device copy that waits for
    the device queue; in per-step code these copies dominated the synchronizations.
    The cached tensor is shared: never modify it in place. Compiled code keeps
    ``new_tensor``, which torch.compile folds into the graph.
    """
    if torch.compiler.is_compiling():
        return like.new_tensor(values)
    return _cached(_frozen(values), like.device, like.dtype)


def _frozen(values):
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        return tuple(_frozen(value) for value in values)
    return values


@lru_cache(maxsize=None)
def _cached(values, device, dtype):
    # A normal (not inference-mode) tensor, usable later under autograd.
    with torch.inference_mode(False), torch.no_grad():
        return torch.tensor(values, device=device, dtype=dtype)
