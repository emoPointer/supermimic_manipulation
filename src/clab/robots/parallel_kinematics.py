"""Ordered CPU process pool for independent kinematic proposal work.

Each spawned process constructs one private state with ``factory(**kwargs)``.
Task-owned module-level functions receive ``(state, seed_words, *job_args)``;
they own sampling, context updates and acceptance. This module knows no task,
reset schema or candidate distribution. The same seed, jobs, chunking and call
order reproduce streams; parallel and serial samplers need not emit equal rows.
"""

from __future__ import annotations

import multiprocessing
import os
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from threading import Lock
from typing import Any

import numpy as np

_SPAWN_LOCK = Lock()
_WORKER_STATE: Any = None

# Bound the automatic policy, not explicit overrides. Respect CPU affinity on
# shared servers; GPU world count is an independent task-specific capacity.
DEFAULT_PROPOSAL_WORKERS = min(
    64, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
)


@contextmanager
def _worker_environment():
    """Limit imports in spawned children, then restore every parent variable."""
    settings = dict.fromkeys(
        (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ),
        "1",
    )
    settings["CUDA_VISIBLE_DEVICES"] = ""
    with _SPAWN_LOCK:
        previous = {name: os.environ.get(name) for name in settings}
        os.environ.update(settings)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _initialize(factory: Callable, kwargs: dict) -> None:
    """Private model/state per CPU worker; no CUDA context is inherited."""
    import torch

    global _WORKER_STATE
    torch.set_num_threads(1)
    _WORKER_STATE = factory(**kwargs)


def _run(function: Callable, seed_words: list[int], job: tuple):
    return function(_WORKER_STATE, seed_words, *job)


class CpuProposalPool:
    """Reusable ordered spawn pool with explicit seed streams and clean shutdown.

    ``factory`` and worker functions must be importable module-level callables.
    Jobs are tuples of positional arguments. Exceptions propagate to the caller;
    use a context manager or ``finally: pool.close()`` to reclaim worker models.
    """

    def __init__(
        self, factory: Callable, *, factory_kwargs: dict | None = None, seed: int, workers: int
    ):
        if workers < 1 or seed < 0:
            raise ValueError("Positive CPU worker count and non-negative seed required")
        self.factory = factory
        self.factory_kwargs = dict(factory_kwargs or {})
        self.seed = seed
        self.workers = workers
        self._counter = 0
        self._executor: ProcessPoolExecutor | None = None

    def seed_state(self) -> dict[str, int]:
        """Return the next ordered job seed cursor, not live executor state.

        Collection checkpoints must call this only after consuming all submitted
        work. In particular, an active ``imap`` prefetch cannot be resumed from
        this cursor: those submitted jobs and their results are not serialized.
        """
        return {"seed": self.seed, "workers": self.workers, "next_job": self._counter}

    def restore_seed_state(self, state: dict[str, int]) -> None:
        """Restore a compatible cursor before this pool submits any work."""
        if self._executor is not None or self._counter != 0:
            raise RuntimeError("Proposal seed state must be restored before submitting work")
        if (
            not isinstance(state, dict)
            or set(state) != {"seed", "workers", "next_job"}
            or any(type(value) is not int for value in state.values())
            or state["seed"] != self.seed
            or state["workers"] != self.workers
            or state["next_job"] < 0
        ):
            raise ValueError("Incompatible proposal seed state or invalid next_job cursor")
        self._counter = state["next_job"]

    def map(self, function: Callable, jobs: Iterable[tuple]) -> list:
        """Return results in input order, never completion/acceptance-speed order."""
        jobs = list(jobs)
        if not all(isinstance(job, tuple) for job in jobs):
            raise TypeError("Each proposal job must be a tuple of positional arguments")
        if not jobs:
            return []
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(self.factory, self.factory_kwargs),
            )
        futures = []
        # Spawn occurs during submit, before the child imports NumPy or Torch.
        with _worker_environment():
            for job in jobs:
                words = (
                    np.random.SeedSequence([self.seed, self._counter]).generate_state(4).tolist()
                )
                self._counter += 1
                futures.append(self._executor.submit(_run, function, words, job))
        return [future.result() for future in futures]

    def imap(self, function: Callable, jobs: Iterable[tuple], *, max_pending: int):
        """Yield in input order with bounded running/queued/completed lookahead.

        Workers immediately take another queued job, including future batches.
        Result order never depends on completion speed. Closing the iterator
        cancels work that has not started; active calls finish normally.
        """
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        iterator = iter(jobs)
        pending = deque()

        def submit_next():
            try:
                job = next(iterator)
            except StopIteration:
                return False
            if not isinstance(job, tuple):
                raise TypeError("Each proposal job must be a tuple of positional arguments")
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize,
                    initargs=(self.factory, self.factory_kwargs),
                )
            words = np.random.SeedSequence([self.seed, self._counter]).generate_state(4).tolist()
            self._counter += 1
            pending.append(self._executor.submit(_run, function, words, job))
            return True

        try:
            with _worker_environment():
                for _ in range(max_pending):
                    if not submit_next():
                        break
            while pending:
                result = pending[0].result()
                pending.popleft()
                with _worker_environment():
                    submit_next()
                yield result
        finally:
            for future in pending:
                future.cancel()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
