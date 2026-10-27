"""CUDA graphs captured in segments around operations that cannot be captured.

``graph_break(fn, *inputs)`` marks such an operation (cuSOLVER's batched
eigendecomposition on this driver, for example). Outside a capture it simply
calls ``fn``. Inside :meth:`SegmentedGraph.capture` it closes the open graph,
runs ``fn`` eagerly and opens the next graph; replay then runs every graph and
every eager operation in their original order, on the same memory, so results
are bitwise those of the uncaptured code.
"""

from __future__ import annotations

import gc

import torch
import warp as wp

_capture: SegmentedGraph | None = None


def graph_break(fn, *inputs):
    """``fn(*inputs)``, run eagerly between graph segments while a capture is open.

    ``fn`` must return a tuple of tensors computed only from ``inputs``. During a
    capture they become static tensors that every replay overwrites in place.
    """
    if _capture is None:
        return fn(*inputs)
    return _capture.eager(fn, inputs)


class SegmentedGraph:
    """CUDA graphs and eager operations, replayed in order on the current stream.

    Torch kernels, and Warp kernels through an external Warp capture, issued on
    the capture stream are recorded; all segments share one memory pool. Each
    segment is replayed right after it is captured, so the code being captured
    runs once for real and every eager operation sees real inputs.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(self.device)
        self.warp_stream = wp.stream_from_torch(self.stream)
        self.pool = torch.cuda.graph_pool_handle()
        self.steps = []
        self.graphs = 0
        self._graph = None
        # Warp ties allocations made during a capture to its graph object.
        self._warp_graphs = []

    def run_eagerly(self, body):
        """Run ``body`` uncaptured on the capture stream (warm-up)."""
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with (
            torch.cuda.stream(self.stream),
            wp.ScopedStream(self.warp_stream, sync_enter=False, sync_exit=False),
        ):
            body()
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def capture(self, body):
        global _capture
        if _capture is not None or self.steps:
            raise RuntimeError("SegmentedGraph captures once, one capture at a time")
        torch.cuda.synchronize(self.device)
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with (
                torch.cuda.stream(self.stream),
                wp.ScopedStream(self.warp_stream, sync_enter=False, sync_exit=False),
            ):
                _capture = self
                self._open()
                body()
                self._close()
        except BaseException:
            if self._graph is not None:
                self._abort()
            raise
        finally:
            _capture = None
            if enabled:
                gc.enable()
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def replay(self):
        for step in self.steps:
            step()

    def eager(self, fn, inputs):
        self._close()
        outputs = tuple(fn(*inputs))

        def run():
            for static, value in zip(outputs, fn(*inputs), strict=True):
                static.copy_(value)

        self.steps.append(run)
        self._open()
        return outputs

    def _open(self):
        self._graph = torch.cuda.CUDAGraph()
        self._graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
        wp.capture_begin(stream=self.warp_stream, external=True)

    def _close(self):
        self._warp_graphs.append(wp.capture_end(stream=self.warp_stream))
        graph, self._graph = self._graph, None
        graph.capture_end()
        graph.replay()
        self.steps.append(graph.replay)
        self.graphs += 1

    def _abort(self):
        graph, self._graph = self._graph, None
        try:
            wp.capture_end(stream=self.warp_stream)
        except RuntimeError:
            pass
        try:
            graph.capture_end()
        except RuntimeError:
            pass
