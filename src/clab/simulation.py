"""Simulation helpers shared by the training workflows."""

from __future__ import annotations

import warp as wp


def configure_cuda_graph(env, mode: str) -> dict:
    """Keep mjlab's automatic CUDA Graph selection, or capture plain physics graphs.

    ``plain`` disables conditional graphs (for older drivers), compiles the step and
    lazily loaded reset kernels outside capture, then captures step/forward/reset.
    Callers reset the environment afterwards, discarding the warmup state.
    """
    if mode not in ("auto", "plain"):
        raise ValueError(f"Unsupported CUDA Graph mode: {mode}")
    sim = env.sim
    if mode == "plain":
        if not sim.wp_device.is_cuda:
            raise ValueError("Plain CUDA Graph mode requires a CUDA device")
        sim.wp_model.opt.graph_conditional = False
        sim.reset()
        sim.step()
        wp.synchronize_device(sim.wp_device)
        sim.use_cuda_graph = True
        sim.create_graph()
        if any(getattr(sim, f"{name}_graph") is None for name in ("step", "forward", "reset")):
            raise RuntimeError("Plain CUDA Graph capture did not create step/forward/reset graphs")
        print("Plain CUDA Graph active (fixed-structure physics)", flush=True)
    return {
        "enabled": bool(sim.use_cuda_graph),
        "graph_conditional": bool(sim.wp_model.opt.graph_conditional),
        "step_graph_created": sim.step_graph is not None,
        "forward_graph_created": sim.forward_graph is not None,
        "reset_graph_created": sim.reset_graph is not None,
    }
