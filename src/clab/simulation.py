"""Simulation helpers shared by the training workflows."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import types
from contextlib import contextmanager

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim_data import TorchArray, WarpBridge, _contains_warp_arrays

from clab.cuda_graphs import SegmentedGraph

MODES = ("auto", "plain", "substeps")
# mujoco_warp OverflowType.CONTACT_MATCH: a contact sensor matched more contacts
# than its storage. Extra matches are dropped (the sensor still reports a found
# contact) and physics is unchanged. Every other overflow bit drops contacts or
# constraints, so that world's physics is no longer trustworthy.
CONTACT_MATCH_OVERFLOW = 1 << 6
# ManagerBasedRlEnv.step of mjlab 1.5.3, which _step_with_substep_graph mirrors.
_MJLAB_STEP_SHA256 = "5608d76da560356fc8be7d95dd11a49c564c67eee0ad5520cee7e57b89ce6a14"


def configure_cuda_graph(env, mode: str) -> dict:
    """Keep mjlab's automatic CUDA Graph selection, or capture plain physics graphs.

    ``plain`` disables conditional graphs (for older drivers), compiles the step and
    lazily loaded reset kernels outside capture, then captures step/forward/reset.
    ``substeps`` also captures the whole decimation loop of ``env.step`` (actions,
    actuators, physics and sensor updates; see :func:`capture_substeps`).
    Callers reset the environment afterwards, discarding the warmup state.
    """
    if mode not in MODES:
        raise ValueError(f"Unsupported CUDA Graph mode: {mode}")
    sim = env.sim
    if mode in ("plain", "substeps"):
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
    substeps = capture_substeps(env) if mode == "substeps" else None
    return {
        "enabled": bool(sim.use_cuda_graph),
        "graph_conditional": bool(sim.wp_model.opt.graph_conditional),
        "step_graph_created": sim.step_graph is not None,
        "forward_graph_created": sim.forward_graph is not None,
        "reset_graph_created": sim.reset_graph is not None,
        "substep_graph": substeps,
    }


def capture_substeps(env) -> dict:
    """Replay the decimation loop of ``env.step`` from CUDA graphs.

    Per policy step the loop runs ``decimation`` times: action terms (the OSC),
    actuator controls, ``mujoco_warp.step`` and entity/sensor updates, thousands
    of small kernel launches. One substep is captured, as graph segments around
    the eager eigendecompositions of :func:`clab.cuda_graphs.graph_break`, and
    ``env.step`` replays it ``decimation`` times; the kernels, memory and order
    are unchanged. (Capturing all substeps would keep every substep's
    ``mujoco_warp`` scratch allocations alive at once.) Host-side effects of the
    loop (the substep counter, sensor cache invalidation) are applied after the
    replays. Physics advances by one policy step and one substep during capture;
    callers reset afterwards.
    """
    _check_capturable(env)
    sim = env.sim
    histories = [
        sensor._history_state
        for sensor in env.scene.sensors.values()
        if getattr(sensor, "_history_state", None) is not None
    ]

    def substep():
        # Contact history rolls into a new tensor every substep; a graph would keep
        # reading the tensor from capture time, so copy the result back into it.
        start = [dict(history) for history in histories]
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        sim.step()
        env.scene.update(dt=env.physics_dt)
        env.metrics_manager.compute_substep()
        for history, tensors in zip(histories, start, strict=True):
            for key, tensor in tensors.items():
                tensor.copy_(history[key])
                history[key] = tensor

    def loop():
        for _ in range(env.cfg.decimation):
            substep()

    graph = SegmentedGraph(env.device)
    for bridge in (sim.data, sim.model):
        _wrap_arrays(bridge)
    before = _tensor_addresses(env)
    with _capturable_writes(sim):
        # Loads every kernel and initializes cuSOLVER before capture.
        graph.run_eagerly(loop)
        graph.capture(substep)
    torch.cuda.synchronize(env.device)
    warp_stream = wp.get_stream(sim.wp_device).cuda_stream
    for bridge in (sim.data, sim.model):
        _check_arrays(bridge, warp_stream)
    after = _tensor_addresses(env)
    moved = sorted(path for path, address in before.items() if after.get(path) != address)
    if moved:
        raise RuntimeError(f"The decimation loop rebinds tensors a graph cannot follow: {moved}")
    env._clab_substep_graph = graph
    env._clab_warp_stream = torch.cuda.ExternalStream(
        wp.get_stream(sim.wp_device).cuda_stream, device=env.device
    )
    env.step = types.MethodType(_step_with_substep_graph, env)
    for sensor in env.scene.sensors.values():
        sensor._invalidate_cache()
    print(
        f"Substep CUDA Graph active ({graph.graphs} segments per substep, "
        f"{env.cfg.decimation} substeps per policy step)",
        flush=True,
    )
    return {
        "segments_per_substep": graph.graphs,
        "eager_per_substep": len(graph.steps) - graph.graphs,
    }


def _check_capturable(env):
    if type(env).step is not ManagerBasedRlEnv.step:
        raise ValueError("Substep graphs mirror ManagerBasedRlEnv.step; this env overrides it")
    source = inspect.getsource(ManagerBasedRlEnv.step)
    if hashlib.sha256(source.encode()).hexdigest() != _MJLAB_STEP_SHA256:
        raise ValueError("mjlab's ManagerBasedRlEnv.step changed; update _step_with_substep_graph")
    if env.sim.nan_guard.enabled:
        raise ValueError("The NaN guard copies state on the host every physics step")
    if getattr(env.metrics_manager, "_substep_term_indices", None):
        raise ValueError("Per-substep metrics keep host-side counters")
    for name, entity in env.scene.entities.items():
        for actuator in entity._actuators:
            if getattr(actuator, "_delay_buffer", None) is not None:
                raise ValueError(f"{name}: delayed actuators are not supported in substep graphs")


def _tensor_addresses(env) -> dict:
    """Device addresses of the tensors the decimation loop's objects hold.

    A captured graph reads and writes fixed addresses; an attribute rebound to a
    new tensor inside the loop would silently fall out of the replay.
    """
    addresses = {}

    def visit(path, value, depth):
        if torch.is_tensor(value):
            addresses[path] = value.data_ptr()
        elif depth < 3 and isinstance(value, dict):
            for key, item in value.items():
                visit(f"{path}[{key!r}]", item, depth + 1)
        elif depth < 3 and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(f"{path}[{index}]", item, depth + 1)
        elif depth < 3 and dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(f"{path}.{field.name}", getattr(value, field.name), depth + 1)

    def members(path, owner):
        for name, value in vars(owner).items():
            visit(f"{path}.{name}", value, 1)

    for name, term in env.action_manager._terms.items():
        members(f"action:{name}", term)
    for name, entity in env.scene.entities.items():
        members(f"entity:{name}.data", entity.data)
        for index, actuator in enumerate(entity._actuators):
            members(f"entity:{name}.actuator[{index}]", actuator)
    for name, sensor in env.scene.sensors.items():
        members(f"sensor:{name}", sensor)
    return addresses


def _wrap_arrays(bridge):
    """Build mjlab's lazily created TorchArray wrappers now, outside any capture.

    A wrapper records Warp's current stream when first built and writes on it.
    Built inside the capture's scoped stream, it would keep writing on that
    non-blocking stream afterwards, unordered with the default stream (writes
    read index tensors the default stream had already reused).
    """
    struct = bridge.struct
    names = (
        [field.name for field in dataclasses.fields(struct)]
        if dataclasses.is_dataclass(struct)
        else list(vars(struct))
    )
    for name in names:
        value = getattr(struct, name)
        if isinstance(value, wp.array) or _contains_warp_arrays(value):
            child = getattr(bridge, name)
            if isinstance(child, WarpBridge):
                _wrap_arrays(child)


def _check_arrays(bridge, warp_stream):
    for name, wrapped in bridge._wrapped_cache.items():
        if isinstance(wrapped, WarpBridge):
            _check_arrays(wrapped, warp_stream)
        elif (
            isinstance(wrapped, TorchArray)
            and wrapped._torch_stream is not None
            and wrapped._torch_stream.cuda_stream != warp_stream
        ):
            raise RuntimeError(f"Simulation array {name} was wrapped on the capture stream")


@contextmanager
def _capturable_writes(sim):
    """During capture: eager physics kernels, and TorchArray writes on the capture stream.

    TorchArray writes on Warp's stream; captured, they must be issued on the
    capture stream, in the same order relative to the kernels that read them.
    """
    setitem = TorchArray.__setitem__

    def direct(self, idx, value):
        self._tensor[idx] = value

    TorchArray.__setitem__ = direct
    sim.use_cuda_graph = False
    try:
        yield
    finally:
        sim.use_cuda_graph = True
        TorchArray.__setitem__ = setitem


def _replay_substeps(env):
    stream = torch.cuda.current_stream(env.device)
    stream.wait_stream(env._clab_warp_stream)
    for _ in range(env.cfg.decimation):
        env._clab_substep_graph.replay()
    env._clab_warp_stream.wait_stream(stream)
    env._sim_step_counter += env.cfg.decimation
    for sensor in env.scene.sensors.values():
        sensor._invalidate_cache()


def _step_with_substep_graph(self, action):
    """``ManagerBasedRlEnv.step`` (mjlab 1.5.3) with the decimation loop replayed."""
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
        pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
        raise RuntimeError(
            f"Environments {pending_ids.cpu().tolist()} must be reset via "
            "reset(env_ids=...) before calling step() again when auto_reset=False."
        )

    self.extras["log"] = dict()
    self.action_manager.process_action(action.to(self.device))

    _replay_substeps(self)

    self.episode_length_buf += 1
    self.common_step_counter += 1

    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
        self.recorder_manager.record_pre_reset(reset_env_ids)
        self._reset_idx(reset_env_ids)
        self.scene.write_data_to_sim()

    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
        self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
        self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
        self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
        self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
        self.obs_buf,
        self.reward_buf,
        self.reset_terminated,
        self.reset_time_outs,
        self.extras,
    )
