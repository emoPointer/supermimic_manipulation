"""Shared SuperMimic task adapters: one free object, one fixed target, an OSC arm.

The adapter snapshots physical POST state before automatic reset through a
read-only metrics hook; it never replaces the simulator or the control law. The
operational-space controller re-anchors at the current pose on every action, so
it has no memory to model. The world model predicts physical state increments
only; action history, clocks and task process counters are advanced analytically.
Imagined rewards and success are explicit functions of imagined states, never a
learned head. Contact, support-force and release tests have no exact analogue in
the reduced state, so each task documents its geometric surrogates; native real
rewards and evaluation remain authoritative.

A task subclasses ``RigidObjectMDP`` (success surrogate, reward, optional extra
observations and process fields) and ``RigidObjectAdapter`` (entity names,
contact geometry, native success term) and installs the capture hook.
"""

import math
from dataclasses import asdict

import torch

from clab.controllers.torch_kinematics import (
    axis_angle,
    quat_apply,
    quat_from_rotation_vector,
    quat_inverse,
    quat_mul,
    quat_to_six,
    relative_pose,
    serial_chain_fk,
    six_to_quat,
)
from clab.tensors import constant

CAPTURE_NAME = "supermimic_transition_capture"
# Controller-memory and bookkeeping fields advanced analytically, never predicted.
ANALYTIC_FIELDS = ("previous_action", "clock", "success_count")


class Fields:
    def __init__(self, widths):
        self.slices = {}
        self.widths = dict(widths)
        self.size = 0
        for name, width in widths:
            self.slices[name] = slice(self.size, self.size + width)
            self.size += width

    def get(self, values, name):
        return values[..., self.slices[name]]

    def pack(self, values):
        return torch.cat([values[name].flatten(1) for name in self.slices], -1)

    def contract(self):
        return {name: [part.start, part.stop] for name, part in self.slices.items()}


def physical_links(model, joint_ids, root_body, end_body):
    """Extract the compiled arm chain from the robot base to the end-effector body."""
    from clab.controllers.serial_chain import SerialLinkCfg

    if len(joint_ids) != 6:
        raise ValueError("Physical reward FK requires six explicitly ordered arm joints")
    links, parent = [], int(root_body)
    for joint_id in joint_ids:
        joint_id = int(joint_id)
        body = int(model.jnt_bodyid[joint_id])
        if (
            int(model.jnt_type[joint_id]) != 3  # MuJoCo hinge
            or int(model.body_parentid[body]) != parent
            or int(model.body_jntnum[body]) != 1
            or any(abs(float(value)) > 1e-10 for value in model.jnt_pos[joint_id])
            or abs(float(model.qpos0[model.jnt_qposadr[joint_id]])) > 1e-10
        ):
            raise ValueError(
                "Physical reward FK requires a serial zero-anchor, zero-ref hinge chain"
            )
        links.append(
            SerialLinkCfg(
                position=tuple(float(v) for v in model.body_pos[body]),
                quaternion=tuple(float(v) for v in model.body_quat[body]),
                axis=tuple(float(v) for v in model.jnt_axis[joint_id]),
                com=tuple(float(v) for v in model.body_ipos[body]),
                mass=float(model.body_mass[body]),
            )
        )
        parent = body
    if parent != int(end_body):
        raise ValueError("Physical reward FK does not end at the declared grasp body")
    return tuple(links)


def _limit_norm(value, limit):
    """Bound a vector's length like the OSC action, preserving its direction."""
    if limit is None:
        return value
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)
    return value * (limit / norm).clamp_max(1.0)


def _step_limit(velocity, dt):
    return None if velocity is None else velocity * dt


def install_capture(cfg):
    """Add a read-only POST capture hook; do not alter actions, RNG or autoreset."""
    from mjlab.managers.metrics_manager import MetricsTermCfg

    if not cfg.auto_reset:
        raise ValueError("SuperMimic capture expects native auto_reset=True")
    if CAPTURE_NAME in cfg.metrics:
        raise ValueError("SuperMimic transition capture is already installed")
    cfg.metrics[CAPTURE_NAME] = MetricsTermCfg(func=TransitionCapture, reduce="last")


class TransitionCapture:
    def __init__(self, cfg, env):
        del cfg
        if hasattr(env, "_clab_supermimic_capture"):
            raise ValueError("Duplicate SuperMimic transition capture")
        self.adapter = None
        self.state = None
        self.native_candidate = None
        self.step = -1
        env._clab_supermimic_capture = self

    def __call__(self, env):
        if self.adapter is None:
            raise RuntimeError("Construct the task adapter before stepping the capture-enabled env")
        self.state = self.adapter._capture_state()
        # Terminations ran before metrics and autoreset clears the predicate.
        self.native_candidate = self.adapter.native_candidate().clone()
        self.step = int(env.common_step_counter)
        return torch.zeros(env.num_envs, device=env.device)


class RigidObjectMDP:
    """Differentiable surrogate for one free object placed relative to a fixed target.

    Subclasses set ``task``, ``object_name``, ``target_name`` and the reward mode, and
    implement ``_candidate`` (one-step success surrogate) and ``reward``.
    """

    task = ""
    object_name = "object"
    target_name = "target"
    action_dim = 7
    gripper_action_index = 6  # Raw binary gripper: < 0 closed, >= 0 open.
    actor_widths = (6, 7, 8, 6, 6, 6)
    reward_matches_native = False
    reward_mode = ""
    # Analytic task fields beyond ANALYTIC_FIELDS (process flags, counters).
    process_fields = ()
    # Per-finger joint position while holding the object, and the opening that
    # counts as released (both from the task's grasped reset states).
    held_width = 0.0
    release_opening = 0.0
    # Ordered actor terms: object in target, previous action, joints, EE in base,
    # object in EE, target in EE, then ``extra_observation`` columns.

    def __init__(
        self,
        *,
        state_fields,
        context_fields,
        physical_links,
        grasp_offset,
        position_scale,
        orientation_scale,
        input_limit,
        linear_step_limit,
        angular_step_limit,
        closed_position,
        open_position,
        weights,
        dt,
        max_steps,
        required_steps,
        device="cpu",
        smooth_temperature=0.01,
        gripper_gradient_temperature=1.0,
        release_opening=None,
        release_temperature=0.002,
        success_gradient_scale=0.02,
        stable_linear_speed=0.03,
        stable_angular_speed=0.5,
        release_weight=0.2,
        held_width=None,
        action_bound_weight=0.0,
        gripper_bound=3.0,
        gripper_margin=1.0,
    ):
        release_opening = self.release_opening if release_opening is None else release_opening
        held_width = self.held_width if held_width is None else held_width
        if smooth_temperature <= 0 or not math.isfinite(smooth_temperature):
            raise ValueError("Predicate gradient temperature must be positive and finite")
        if gripper_gradient_temperature <= 0 or not math.isfinite(gripper_gradient_temperature):
            raise ValueError("Gripper gradient temperature must be positive and finite")
        if not all(
            math.isfinite(v) and v > 0
            for v in (
                release_opening,
                release_temperature,
                success_gradient_scale,
                stable_linear_speed,
                stable_angular_speed,
                release_weight,
            )
        ):
            raise ValueError("Release, stability and success-gradient settings must be positive")
        if release_opening <= held_width:
            raise ValueError("release_opening must exceed the held finger width")
        if not math.isfinite(action_bound_weight) or action_bound_weight < 0:
            raise ValueError("action_bound_weight must be finite and nonnegative")
        if not math.isfinite(gripper_bound) or gripper_bound <= 0:
            raise ValueError("gripper_bound must be positive and finite")
        if not math.isfinite(gripper_margin) or gripper_margin <= 0:
            raise ValueError("gripper_margin must be positive and finite")
        self.action_bound_weight = float(action_bound_weight)
        self.gripper_bound = float(gripper_bound)
        self.gripper_margin = float(gripper_margin)
        self.held_width = float(held_width)
        self.gripper_gradient_temperature = float(gripper_gradient_temperature)
        self.release_opening = float(release_opening)
        self.release_temperature = float(release_temperature)
        self.success_gradient_scale = float(success_gradient_scale)
        self.stable_linear_speed = float(stable_linear_speed)
        self.stable_angular_speed = float(stable_angular_speed)
        self.release_weight = float(release_weight)
        self.state = state_fields
        self.context = context_fields
        self.physical_links = physical_links
        self.grasp_offset = tuple(grasp_offset)
        self.position_scale, self.orientation_scale = (
            float(position_scale),
            float(orientation_scale),
        )
        self.closed_position, self.open_position = float(closed_position), float(open_position)
        self.input_limit = None if input_limit is None else float(input_limit)
        self.linear_step_limit = None if linear_step_limit is None else float(linear_step_limit)
        self.angular_step_limit = None if angular_step_limit is None else float(angular_step_limit)
        if (
            not all(
                math.isfinite(v) and v > 0 for v in (self.position_scale, self.orientation_scale)
            )
            or not all(math.isfinite(v) for v in (self.closed_position, self.open_position))
            or self.closed_position >= self.open_position
        ):
            raise ValueError(
                "Processed actions require finite positive scales and ordered gripper targets"
            )
        self.weights = dict(weights)
        self.dt, self.max_steps = float(dt), int(max_steps)
        self.required_steps = int(required_steps)
        self.smooth_temperature = float(smooth_temperature)
        self.obs_dim = 5 * sum(self.actor_widths)
        self.rotation_fields = (f"{self.object_name}_rotation",)
        self.auxiliary_fields = ("previous_action", "clock", "success_count", *self.process_fields)
        self.state_dim = self.state.size
        self.model_state = Fields(
            [
                (name, width)
                for name, width in self.state.widths.items()
                if name not in self.auxiliary_fields
            ]
        )
        self.model_state_dim = self.model_state.size
        analytic = {*ANALYTIC_FIELDS, *self.process_fields}
        self.delta = Fields(
            [
                (name, 3 if name in self.rotation_fields else width)
                for name, width in self.state.widths.items()
                if name not in analytic
            ]
        )
        self.delta_dim = self.delta.size
        self.context_dim = self.context.size
        self.input_dim = self.model_state_dim + self.action_dim + self.context_dim
        self.critic_dim = self.state_dim + self.context_dim + self.obs_dim
        self.error_floor = torch.full((self.delta_dim,), 1e-3, device=device)
        for name in ("robot_q", "robot_dq", f"{self.object_name}_velocity"):
            self.error_floor[self.delta.slices[name]] = 1e-2

    def obj(self, values, part):
        return self.state.get(values, f"{self.object_name}_{part}")

    def target(self, context, part):
        return self.context.get(context, f"{self.target_name}_{part}")

    def inputs(self, s, a, c):
        physical = self.model_state.pack(
            {name: self.state.get(s, name) for name in self.model_state.slices}
        )
        return torch.cat((physical, self.process_action(a), c), -1)

    def process_action(self, action):
        """Executed OSC command forward; explicit surrogate gripper gradient backward.

        Raw actions remain untouched for native execution, penalties and history.
        The pose follows the controller: per-axis input limit, scaling, then the
        per-step linear/angular norm bounds. The binary gripper's sigmoid
        straight-through derivative is not physical.
        """
        pose = action[:, :6]
        if self.input_limit is not None:
            pose = pose.clamp(-self.input_limit, self.input_limit)
        pose = pose * constant(action, [self.position_scale] * 3 + [self.orientation_scale] * 3)
        pose = torch.cat(
            (
                _limit_norm(pose[:, :3], self.linear_step_limit),
                _limit_norm(pose[:, 3:], self.angular_step_limit),
            ),
            -1,
        )
        raw_gripper = action[:, 6:7]
        hard = torch.where(raw_gripper < 0, self.closed_position, self.open_position)
        soft = self.closed_position + (self.open_position - self.closed_position) * torch.sigmoid(
            raw_gripper / self.gripper_gradient_temperature
        )
        gripper = hard + (soft - soft.detach())
        return torch.cat((pose, gripper), -1)

    def targets(self, s, s1):
        fields = {}
        for name in self.delta.slices:
            before, after = self.state.get(s, name), self.state.get(s1, name)
            if name in self.rotation_fields:
                fields[name] = axis_angle(
                    quat_mul(six_to_quat(after), quat_inverse(six_to_quat(before)))
                )
            else:
                fields[name] = after - before
        return self.delta.pack(fields)

    def state_error(self, pred, target):
        return self.targets(target, pred)

    def _gate(self, margin, scale=None):
        hard = (margin >= 0).to(margin.dtype)
        soft = torch.sigmoid(margin / (self.smooth_temperature if scale is None else scale))
        return hard + (soft - soft.detach())

    def _commands_open(self, a):
        """Native gripper decision (raw >= 0 opens) of ``a``, for a cost on opening.

        Forward: the binary decision. Backward: a hinge, constant while the raw
        command is above -gripper_margin and zero once it is closed by that margin.
        The cost corrects a wrong or marginal decision however saturated, and stops
        pushing a correct one: a sigmoid derivative vanishes on a saturated command,
        and one that never vanishes drives a correct command on without end.
        """
        raw = a[:, self.gripper_action_index]
        soft = (raw + self.gripper_margin).clamp_min(0) / (2 * self.gripper_margin)
        return (raw >= 0).to(raw.dtype) + (soft - soft.detach())

    def _commands_closed(self, a):
        """Native gripper decision (raw < 0 closes) of ``a``, for a cost on closing."""
        raw = a[:, self.gripper_action_index]
        soft = (self.gripper_margin - raw).clamp_min(0) / (2 * self.gripper_margin)
        return (raw < 0).to(raw.dtype) + (soft - soft.detach())

    def _band(self, value, low, high, scale):
        """Straight-through ``low <= value <= high`` with sigmoid edges of width ``scale``."""
        hard = ((value >= low) & (value <= high)).to(value.dtype)
        soft = torch.sigmoid((value - low) / scale) * torch.sigmoid((high - value) / scale)
        return hard + (soft - soft.detach())

    def _stable(self, s):
        velocity = self.obj(s, "velocity")
        return self._gate(self.stable_linear_speed - velocity[:, :3].norm(dim=-1)) * self._gate(
            self.stable_angular_speed - velocity[:, 3:].norm(dim=-1)
        )

    def _released(self, s):
        """Both finger joints opened past ``release_opening``; contact is not modelled."""
        fingers = self.state.get(s, "robot_q")[:, 6:8] - self.release_opening
        return self._gate(fingers, self.release_temperature).prod(-1)

    def _release_progress(self, s):
        """Mean finger opening fraction from the held width to ``release_opening``."""
        fingers = self.state.get(s, "robot_q")[:, 6:8]
        opening = (fingers - self.held_width) / (self.release_opening - self.held_width)
        return opening.clamp(0, 1).mean(-1)

    def _success_term(self, s, s1, c):
        """Counted success with the actor's straight-through derivative scaled.

        The forward value (critic targets keep the native terminal reward) is unchanged.
        """
        count = self._candidate(s1, c) * (self.state.get(s, "success_count").squeeze(-1) + 1)
        success = self._gate(count - self.required_steps, 0.5)
        return success.detach() + self.success_gradient_scale * (success - success.detach())

    def _costs(self, s, a, s1):
        g = self.state.get
        return {
            "action_magnitude": a.square().sum(-1).clamp_max(1e4),
            "action_rate": (a - g(s, "previous_action")).square().sum(-1).clamp_max(1e4),
            "joint_vel": g(s1, "robot_dq")[:, :6].square().sum(-1).clamp_max(1e4),
        }

    def action_bound_cost(self, a):
        """Imagination-only per-step cost of raw commands outside the responsive range.

        Pose axes past ``input_limit`` are clipped and the binary gripper's sigmoid
        derivative vanishes far from zero, so an actor that drifts there receives no
        gradient and its decision freezes. ``action_bound_weight`` (per second) pulls
        it back; 0 disables the cost.
        """
        index = self.gripper_action_index
        excess = (a[:, index].abs() - self.gripper_bound).clamp_min(0).square()
        if self.input_limit is not None:
            pose = (a[:, :index].abs() - self.input_limit).clamp_min(0).square()
            excess = excess + pose.sum(-1)
        return self.action_bound_weight * excess * self.dt

    def _timeout(self, s1):
        return (self.state.get(s1, "clock").squeeze(-1) >= self.max_steps).to(s1.dtype)

    def _candidate(self, s, c):
        raise NotImplementedError

    def reward(self, s, a, s1, c):
        raise NotImplementedError

    def physical_grasp(self, s, c):
        position, quaternion = serial_chain_fk(
            self.state.get(s, "robot_q")[:, :6], self.physical_links
        )
        position = position + quat_apply(
            quaternion, constant(s, self.grasp_offset).expand_as(position)
        )
        base_quat = six_to_quat(self.context.get(c, "base_rotation"))
        return self.context.get(c, "base_position") + quat_apply(base_quat, position)

    def integrate(self, s, delta, a, c):
        fields = {name: self.state.get(s, name) for name in self.state.slices}
        for name in self.delta.slices:
            increment = self.delta.get(delta, name)
            if name in self.rotation_fields:
                fields[name] = quat_to_six(
                    quat_mul(quat_from_rotation_vector(increment), six_to_quat(fields[name]))
                )
            else:
                fields[name] = fields[name] + increment
        fields["previous_action"] = a
        fields["clock"] = self.state.get(s, "clock") + 1
        self.advance_task_fields(fields, s, c)
        return self.state.pack(fields)

    def advance_task_fields(self, fields, s, c):
        """Advance process fields, then the success counter, from the provisional state."""
        provisional = self.state.pack(fields)
        fields["success_count"] = self._candidate(provisional, c)[:, None] * (
            self.state.get(s, "success_count") + 1
        )

    def observe(self, s, c):
        g, h = self.state.get, self.context.get
        joints = g(s, "robot_q")
        ee_local, ee_quat_local = serial_chain_fk(joints[:, :6], self.physical_links)
        base_pos, base_quat = h(c, "base_position"), six_to_quat(h(c, "base_rotation"))
        ee_pos = base_pos + quat_apply(base_quat, ee_local)
        ee_quat = quat_mul(base_quat, ee_quat_local)
        obj_pos, obj_quat = self.obj(s, "position"), six_to_quat(self.obj(s, "rotation"))
        target_pos = self.target(c, "position")
        target_quat = six_to_quat(self.target(c, "rotation"))
        return torch.cat(
            (
                relative_pose(obj_pos, obj_quat, target_pos, target_quat),
                g(s, "previous_action"),
                joints,
                relative_pose(ee_pos, ee_quat, base_pos, base_quat),
                relative_pose(obj_pos, obj_quat, ee_pos, ee_quat),
                relative_pose(target_pos, target_quat, ee_pos, ee_quat),
                *self.extra_observation(s, c),
            ),
            -1,
        )

    def extra_observation(self, s, c):
        """Actor terms after the six common ones, in order; widths extend actor_widths."""
        return ()

    def advance_obs(self, obs, s1, a, c):
        del a  # integrate/capture already stores the issued raw action in s1.
        current = self.observe(s1, c)
        histories, old_offset, current_offset = [], 0, 0
        for width in self.actor_widths:
            history = obs[:, old_offset : old_offset + 5 * width].reshape(-1, 5, width)
            value = current[:, current_offset : current_offset + width]
            histories.append(torch.cat((history[:, 1:], value[:, None]), 1).flatten(1))
            old_offset += 5 * width
            current_offset += width
        return torch.cat(histories, -1)

    def critic_features(self, s, c, obs):
        return torch.cat((s, c, obs), -1)

    def task_contract(self):
        """Task-specific surrogate description merged into ``contract``."""
        return {}

    def contract(self):
        widths = ",".join(str(width) for width in self.actor_widths)
        return {
            "task": self.task,
            "state_fields": self.state.contract(),
            "context_fields": self.context.contract(),
            "state_dim": self.state_dim,
            "model_state_fields": self.model_state.contract(),
            "model_state_dim": self.model_state_dim,
            "auxiliary_state_fields": list(self.auxiliary_fields),
            "input_dim": self.input_dim,
            "delta_dim": self.delta_dim,
            "delta_fields": self.delta.contract(),
            "rotation_integration": "six-dimensional state representation; three-dimensional world-axis SO(3) tangent delta; log(R_next R_prev^-1) target, exp(delta) R_prev integration",
            "analytic_state_fields": [*ANALYTIC_FIELDS, *self.process_fields],
            "context_dim": self.context_dim,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "world_model_action": {
                "pose": "first six raw axes clipped to the input limit, scaled, then norm-bounded per step, in world axes",
                "position_scale": self.position_scale,
                "orientation_scale": self.orientation_scale,
                "input_limit": self.input_limit,
                "linear_step_limit": self.linear_step_limit,
                "angular_step_limit": self.angular_step_limit,
                "gripper": "raw<0 closed, raw>=0 open; actual commanded finger position",
                "closed_position": self.closed_position,
                "open_position": self.open_position,
                "pose_gradient": "true clip and norm-bound derivatives",
                "gripper_gradient": "hard forward; sigmoid straight-through backward",
                "gripper_gradient_temperature": self.gripper_gradient_temperature,
                "raw_action_consumers": "native controller, action penalties and actor history",
            },
            "coordinates": "physical joints; free/mocap positions relative to env origin in world axes; rotations first-two matrix columns",
            "actor_history": f"terms of widths {widths}; each has five oldest-to-newest frames",
            "transition": "20 Hz policy; operational-space control at the native physics rate",
            "reward_mode": self.reward_mode,
            "reward_matches_native": self.reward_matches_native,
            "reward_timing": "predicted endpoint physical state; no native pre-forward lagged witnesses",
            "reward_source": "explicit functions of imagined state and raw actions; no learned reward head",
            "stable_linear_speed": self.stable_linear_speed,
            "stable_angular_speed": self.stable_angular_speed,
            "release_shaping_held_width": self.held_width,
            "release_weight": self.release_weight,
            "release_opening": self.release_opening,
            "release_temperature": self.release_temperature,
            "success_gradient_scale": self.success_gradient_scale,
            "action_bound": "imagination-only cost action_bound_weight x squared excess of the raw gripper past gripper_bound and of raw pose axes past input_limit, per second",
            "action_bound_weight": self.action_bound_weight,
            "gripper_bound": self.gripper_bound,
            "gripper_margin": self.gripper_margin,
            "gripper_decision_costs": "native binary decision forward; hinge derivative backward, constant until the raw command is past the decision boundary by gripper_margin, zero beyond",
            "success_counter_initialization": "native counter at replay start; subsequent imagined increments use the surrogate",
            "reward_weights": self.weights,
            "reward_dt": self.dt,
            "max_steps": self.max_steps,
            "required_steps": self.required_steps,
            "predicate_gradient": "hard surrogate forward; sigmoid straight-through backward",
            "smooth_temperature": self.smooth_temperature,
            "success_gradient_temperature": 0.5,
            "model_limitations": "reduced physical transition without solver warmstart; endpoint reward/success approximation does not certify native contact/release/support or exact physics",
            "physical_links": [asdict(link) for link in self.physical_links],
            "physical_grasp_offset": self.grasp_offset,
            "binding": getattr(self, "binding_contract", None),
            **self.task_contract(),
        }


class RigidObjectAdapter:
    """Bind a task surrogate to one already-constructed live environment.

    Subclasses set the class attributes below and implement ``make_mdp``; tasks with
    process state add ``process_state`` fields and capture them in ``capture_process``.
    """

    mdp_class = RigidObjectMDP
    object_entity = "object"
    target_entity = "target"
    # Body of the target entity whose pose the actor observes (None: entity root).
    target_body = None
    object_geoms = ()
    target_geoms = ()
    actor_terms = ()
    reward_terms = frozenset()

    def __init__(self, env, *, smooth_temperature=0.01, capture=True, **imagination):
        from clab.controllers.osc import OscActionCfg

        robot_binding = self.robot_binding()
        self.env = env
        self.observation_rotation_coordinate_max_error = 0.0
        name = self.mdp_class.task
        self.action_cfg = env.cfg.actions.get(robot_binding.action_name)
        if not isinstance(self.action_cfg, OscActionCfg):
            raise ValueError(f"{name} SuperMimic requires the relative OSC action")
        if (capture and not env.cfg.auto_reset) or not env.cfg.is_finite_horizon:
            raise ValueError(f"{name} requires native autoreset and finite-horizon semantics")
        if not math.isclose(env.step_dt, robot_binding.control_dt):
            raise ValueError(
                f"{name} SuperMimic expects the robot's {robot_binding.control_dt} s step"
            )
        if tuple(env.observation_manager.active_terms["actor"]) != tuple(self.actor_terms):
            raise ValueError(f"{name} actor term order differs from the declared contract")
        self.capture = getattr(env, "_clab_supermimic_capture", None) if capture else None
        if capture and self.capture is None:
            raise ValueError("Call install_capture(cfg) before constructing the environment")
        model = env.sim.mj_model
        self.robot = env.scene["robot"]
        self.origin = env.scene.env_origins
        self.joint_ids, _ = self.robot.find_joints(robot_binding.joints, preserve_order=True)
        self.q_ids = self.robot.indexing.joint_q_adr[self.joint_ids]
        self.v_ids = self.robot.indexing.joint_v_adr[self.joint_ids]
        entity = env.scene[self.object_entity]
        # Entity indexing, not a guessed qpos offset, identifies its free joint.
        body = int(entity.indexing.root_body_id)
        jid = int(model.body_jntadr[body])
        if int(model.jnt_type[jid]) != 0:
            raise ValueError(f"{self.object_entity} must have exactly the native free-joint root")
        self.object_q = int(model.jnt_qposadr[jid])
        self.object_v = int(model.jnt_dofadr[jid])
        self.ee_body = model.body(f"robot/{robot_binding.ee_body}").id
        self.success = env.termination_manager.get_term_cfg("success").func
        bodies = [f"robot/{name}" for name in robot_binding.rigid_bodies] + [
            *self.task_bodies(),
            "table/table",
        ]
        self.body_ids = [model.body(name).id for name in bodies]
        geom_names = (
            [f"robot/{name}" for name in robot_binding.collision_geoms]
            + list(self.object_geoms)
            + list(self.target_geoms)
            + ["table/table_collision"]
        )
        self.geom_ids = [model.geom(name).id for name in geom_names]
        obj = self.mdp_class.object_name
        state = Fields(
            [
                ("robot_q", 8),
                ("robot_dq", 8),
                (f"{obj}_position", 3),
                (f"{obj}_rotation", 6),
                (f"{obj}_velocity", 6),
                *self.extra_physical_state(),
                ("previous_action", 7),
                ("clock", 1),
                ("success_count", 1),
                *self.process_state(),
            ]
        )
        target = self.mdp_class.target_name
        context = Fields(
            [
                ("base_position", 3),
                ("base_rotation", 6),
                (f"{target}_position", 3),
                (f"{target}_rotation", 6),
                ("origin_z", 1),
                ("gravity", 3),
                ("body_mass", len(bodies)),
                ("body_inertia", 3 * len(bodies)),
                ("geom_friction", 3 * len(geom_names)),
                ("dof_armature", 8),
                ("dof_damping", 8),
                ("dof_frictionloss", 8),
            ]
        )
        self.state, self.context = state, context
        if set(env.cfg.rewards) != set(self.reward_terms):
            raise ValueError(f"Unexpected {name} reward terms; analytic MDP must be updated")
        if not env.cfg.scale_rewards_by_dt:
            raise ValueError(f"{name} analytic reward expects native dt scaling")
        self.mdp = self.mdp_class(
            state_fields=state,
            context_fields=context,
            physical_links=physical_links(
                model,
                [model.joint(f"robot/{name}").id for name in robot_binding.arm_joints],
                model.body(f"robot/{robot_binding.root_body}").id,
                self.ee_body,
            ),
            grasp_offset=robot_binding.grasp_offset,
            position_scale=self.action_cfg.position_scale,
            orientation_scale=self.action_cfg.orientation_scale,
            input_limit=self.action_cfg.input_limit,
            linear_step_limit=_step_limit(self.action_cfg.linear_velocity_limit, env.step_dt),
            angular_step_limit=_step_limit(self.action_cfg.angular_velocity_limit, env.step_dt),
            closed_position=self.action_cfg.closed_position,
            open_position=self.action_cfg.open_position,
            weights={name: term.weight for name, term in env.cfg.rewards.items()},
            dt=env.step_dt,
            max_steps=env.max_episode_length,
            required_steps=self.required_steps(),
            device=env.device,
            smooth_temperature=smooth_temperature,
            **imagination,
        )
        if self.mdp.obs_dim != env.observation_manager.group_obs_dim["actor"][0]:
            raise ValueError(f"{name} actor width differs from the declared surrogate layout")
        self.body_names, self.geom_names = bodies, geom_names
        self.mdp.binding_contract = {
            "physical_body_names": bodies,
            "physical_geom_names": geom_names,
            "controller": asdict(self.action_cfg),
        }
        if capture:
            self.capture.adapter = self

    @staticmethod
    def robot_binding():
        raise NotImplementedError

    def task_bodies(self):
        """Task body names whose mass and inertia enter the context, after the robot."""
        return [f"{self.object_entity}/{self.object_entity}"]

    def extra_physical_state(self):
        """Predicted physical fields beyond robot and object (e.g. a passive joint)."""
        return ()

    def process_state(self):
        """Analytic task-process fields, appended after the success counter."""
        return ()

    def required_steps(self):
        return self.success.required_steps

    def native_candidate(self):
        return self.success.candidate

    def native_success_count(self):
        return self.success.counter.float()[:, None]

    def capture_extra(self):
        """Values for ``extra_physical_state`` and ``process_state`` fields."""
        return {}

    def target_pose(self):
        data = self.env.scene[self.target_entity].data
        if self.target_body is None:
            return data.root_link_pos_w, data.root_link_quat_w
        index = self.env.scene[self.target_entity].find_bodies(self.target_body)[0][0]
        return data.body_link_pos_w[:, index], data.body_link_quat_w[:, index]

    def _capture_state(self):
        env = self.env
        qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
        obj = self.mdp.object_name
        fields = {
            "robot_q": qpos[:, self.q_ids],
            "robot_dq": qvel[:, self.v_ids],
            f"{obj}_position": qpos[:, self.object_q : self.object_q + 3] - self.origin,
            f"{obj}_rotation": quat_to_six(qpos[:, self.object_q + 3 : self.object_q + 7]),
            f"{obj}_velocity": qvel[:, self.object_v : self.object_v + 6],
            "previous_action": env.action_manager.action,
            "clock": env.episode_length_buf.float()[:, None],
            "success_count": self.native_success_count(),
            **self.capture_extra(),
        }
        return self.state.pack(fields).detach().clone()

    def _capture_context(self):
        env, physical = self.env, self.env.sim.model
        robot = env.scene["robot"].data
        target_pos, target_quat = self.target_pose()
        target = self.mdp.target_name
        return (
            self.context.pack(
                {
                    "base_position": robot.root_link_pos_w - self.origin,
                    "base_rotation": quat_to_six(robot.root_link_quat_w),
                    f"{target}_position": target_pos - self.origin,
                    f"{target}_rotation": quat_to_six(target_quat),
                    "origin_z": self.origin[:, 2:3],
                    "gravity": physical.opt.gravity[:],
                    "body_mass": physical.body_mass[:, self.body_ids],
                    "body_inertia": physical.body_inertia[:, self.body_ids],
                    "geom_friction": physical.geom_friction[:, self.geom_ids],
                    "dof_armature": physical.dof_armature[:, self.v_ids],
                    "dof_damping": physical.dof_damping[:, self.v_ids],
                    "dof_frictionloss": physical.dof_frictionloss[:, self.v_ids],
                }
            )
            .detach()
            .clone()
        )

    def read(self):
        return self._capture_state(), self._capture_context()

    def post_step(self):
        if (
            self.capture is None
            or self.capture.step != int(self.env.common_step_counter)
            or self.capture.state is None
        ):
            raise RuntimeError("No complete pre-autoreset capture for this environment step")
        return self.capture.state.clone()

    def candidate_comparison(self, post, context):
        """Per-step surrogate versus native success predicates on the same real POST.

        Unlike the reward comparison, this does not depend on the native success
        counter, so a surrogate that credits states native success rejects is visible.
        """
        if self.capture is None or self.capture.native_candidate is None:
            raise RuntimeError("No captured native success predicate for this environment step")
        return self.mdp._candidate(
            post, context
        ).detach() > 0.5, self.capture.native_candidate.clone()

    def validate_observation(self, state, context, observations, *, atol=1e-5):
        """Check physical rotations and scalar features without rewriting observations.

        Rotation-vector coordinates are discontinuous at pi. Their raw mismatch
        remains visible separately; geometric agreement is not MLP-input equality.
        """
        name = self.mdp_class.task
        if observations.shape != (state.shape[0], self.mdp.obs_dim):
            raise ValueError(f"Expected native term-major {self.mdp.obs_dim}-dimensional actor")
        newest, offset = [], 0
        for width in self.mdp.actor_widths:
            newest.append(observations[:, offset : offset + 5 * width].reshape(-1, 5, width)[:, -1])
            offset += 5 * width
        reference = torch.cat(newest, -1)
        predicted = self.mdp.observe(state, context)
        if not bool(torch.isfinite(predicted).all() & torch.isfinite(reference).all()):
            raise RuntimeError(f"Non-finite analytic/native {name} actor observation")
        rotation_slices = (slice(3, 6), slice(24, 27), slice(30, 33), slice(36, 39))
        predicted_rotations = torch.stack([predicted[:, part] for part in rotation_slices], 1)
        reference_rotations = torch.stack([reference[:, part] for part in rotation_slices], 1)
        raw_rotvec_max = float((predicted_rotations - reference_rotations).abs().max())
        self.observation_rotation_coordinate_max_error = max(
            getattr(self, "observation_rotation_coordinate_max_error", 0.0), raw_rotvec_max
        )
        scalar_error = (predicted - reference).abs()
        for part in rotation_slices:
            scalar_error[:, part] = 0
        linear_max = float(scalar_error.max())
        relative = quat_mul(
            quat_from_rotation_vector(predicted_rotations),
            quat_inverse(quat_from_rotation_vector(reference_rotations)),
        )
        relative = torch.nn.functional.normalize(relative, dim=-1)
        angle = 2 * torch.atan2(
            torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., 0].abs()
        )
        rotation_angle_max = float(angle.max())
        canonical = bool(
            (torch.linalg.vector_norm(predicted_rotations, dim=-1) <= math.pi + atol).all()
            & (torch.linalg.vector_norm(reference_rotations, dim=-1) <= math.pi + atol).all()
        )
        detail = (
            f"linear_max={linear_max:.9g}, rotation_angle_max={rotation_angle_max:.9g}, "
            f"raw_rotvec_max={raw_rotvec_max:.9g}, tolerance={atol}"
        )
        if not canonical or not bool(torch.isfinite(angle).all()):
            raise RuntimeError(f"Noncanonical/non-finite {name} observation rotation: {detail}")
        error = max(linear_max, rotation_angle_max)
        if error > atol:
            raise RuntimeError(f"Analytic/native {name} actor observation mismatch: {detail}")
        return error

    def contract(self):
        return {
            **self.mdp.contract(),
            "physical_body_names": self.body_names,
            "physical_geom_names": self.geom_names,
            "controller": asdict(self.action_cfg),
        }
