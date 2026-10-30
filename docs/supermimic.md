# SuperMimic

SuperMimic trains a policy by backpropagating the task return through a learned,
differentiable world model instead of using policy-gradient estimates. The implementation
in `src/clab/algorithms/supermimic/` adapts the open-source
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) code (MIT license kept in
`REFERENCE_LICENSE`) to state-based manipulation; it does not claim to reproduce the
original paper's experiments.

The algorithm is task-agnostic. Everything task-specific lives in a task adapter:
`src/clab/tasks/supermimic.py` holds the shared rigid-object adapter, and each task adds
`src/clab/tasks/<task>/supermimic.py`.

## Algorithm

Each outer iteration:

1. **Collect.** The deterministic actor plus Gaussian noise (std 0.1) acts in the real
   simulator; transitions enter a time × world replay that keeps native terminal states.
2. **Fit the world model.** Windows of 8 steps are predicted open-loop and trained with a
   normalized delta loss; steps after a termination are masked. By default the number of
   batches is sized so that each new transition is seen in about 4 windows
   (`--world-model-schedule epochs --world-model-epochs 4`).
3. **Improve the policy.** From replayed states and actor histories, the actor imagines
   16 steps through the frozen world model and critic (parameters frozen, input gradients
   kept) and maximizes the discounted imagined return plus the terminal value.
4. **Fit the critic** to TD(λ) targets of the same imagined rollouts.

The first 1,000 iterations train only the world model. The default budget is 10,000
iterations. Optimizers are Adam (2e-4) with a cosine decay to 3% over
`--learning-rate-decay-iterations` (default 10,000, the full budget); γ = 0.99, λ = 0.95,
gradient-norm clip 1. MLPs run under BF16 autocast with FP32 parameters, optimizer state,
normalizers, integration and rewards; TF32 matmuls are on by default. The world-model loss
and one imagined transition are compiled with `torch.compile`.

## Rollout state

The adapter captures the physical state after each control step (before an automatic
reset) through a read-only hook; it never replaces the simulator or the controller.

| Part | Content |
| --- | --- |
| Physical state (predicted) | Arm and finger joint positions and velocities, object position, rotation (first two matrix columns) and linear/angular velocity, plus task extras (Drawer: slide position and velocity) |
| Analytic state (not predicted) | Previous action, episode clock, success counter and task process flags (Drawer: the full physical-sequence milestones) |
| Context (constant per episode) | Base and target poses, gravity, randomized masses, inertias, friction and joint parameters |

A world whose physics diverged is terminated by the task; its transition is stored with the
finite pre-step state, zero reward and an invalid flag, and world-model windows mask it
(`diverged_transitions` in the completion report counts them). A contact sensor that
matches more contacts than its storage only drops the extra matches, so such world-steps
are counted (`Train/contact_match_overflow_world_steps`). A physics capacity overflow drops
contacts or constraints: that world's transitions are excluded like diverged ones until it
resets (`Train/physics_overflow_world_steps`), and the run stops if more than 1% of the
worlds overflow at once.

## World model

The operational-space controller re-anchors at the current end-effector pose on every
action, so it has no internal memory to model: the world model only needs the physical state
of the arm and the object (and Drawer's slide).

| Task | World-model input | Output | Rollout state | Context | Actor | Critic input |
| --- | --- | --- | --- | --- | --- | --- |
| `box_in_bin` | 180 | 28 | 40 | 142 | 195 | 377 |
| `lemon` | 183 | 28 | 40 | 145 | 195 | 380 |
| `square` | 183 | 28 | 40 | 145 | 195 | 380 |
| `mug_hang` | 267 | 28 | 40 | 229 | 210 | 479 |
| `drawer` | 201 | 30 | 55 | 161 | 265 | 468 |

**Input** is the concatenation of three parts.

| Part | Dims | Meaning |
| --- | --- | --- |
| `robot_q` | 8 | Joint positions: six arm joints (rad) and two finger joints (m) |
| `robot_dq` | 8 | Joint velocities of the same joints |
| `<object>_position` | 3 | Object position (m) in world axes, relative to the environment origin |
| `<object>_rotation` | 6 | Object orientation as the first two columns of its rotation matrix (continuous, unlike quaternions or Euler angles) |
| `<object>_velocity` | 6 | Object linear (m/s) and angular (rad/s) velocity in world axes |
| `slide_q`, `slide_dq` | 1 + 1 | Drawer only: slide position (m) and velocity (m/s) |
| Executed command | 7 | What the controller actually executes for the raw action: position offset (3, m) and rotation vector (3, rad) after the controller's clip, scaling and per-step norm bound, plus the commanded finger position (1, m: 0.002 closed, 0.044 open; binary forward, with a sigmoid straight-through gradient) |
| Context | 142-229 | Constants of the episode, listed below |

| Context field | Dims | Meaning |
| --- | --- | --- |
| `base_position`, `base_rotation` | 3 + 6 | Robot base pose |
| `<target>_position`, `<target>_rotation` | 3 + 6 | Pose of the fixed target (bin, plate, peg, tree or drawer housing) |
| `origin_z` | 1 | Height of the environment origin |
| `gravity` | 3 | Gravity vector |
| `body_mass` | 12 (BoxInBin, Lemon, MugHang) | Randomized masses of the 9 robot bodies, the object, the target and the table |
| `body_inertia` | 36 (BoxInBin, Lemon, MugHang) | Their principal inertias (3 each) |
| `geom_friction` | 48 / 51 / 135 (BoxInBin / Lemon / MugHang) | Sliding, torsional and rolling friction (3 each) of the 16 / 17 / 45 collision geoms of robot, object, target and table (MugHang's mug has 32 collision pieces) |
| `dof_armature`, `dof_damping`, `dof_frictionloss` | 8 each | Joint armature, damping and dry friction of the 8 robot joints |

Square and Drawer have the same fields over their own bodies, collision geoms and joints.
Feeding the actual randomized physical parameters lets one model cover the domain
randomization of training without having to infer it from history.

**Output** (28) is the normalized increment of the physical state over one 0.05 s control
step: `robot_q` 8, `robot_dq` 8, object position 3, object rotation 3 and object velocity 6
(Drawer adds the slide, 30). The rotation increment is a world-axis rotation vector (the
SO(3) tangent), `log(R_next R_prev⁻¹)`, integrated as `exp(δ) R_prev`; all other parts are
added. Inputs and outputs are normalized with running means and standard deviations.

**Analytic state** (9, not predicted) completes the 40-D rollout state: the previous raw
action (7), the episode clock (1) and the success counter (1). They are advanced by rules,
not learned. Drawer adds its 13 physical-sequence columns (55-D).

| Network | Architecture | Parameters (BoxInBin) |
| --- | --- | --- |
| World model | MLP 5 × 2048, LayerNorm and SiLU, zero-initialized output layer | 17.2 M |
| Actor | MLP 2048-1024-512-256, ELU; observation normalizer; no output squashing | 3.2 M |
| Critic | MLP 256-128, ELU; input = rollout state + context + actor observation (Drawer: without the 13 process columns) | 0.13 M |

**Training loss.** Windows of 8 consecutive steps are rolled out open-loop: each prediction
is fed back as the next input. At every step the error between the predicted and the real
state (as an output increment, with the rotation error as a rotation vector) is divided by the
output standard deviation, floored at 1e-3 (1e-2 for joint positions, joint velocities and
object velocity), squared and averaged. Steps after an episode end, and transitions from a
world whose physics diverged, are masked.

The actor observation (195 = 39 × 5 for BoxInBin, Lemon and Square, 210 = 42 × 5 for
MugHang) holds five frames of: object pose in the target frame (6), previous action (7),
joint positions (8), end-effector pose in the base frame (6), object pose in the
end-effector frame (6) and target pose in the end-effector frame (6), poses as position plus
axis-angle; MugHang adds the handle-to-hanging-point vector (3) and Drawer (265) the slide
and its physical-sequence observation ([tasks.md](tasks.md)). In imagination these frames
are rebuilt from the predicted state with differentiable forward kinematics and match the
native observation to about 1e-6 on real states. The world model is a reduced surrogate,
not an exact MuJoCo state.

Drawer's actor and critic see its physical sequence only as the stage and the history the
rest of the episode still depends on: each stage's references and counters during that stage
(zero elsewhere) and the success flag. The reset prefix, the episode's
accomplishments and the initial slide never change the outcome and are not observed, so a pod
released in the drawer looks the same whether it was released in this episode or before a
`PodReleasedInOpenDrawer` reset. With them observed, the critic valued a pod a
`PodHeldAboveOpenDrawer` episode had just released at −1.8 and the same pod at that reset at
+6, and the transport categories never learned to release it in the drawer.

## Imagined reward and success

The world model predicts states, not rewards. Imagined rewards use the native reward
weights and are computed analytically from imagined states (reaching from forward
kinematics plus the grasp offset, task guidance from object poses, joint-velocity cost from
predicted joint speeds). Contact, support-force and release tests have no exact analogue in
the reduced state, so each task replaces them with geometric surrogates:

| Task | Imagined success |
| --- | --- |
| `box_in_bin` | Native placement geometry; speeds below 0.03 m/s and 0.5 rad/s (native 0.01 and 0.1); both fingers opened past 0.028 m |
| `lemon` | Native centre XY and height thresholds; height at most `contact_margin` (0.01 m) above the resting height as the contact proxy; both fingers opened past 0.023 m |
| `square` | Native threaded/height/tilt geometry; released when both fingers' collision boxes clear every nut box by `release_clearance` (1 mm); looser speeds. The grasp the shaping terms need is the grasp point inside a nut box with both fingers in a closed-on-object band |
| `mug_hang` | Native tree-frame envelope and handle-on-bar geometry; looser speed limit; both fingers opened past 0.015 m |
| `drawer` | Native physical-sequence update rules; pod grasp from grasp-point distance (`grasp_radius` 0.03 m) and finger band; push contact from the grasp point at the front panel; native final candidate |

Common to all tasks:

- the native 0.5 s hold is kept (success counter carried as analytic state);
- the success term's gradient to the actor is scaled by `success_gradient_scale` (0.02),
  keeping its forward value (+10) for the critic;
- geometry, speed and success gates are hard in the forward pass and use sigmoid
  straight-through gradients backward.

BoxInBin, Lemon and MugHang add one imagination-only shaping term: once the object is
placed, reaching stops paying and `release_weight` (0.2) × the finger opening fraction from
the held width to the release opening is added, so holding on stops being rewarded. Square
and Drawer already have native release terms; for Square, reaching keeps paying while the
nut is grasped (natively it stops, so hovering an open gripper at the handle paid more than
holding it) until the nut is seated, and holding stops paying once the nut is threaded (below). MugHang's native threading term vanishes
while the handle is misaligned with the bar, so its imagination also adds a long-range pull of
the handle toward the hanging target, `carry_weight` (0.2) × exp(−distance / 1 m). Square's
alignment and insertion terms are 3 cm Gaussians, so its imagination likewise pulls the nut toward
the nearer of an above-peg pose and the seated pose while the nut is grasped or threaded.
Drawer's
native progress restarts at every stage (lift, place, close), so completing a stage would lower
the reward; its imagined progress adds the number of completed stages instead, and its
closing-stage reaching target sits 2 cm in front of the panel (natively on it), since a hand left
inside the drawer pushed it only part-way before jamming against the housing. Releasing the pod
likewise ends reaching for the held pod (about 1) while the closing-stage reach starts near 0.1,
so the held pod was carried away from the drawer; the closing and final stages' imagined reaching
adds 1 for the completed reach.

In Square and Drawer the full-task policy closed its gripper on the first step and never
opened it again: a closed hand beside the object, which cannot grasp it, was paid the same
reaching reward as an open one. Their imagined reaching therefore pays only an open hand (both
fingers past the grip band) or a grasp. Drawer applies this only until the pod grasp is first
confirmed or the lift completed: gating the held carry on the grasp made every slip cost
reaching too, and the grasped starts held the pod aloft without carrying it. Their native
reaching tanh (5 cm for Square, 10 cm for Drawer) is flat at the ~0.5 m full-task start, so
their imagination adds a far term 1 − tanh(distance / `far_reaching_scale`, 0.5 m) toward the
handle or pod; Drawer averages the two, keeping the native range [0, 1].

Square and Drawer also make letting go visible to the actor. Without this, every grasped start
category ended with the gripper opened on the first step: the grasp tests are hard, so their
derivative is zero, while the dropped object falling toward its target improved the smooth
terms.

- Their grasp proxies are straight-through in the closed-on-object finger band, so the rewards
  that need a grasp show that opening the gripper loses them.
- That derivative reaches the gripper only through the world model's finger prediction, so
  Drawer also shapes the raw command: opening a held pod costs `outside_release_penalty` (2)
  in the lift stage and, in the release stage, unless the pod is over the drawer opening (the
  inside footprint at any height), from where it drops into the drawer; keeping the gripper
  closed once the pod is inside costs `inside_hold_cost` (2 per second). Without the
  lift-stage cost the grasped-on-table starts learned to open at once and dropped the pod.
  Drawer's imagined `done` is success or the time limit, as natively.
- Square shapes the release on the raw gripper command, whose straight-through derivative
  acts directly; through the world model's finger prediction, which is trained only at the two
  binary targets, the derivative was too weak to change a decision. A nut over the peg drops
  and self-centres, so the grasped starts taught "open at once", which drops the nut on the
  table everywhere else. Commanding open on a held nut away from the peg (not threaded, centre
  beyond `release_radius`, 1 cm, of the axis) costs `early_release_penalty` (2); commanding
  closed while the nut is threaded costs `peg_hold_cost` (2 per second); and holding stops
  paying once the nut is threaded (natively once seated), since pushing a held, threaded nut
  down tilts and jams it, while released it slides down and seats.
- Square also makes letting go over the peg pay. The any-grasp policy carried the nut over
  the peg and hovered there, centre 9.2 cm above the peg's and 1 cm off its axis, until the
  time limit, in every grasped start. Released there, 57% of nuts self-centred and seated, 86%
  once centred on the axis, but letting go lost the grasp, alignment and carry terms at once
  and the success came 0.5–1 s later. A nut in the drop zone (centre within `release_radius`
  of the axis, between 5 mm above a nut resting on the peg top and 4 cm above the carry
  target) counts as held for those terms; commanding open on a held nut there pays
  `drop_release_bonus` (2 per second); and `centring_weight` (0.1) × exp(−(axis distance /
  `release_radius`)²) pulls a held nut onto the axis, where the native 3 cm alignment
  Gaussian barely pulls.
- Square's release proxy is geometric: each finger's collision boxes (its pad within 2 cm of
  the grasp point and the rest of the finger), posed by the arm joints and its opening, must
  clear every nut box by `release_clearance` (1 mm). Native release is
  the end of every nut-robot contact. The previous proxy, not both fingers inside the finger
  band, counted any opened hand as released: in most stalled episodes the nut was seated but
  one open finger still pressed on the ring (about 1.5 N), and imagination counted a success
  the native test never did, so nothing taught the hand to move away.
- These release costs use the native binary decision forward and a hinge derivative
  backward: constant until the raw command is past the decision boundary by `gripper_margin`
  (1), zero beyond. A sigmoid derivative vanishes on a saturated command (Square's reached
  +12, later −5, and froze there); one that never vanishes kept driving correct decisions on,
  to −7 against the action bound below, whose cost then dominated the imagined return, and the
  policy drifted to wherever its own command was cheapest, away from the peg.
- Both set `action_bound_weight` (0.05 per second; 0 elsewhere): a squared cost on the raw
  gripper command beyond ±`gripper_bound` (3) and on raw pose axes beyond the controller's
  input limit, where the clip has no derivative.

These settings are recorded in `params/mdp.json` and can be overridden with
`--adapter-option NAME=VALUE` (`stable_linear_speed`, `stable_angular_speed`,
`release_opening`, `release_weight`, `success_gradient_scale`, `action_bound_weight`,
`gripper_bound`, `gripper_margin`; Lemon also `contact_margin`, MugHang and Square
also `carry_weight`, Square also `early_release_penalty`, `release_radius`, `peg_hold_cost`, `drop_release_bonus`, `centring_weight` and `release_clearance`,
Drawer also `grasp_radius`, `outside_release_penalty` and `inside_hold_cost`). Real collection and evaluation always use the native reward
and success.

## Options and diagnostics

| Option | Effect |
| --- | --- |
| `--iterations N`, `--learning-rate-decay-iterations N` | Budget and cosine-decay length; keep them equal |
| `--world-model-schedule {batches,epochs}` | Fixed `--world-model-batches` per iteration, or `--world-model-epochs` passes over new transitions |
| `--validation-fraction F` | Hold the last worlds out of world-model fitting and log validation losses (default 0.05) |
| `--pin-replay` | Freeze the replay collected before policy updates start and stream new data into a second ring |
| `--exploration-fraction F` | In the first worlds, occasionally flip the gripper command for several steps |
| `--wm-ensemble-size K --uncertainty-penalty P` | Train K independent world models and penalize their disagreement in imagination |
| `--policy-update-interval K` | Update actor/critic every K iterations after the world-model-only phase |
| `--full-task-resets` | Start every training episode from reset category 0 (the full task) instead of all categories; an ablation of the reset categories |
| `--reset-start-fraction F` | Draw a fraction F of imagination starts from episode-start (reset) rows instead of uniformly over replay |
| `--no-compile-world-model`, `--no-compile-policy-update` | Disable `torch.compile` (slower; useful for debugging) |
| `--smoke` | Tiny networks and budgets; checks saving, reloading and one more update |

Key logged values: `Model/wm_step_loss` and `Model/wm_val_step_loss` (world-model
accuracy), `Imagination/imagined_reward`, and `Model/candidate_{surrogate,native,
surrogate_only,native_only}_rate`, which compare the imagined success predicate with the
native one on real states. A high `surrogate_only` rate means the policy can exploit the
surrogate; a high `native_only` rate means the surrogate is too strict.

## Adding a task

1. Subclass `RigidObjectMDP`: set `task`, `object_name`, `target_name`, `held_width`,
   `release_opening`, optional `process_fields` and `actor_widths`, and implement the
   success surrogate, the reward and any extra observations.
2. Subclass `RigidObjectAdapter`: entity names, object/target contact geoms, the actor
   observation terms, the native reward terms it reproduces and the native success term.
3. Point the task binding's `supermimic_adapter` at the adapter class.
4. Check it with `clab train supermimic --smoke`: the completion report lists
   `native_observation_max_error` and the candidate agreement rates.
