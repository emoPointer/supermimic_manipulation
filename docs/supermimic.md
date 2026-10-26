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
| Analytic state (not predicted) | Previous action, episode clock, success counter and task process flags (Square: `grasp_seen`, `invalid`; Drawer: the full physical-sequence milestones) |
| Context (constant per episode) | Base and target poses, gravity, randomized masses, inertias, friction and joint parameters |

A world whose physics diverged is terminated by the task; its transition is stored with the
finite pre-step state, zero reward and an invalid flag, and world-model windows mask it
(`diverged_transitions` in the completion report counts them). A contact sensor that
matches more contacts than its storage only drops the extra matches, so such world-steps
are counted (`Train/contact_match_overflow_world_steps`). A physics capacity overflow drops
contacts or constraints: that world's transitions are excluded like diverged ones until it
resets (`Train/physics_overflow_world_steps`), and the run stops if more than 1% of the
worlds overflow at once.

The operational-space controller re-anchors at the current end-effector pose on every
action, so it has no internal memory to model. The world model predicts only the physical
increments (28-D, 30-D for Drawer); rotation increments are SO(3) tangent vectors.

| Task | Rollout state | Context | Actor | Critic input |
| --- | --- | --- | --- | --- |
| `box_in_bin` | 40 | 142 | 195 | rollout state + context + actor history |
| `lemon` | 40 | 145 | 195 | same |
| `square` | 42 | 145 | 195 | same |
| `mug_hang` | 40 | 229 | 210 | same |
| `drawer` | 56 | 161 | 300 | same |

The world model sees the executed command: the action clipped to the controller input
limit, scaled, and bounded in norm per step exactly as the OSC action does; the gripper is
binary forward with a sigmoid straight-through gradient. Imagined actor observations are
rebuilt from the imagined state with differentiable forward kinematics and match the native
observation to about 1e-6 on real states. The world model is a reduced surrogate, not an
exact MuJoCo state.

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
| `square` | Native threaded/height/tilt geometry; handle versus ring grasp from the grasp point in the nut frame and a closed-on-object finger band, carried as `grasp_seen`/`invalid` flags; released when not both fingers are inside that band; looser speeds |
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
handle is grasped (natively it stops, so hovering an open gripper at the handle paid more than
holding it) until the nut is seated, and holding stops paying once the nut is threaded (below). MugHang's native threading term vanishes
while the handle is misaligned with the bar, so its imagination also adds a long-range pull of
the handle toward the hanging target, `carry_weight` (0.2) × exp(−distance / 1 m). Square's
alignment and insertion terms are 3 cm Gaussians, so its imagination likewise pulls the nut toward
the nearer of an above-peg pose and the seated pose while the handle is grasped or the nut
threaded. Drawer's
native progress restarts at every stage (lift, place, close), so completing a stage would lower
the reward; its imagined progress adds the number of completed stages instead, and its
closing-stage reaching target sits 2 cm in front of the panel (natively on it), since a hand left
inside the drawer pushed it only part-way before jamming against the housing.

Square and Drawer also make letting go visible to the actor. Without this, every grasped start
category ended with the gripper opened on the first step: the grasp tests and Drawer's failure
latch are hard, so their derivative is zero, while the dropped object falling toward its
target improved the smooth terms.

- Their grasp proxies are straight-through in the closed-on-object finger band, so the rewards
  that need a grasp show that opening the gripper loses them.
- Drawer's `done` uses the native failure event in the forward pass. Backward, it uses the
  derivative of the active stage's failure condition (grasp lost after it was confirmed but
  before the lift, pod released outside the drawer, pod re-grasped or outside the drawer while
  closing), so the actor sees that failing forfeits the rest of the imagined return.
- Square shapes the release on the raw gripper command, whose straight-through derivative
  acts directly; through the world model's finger prediction, which is trained only at the two
  binary targets, the derivative was too weak to change a decision. A nut over the peg drops
  and self-centres, so the grasped starts taught "open at once", which drops the nut on the
  table everywhere else. Commanding open on a held nut away from the peg (not threaded, centre
  beyond `release_radius`, 1 cm, of the axis) costs `early_release_penalty` (2); commanding
  closed while the nut is threaded costs `peg_hold_cost` (2 per second); and holding stops
  paying once the nut is threaded (natively once seated), since pushing a threaded nut down by
  its handle tilts and jams it, while released it slides down and seats.
- Both set `action_bound_weight` (Square 0.2, Drawer 0.05, per second; 0 elsewhere): a squared
  cost on the raw gripper command beyond ±`gripper_bound` (3) and on raw pose axes beyond the
  controller's input limit. The binary gripper's sigmoid derivative vanishes far from zero and
  the clip has none past the limit, so a saturated command (Square's reached +12, then −5)
  could no longer change. Square also sets `gripper_gradient_bound` (3; 0 elsewhere): the
  sigmoid derivative is taken at the raw command clamped to that bound, so it never vanishes.

These settings are recorded in `params/mdp.json` and can be overridden with
`--adapter-option NAME=VALUE` (`stable_linear_speed`, `stable_angular_speed`,
`release_opening`, `release_weight`, `success_gradient_scale`, `action_bound_weight`,
`gripper_bound`, `gripper_gradient_bound`; Lemon also `contact_margin`, MugHang and Square
also `carry_weight`, Square also `early_release_penalty`, `release_radius` and `peg_hold_cost`,
Drawer also `grasp_radius`). Real collection and evaluation always use the native reward
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
