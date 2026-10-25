# Tasks

All five tasks share one robot (a simulated ARX5 arm with a parallel gripper), one
workcell layout and one control timing: 20 Hz policy steps and 480 Hz `implicitfast`
physics (24 substeps per action). Scenes use elliptic friction cones with `impratio=5`.
Observation term order, dimensions, frames and units are checkpoint contracts: a checkpoint
loads only into an environment with the same saved contract.

## Shared conventions

- **World frame.** The origin is the bottom-centre of the robot's mounting pad, axes along
  the pad, Z up; the tabletop is at Z=0. Parallel environments add only `env_origins`.
- **Action (7-D).** `[dx, dy, dz, rx, ry, rz, gripper]`, a relative end-effector pose
  command in world axes. The six pose values are clipped to [-12, 12], scaled by 0.02 m and
  0.2 rad (rotation vector), and bounded per step to 0.075 m and 0.4 rad (1.5 m/s and
  8 rad/s over the 0.05 s control period). The target is anchored at the current
  end-effector pose on every action. The gripper closes for negative values and opens
  otherwise (finger targets 0.002 m and 0.044 m).
- **Controller.** Operational-space control with a pseudo-inverse task-space inertia,
  decoupled position and orientation, stiffness 200 N/m and 300 N·m/rad, damping ratios
  3 and 1, joint torques limited to 50 N·m, gravity compensated in the model.
- **Actor observations** use joint positions and end-effector poses from the simulator,
  with five history frames (oldest to newest, flattened per term). **Critic observations**
  add time left, velocities and the actual randomized physical parameters.
- **Episodes** have a finite horizon: timeouts and successes are terminal, and no value is
  bootstrapped from an automatic-reset observation. A world whose physics diverges (rare
  solver blow-ups) is terminated as a failure and reset; the other worlds are unaffected.
- **Success** is an explicit task predicate held for 0.5 s (10 consecutive checks). It pays
  a one-off +10 (`success` weight `10 / control_dt`) and ends the episode. Termination alone
  never implies success; evaluation reads the explicit outcome.
- **Rewards** share the names `reaching`, `success`, `action_magnitude`, `action_rate` and
  `joint_vel`; task guidance terms are `placement`, `insertion`, `threading` or `progress`.
  Logged `Episode_Reward/*` values are episode sums divided by the maximum episode length.
- **Reset categories** are sampled uniformly from the task's reset bank; every task's
  category 0 is the complete task, later categories start part-way through it.

## BoxInBin (`box_in_bin`)

Carry a 100 g box into a fixed bin, release it, and let it rest.

| Item | Contract |
| --- | --- |
| Props | Free box (one box geom, sliding friction 1); bin is a mocap body (five box geoms) |
| Horizon | 16 s |
| Observations | actor 195 = 39 × 5; critic 164 |
| Actor terms | box pose in bin (6), previous action (7), joint positions (8), EE pose in base (6), box pose in EE (6), bin pose in EE (6); poses are xyz + axis-angle |
| Rewards | `action_magnitude` −1e-4, `action_rate` −1e-3, `joint_vel` −1e-2, `reaching` 0.2, `placement` 0.2, `success` +10 |
| Success | Box fully inside the bin at a valid height; both fingers released; bin support at least 90% of the box's actual weight; linear speed ≤ 0.01 m/s and angular speed ≤ 0.1 rad/s; held 0.5 s |
| Reset types | `ObjectAnywhereEEAnywhere`, `ObjectRestingEEGrasped`, `ObjectAnywhereEEGrasped`, `ObjectPartiallyAssembledEEGrasped` |
| Bank frame | Robot-base-local poses |
| Reset placement | Box and bin root centres in X [0.15, 0.35] m, Y [−0.20, 0.20] m, yaw ±180° |

## Lemon (`lemon`)

Put an 80 g lemon on a fixed plate and release it.

| Item | Contract |
| --- | --- |
| Props | Free lemon (two convex pieces, 80 g); plate is a mocap body (five convex pieces) |
| Horizon | 32 s |
| Observations | actor 195; critic 153 |
| Rewards | `action_magnitude` −1e-3, `action_rate` −1e-2, `joint_vel` −5e-2 (10× and 5× the other tasks), `reaching` 0.2, `placement` 0.2, `success` +10 |
| Success | Centre XY distance < 45 mm, height difference in (5, 120) mm, lemon–plate contact in any physics substep of the step, not grasped by both fingers; held 0.5 s |
| Reset types | `ObjectAnywhereEEAnywhere`, `ObjectRestingEEGrasped`, `ObjectAirborneEEGrasped`, `ObjectNearGoalEEGrasped` |
| Bank frame | Environment-world poses |
| Reset placement | Lemon and plate root centres in X [0.15, 0.35] m, Y [−0.20, 0.20] m, yaw ±180° |

The success predicate is deliberately permissive: it has no speed threshold and does not
exclude other robot links touching the lemon.

## Square (`square`)

Grasp the square nut by its protruding handle and thread it onto the fixed peg.

| Item | Contract |
| --- | --- |
| Props | Free nut (five box geoms keeping the hole and handle); peg with base is a mocap body |
| Horizon | 16 s |
| Observations | actor 195; critic 153 |
| Rewards | `reaching` 0.2, `handle_grasp` 0.1, `alignment` 0.1, `insertion` 0.2, `release` 0.1, the shared action/joint costs, `success` +10, `invalid_grasp` −2 |
| Success | Peg threaded through the hole; nut seated at peg-local Z = −0.02 m (±2 mm) with tilt ≤ 5°; base support ≥ 0.05 N; robot fully released; speeds ≤ 0.01 m/s and 0.1 rad/s; object contact penetration ≤ 1 mm; a legal handle grasp observed earlier in the episode; held 0.5 s |
| Failure | Grasping the ring body (or a mixed handle/ring grasp) with both fingers ends the episode with −2 |
| Reset types | `NutRestingEEAnywhere`, `NutRestingHandleGrasped`, `NutAirborneHandleGrasped`, `NutPartiallyInsertedHandleGrasped` (labelled `above_peg` / `threaded`) |
| Bank frame | Environment-world poses |
| Reset placement | Nut X [0.235, 0.295], Y [−0.090, −0.010], yaw [135°, 225°]; peg X [0.235, 0.295], Y [0.050, 0.110], yaw ±45°; footprints at least 3 cm apart in Y |

Guidance terms are gated by the grasp/insertion stage; the gated guidance totals at most
0.4 per second.

## MugHang (`mug_hang`)

Hang a mug on a fixed tree by threading its handle onto the bar.

| Item | Contract |
| --- | --- |
| Props | Free ShapeNet mug (decomposed collision keeping the handle hole); tree is a mocap body |
| Horizon | 16 s |
| Observations | actor 210; critic 240, both including the handle-anchor → hanging-target vector in the tree frame |
| Rewards | `reaching` 0.2, `threading` 0.2, the shared action/joint costs, `success` +10 |
| Success | Mug root inside the tree-frame envelope (abs x < 0.085, abs y < 0.05, above the table by 0.11 m), translation speed < 0.01 m/s, not grasped by both fingers, and the handle engaged on the bar; held 0.5 s |
| Reset types | `ObjectAnywhereEEAnywhere`, `ObjectRestingEEGrasped`, `ObjectAirborneEEGrasped`, `HandleThreadedEEGrasped` |
| Bank frame | Environment-world poses |
| Reset placement | Mug X [0.20, 0.24], Y [−0.130, −0.090], yaw [−120°, −60°]; tree X [0.165, 0.195], Y [0.220, 0.250], yaw −90° |

Handle engagement: the handle anchor (mug-local `(0, 0.03, 0)`) lies within 0.018 m of the
bar segment (tree-local `(0.015, 0, 0.055)` to `(0.095, 0, 0.055)`), projects inside the
segment ±5 mm, and the bar direction has a dot product ≥ 0.75 with the mug's +X axis. The
observed hanging target is the bar midpoint `(0.055, 0, 0.055)`.

## Drawer (`drawer`)

Pick up a coffee pod, put it into an already opened drawer, and push the drawer closed.

| Item | Contract |
| --- | --- |
| Props | Free pod; fixed housing with a passive drawer slide (range [0, 0.14] m) that starts open at 0.13 m |
| Horizon | 60 s |
| Observations | actor 300 and critic 81, including a 19-D physical-sequence history |
| Rewards | `reaching` 0.2, `progress` 0.2, the shared action/joint costs, `success` +10 |
| Process | Stable two-finger grasp for 0.3 s and lift ≥ 0.04 m → release inside the drawer for 0.3 s → contact push closing the slide by ≥ 0.015 m |
| Success | Process complete, slide q < 0.03 m, pod inside the drawer and not grasped; held 0.5 s |
| Reset types | See below |
| Bank frame | Environment-local world (positions minus the environment origin) |
| Reset placement | Pod X [0.075, 0.205], Y [0.090, 0.250], yaw ±18°; housing X [0.325, 0.335], Y [−0.080, −0.040], yaw [165°, 195°] |

Drawer is a long-horizon task, so its reset bank holds six snapshots of one complete
physical trajectory. Later stages shorten the credit-assignment distance; the process
milestones already achieved in the trajectory (grasp/lift, inside release) are restored as
inherited flags, never counted as new accomplishments.

| Category | Snapshot | Inherited grasp/lift, release |
| --- | --- | --- |
| `DrawerOpenedPodOnTable` | Drawer open, pod on the table, hand free | no, no |
| `DrawerOpenedPodGraspedOnTable` | Pod grasped, still on the table | no, no |
| `DrawerOpenedPodAirborneGrasped` | Pod grasped and lifted | yes, no |
| `PodHeldAboveOpenDrawer` | Pod held above the open drawer | yes, no |
| `PodReleasedInOpenDrawer` | Pod released inside the drawer | yes, yes |
| `DrawerHalfClosed` | Drawer pushed to 0.07 m | yes, yes |

Drawer uses convex-decomposed finger collisions so the finger grooves are not filled by a
single convex hull; the other tasks use one convex hull per finger.
