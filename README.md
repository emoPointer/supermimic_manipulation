# SuperMimic for robot manipulation

This repository applies [SuperMimic](https://arxiv.org/abs/2610.28480), policy optimization by
backpropagating the task return through a learned differentiable world model, to state-based
robot manipulation in [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp). It contains
five manipulation tasks for a simulated ARX5 arm, reset-bank collectors for
[OmniReset](https://arxiv.org/abs/2603.15789)-style start categories, SuperMimic and a PPO
baseline, and a complete-episode evaluator.

中文说明：[README.zh-CN.md](README.zh-CN.md)

<table>
  <tr>
    <th>Lemon: put the lemon on the plate</th>
    <th>BoxInBin: put the box in the bin</th>
    <th>MugHang: hang the mug by its handle</th>
  </tr>
  <tr>
    <td><img src="docs/media/lemon.gif" width="100%"></td>
    <td><img src="docs/media/box_in_bin.gif" width="100%"></td>
    <td><img src="docs/media/mug_hang.gif" width="100%"></td>
  </tr>
  <tr>
    <th>Square: seat the nut on the peg</th>
    <th>Drawer: put the pod in the drawer and close it (2× speed)</th>
    <th></th>
  </tr>
  <tr>
    <td><img src="docs/media/square.gif" width="100%"></td>
    <td><img src="docs/media/drawer.gif" width="100%"></td>
    <td></td>
  </tr>
</table>

Each clip shows two episodes of a trained policy. The arm starts at its home pose (all joints
at zero) and the object and target come from a held-out test bank. The policy only sees
low-dimensional state; the camera is for viewing. From the home pose the Square policy
succeeds in 89% of 64 episodes and its clip shows the first two; the Drawer policy succeeds
in 53%, and its clip shows the first two successes, which took six tries, at 2× speed.

## Results

Every number below is native task success (the simulator's own success test, held for
0.5 s) from complete episodes with the deterministic actor, on a held-out reset bank
collected with a different seed and split than the training bank: 64 episodes from each
reset category. No episode was filtered.

| Task | Iterations | Training time | Overall | Per reset category |
| --- | --- | --- | --- | --- |
| `lemon` | 4,000 | 1.8 h | **100.0%** | 100 / 100 / 100 / 100 |
| `box_in_bin` | 4,000 | 1.9 h | **94.9%** | 97 / 91 / 95 / 97 |
| `mug_hang` | 4,000 | 2.8 h | **96.9%** | 94 / 98 / 95 / 100 |
| `square` | 18,000 | 17.5 h | **96.6%** | 98 / 95 / 95 / 98 / 95 |
| `drawer` | 6,000 | 4.5 h | **77.2%** | 50 / 80 / 86 / 92 / 75 / 97 / 61 |

- Lemon, BoxInBin and MugHang use the 4,000-iteration checkpoints released with the `main`
  branch. Square and Drawer use, for the run in the clip, the checkpoint with the best
  training-bank success (32 episodes per category, every 2,000 iterations). The second seeds
  reached 78.1% (Square, 18,000 iterations) and 68.1% (Drawer, 8,000).
- Drawer's success varies strongly between checkpoints: 59, 77, 22 and 61% at 4,000, 6,000,
  8,000 and 10,000 iterations for the run above.
- Square trained 10,000 iterations and was resumed for 10,000 more with the cosine
  learning-rate schedule stretched to 20,000 iterations, so the rate restarted at about half
  its initial value; its time is the sum of both parts.
- Training time is wall-clock time from launch to the checkpoint, including the first 1,000
  world-model-only iterations, on one A100-SXM4-80GB. Runs shared their GPU with other jobs
  (the two Square seeds shared one), so the times are upper bounds.
- Category 0 is the full task (object on the table, gripper free). Later categories start
  part-way through it; the last Square and Drawer category is an open hand just above the
  object (see [Square and Drawer](#square-and-drawer)).

**Data.** Each iteration collects 10,240 simulated transitions from 4,096 parallel worlds:
41 M for the 4,000-iteration runs (about 24 days of simulated time at 20 Hz), 185 M for
Square (107 days) and 62 M for Drawer (36 days). The policy is trained only through the
world model: no policy-gradient (PPO) update is used. Training banks hold 10,000 start states
per category (Square 9,999, Drawer 1,993), held-out banks 256 (Drawer 255).

## Tasks

| Task | Goal | Horizon | Actor / critic | Reset categories |
| --- | --- | --- | --- | --- |
| `box_in_bin` | Put a box in a bin, release it and let it rest | 16 s | 195 / 164 | 4 |
| `lemon` | Put a lemon on a plate and release it | 32 s | 195 / 153 | 4 |
| `square` | Thread a square nut onto its peg and release it | 16 s | 195 / 153 | 5 |
| `mug_hang` | Hang a mug on a tree by its handle | 16 s | 210 / 240 | 4 |
| `drawer` | Put a pod into an open drawer and push the drawer closed | 60 s | 265 / 74 | 7 |

The arm is driven by an operational-space controller (OSC) at a 20 Hz policy rate and
480 Hz physics. Every task uses the same 7-D action: a relative end-effector pose command in
world axes plus a binary gripper command. Observations, rewards, success tests and reset
categories are listed in [docs/tasks.md](docs/tasks.md).

## Square and Drawer

**Square.** Grasp the square nut, carry it to the peg, lower it over the peg and let go. The
nut counts as seated when the peg passes through its hole, it rests on the base (±2 mm,
tilt at most 5°) with the robot fully released and the nut still, for 0.5 s. Any grasp of
the nut counts; reaching aims at the handle, which leaves the hole free for the peg. Its five
start categories: the nut on the table with the gripper free; the handle grasped with the nut
on the table, lifted, or partly inserted (held beside, above or threaded on the peg); and an
open hand 1-6 cm above the handle of a resting nut.

**Drawer.** Pick up a coffee pod, put it into the drawer, which starts 13 cm open, let go
inside and push the drawer closed. The native process is a stable two-finger grasp held
0.3 s with a 4 cm lift, a release inside the drawer held 0.3 s, then a contact push that
closes the slide by at least 1.5 cm; success is the completed process with the slide below
3 cm and the pod inside and not grasped, for 0.5 s. Only the 60 s time limit ends an episode
early. The reset bank holds six snapshots of each complete physical trajectory (pod on the
table; grasped; lifted; held above the drawer; released inside; drawer half closed) and an
open hand just above the resting pod.

With only the shared adaptation ([docs/supermimic.md](docs/supermimic.md)), the full-task
policy of both tasks closed its gripper on the first step and never opened it again, and
the grasped starts let go at once. What they needed:

- **A pre-grasp start category.** Nothing bridged the full task, with the gripper often half
  a metre away, and the first grasped category, so the critic never valued an open hand at
  the object. Each pre-grasp row is derived physically from a grasped-on-table row (open,
  rise and hold the open hand 1-6 cm above the object) and kept only if the object stays at
  rest and untouched.
- **Task rules that do not end episodes early.** Square ended an episode as soon as both
  fingers bore on the nut's ring and needed a handle grasp for success; any grasp now
  counts. Drawer ended an episode when the pod was released before its centre was below
  6 cm, though released from the held-above start it lands inside the drawer in all of 256
  trials, so the transport starts learned to hold the pod beside the drawer until the time
  limit. Now a lost grasp starts the grasp again, a released pod can still come to rest in
  the drawer or be grasped again, and a pod disturbed while closing returns the process to
  the release stage. Drawer's process observation keeps only the history the rest of the
  episode depends on.
- **Imagination from episode starts.** `--reset-start-fraction 0.25` draws a quarter of the
  imagined rollouts from reset rows instead of uniformly over the replay buffer.
- **Imagination-only shaping.** Reaching pays only an open hand or a grasp (Drawer until the
  pod is first grasped) and adds a far term, since the native 5-10 cm reaching tanh is flat
  at the full-task start. Square's imagined release test poses each finger's collision boxes
  and requires them to clear the nut by 1 mm, as the native test ends at the last contact.
  Opening away from the peg and holding on to a threaded nut cost, and opening over the peg
  pays: the policy otherwise carried the nut above the peg and hovered there until the time
  limit, although released there the nut drops and self-centres. Drawer's imagined progress
  and reaching count completed stages, so finishing a stage never lowers the reward, and
  opening a held pod away from the drawer and holding it once inside cost. Data collection,
  reported rewards and evaluation use the native task.

To reproduce the two runs (Square seed 1, Drawer seed 0):

```bash
# Reset banks: training (seed 42 by default) and held-out
clab resets collect --task square --output banks/square_train.pt --split train \
  --per-type 9999 --insertion-per-stage 3333
clab resets collect --task square --output banks/square_test.pt --split test \
  --seed 7 --per-type 256
clab resets collect --task drawer --output banks/drawer_train.pt --split train --per-type 2000
clab resets collect --task drawer --output banks/drawer_test.pt --split evaluation \
  --seed 7 --per-type 256

# Square: 10,000 iterations, then 10,000 more with the schedule stretched to 20,000
clab train supermimic --task square --reset-bank banks/square_train.pt \
  --output logs/supermimic/square_s1 --seed 1 \
  --reset-start-fraction 0.25 --cuda-graph-mode substeps
clab train supermimic --task square --reset-bank banks/square_train.pt \
  --resume logs/supermimic/square_s1/model_010000.pt --iterations 10000 \
  --learning-rate-decay-iterations 20000 --output logs/supermimic/square_s1_20k --seed 1 \
  --reset-start-fraction 0.25 --cuda-graph-mode substeps

# Drawer
clab train supermimic --task drawer --reset-bank banks/drawer_train.pt \
  --output logs/supermimic/drawer_s0 --seed 0 \
  --reset-start-fraction 0.25 --cuda-graph-mode substeps

# Held-out evaluation
clab eval policy --task square --algorithm supermimic \
  --checkpoint logs/supermimic/square_s1_20k/model_018000.pt \
  --reset-bank banks/square_test.pt --episodes-per-type 64 --num-envs 64 \
  --seeds 100000 --split-label held_out --cuda-graph-mode substeps \
  --output reports/square_heldout.json
clab eval policy --task drawer --algorithm supermimic \
  --checkpoint logs/supermimic/drawer_s0/model_006000.pt \
  --reset-bank banks/drawer_test.pt --episodes-per-type 64 --num-envs 64 \
  --seeds 100000 --split-label held_out --cuda-graph-mode substeps \
  --output reports/drawer_heldout.json
```

A Square run takes about 29 GB of GPU memory and a Drawer run about 45 GB, so two Drawer
runs do not fit one 80 GB GPU. `--cuda-graph-mode substeps` replays each policy step's
physics substeps from CUDA graphs and is faster than `plain`
([docs/workflow.md](docs/workflow.md)). Results vary strongly between seeds.

## Install

Requirements: Linux, an NVIDIA GPU, [uv](https://docs.astral.sh/uv/) and Git LFS.

```bash
git lfs pull
uv sync --locked --extra dev
uv run clab --help
```

## Quick start

Run inside the project environment (`source .venv/bin/activate`, or prefix with `uv run`).

```bash
clab tasks                                                     # tasks and their contracts
clab resets collect --task box_in_bin --output banks/box_in_bin.pt
clab train supermimic --task box_in_bin --reset-bank banks/box_in_bin.pt \
  --output logs/supermimic/box_in_bin --cuda-graph-mode plain
clab train ppo --task box_in_bin --reset-bank banks/box_in_bin.pt     # baseline
clab eval policy --task box_in_bin --algorithm supermimic \
  --checkpoint logs/supermimic/box_in_bin/model_010000.pt \
  --reset-bank banks/box_in_bin.pt --output reports/box_in_bin.json
```

`--cuda-graph-mode plain` is needed on NVIDIA drivers older than CUDA 12.4. Commands never
overwrite existing outputs.

## Documentation

| Topic | Document |
| --- | --- |
| Method, task adapters and options | [docs/supermimic.md](docs/supermimic.md) |
| Tasks, success conditions and reset categories | [docs/tasks.md](docs/tasks.md) |
| Reset banks, training and evaluation | [docs/workflow.md](docs/workflow.md) |

## Layout

```text
src/clab/
├── algorithms/supermimic/  # world model, actor/critic, imagination, runner, training entry
├── algorithms/ppo/         # PPO baseline on mjlab / RSL-RL
├── tasks/supermimic.py     # shared SuperMimic adapter: state capture, integration, observations
├── tasks/<task>/           # config, MDP terms, props, resets, collector, SuperMimic adapter
├── robots/arx5/            # robot model and assets, OSC preset, binding
├── controllers/            # OSC action, pose helpers, differentiable kinematics
├── evaluation/             # complete-episode evaluation
└── cli/                    # the clab command
```

## Acknowledgements

The SuperMimic implementation adapts [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(MIT, see `src/clab/algorithms/supermimic/REFERENCE_LICENSE`). Task props follow
[robosuite](https://github.com/ARISE-Initiative/robosuite) object models; their licence notices
are kept next to the assets.
