# SuperMimic for robot manipulation

This repository applies SuperMimic, policy optimization by backpropagating the task return
through a learned differentiable world model, to state-based robot manipulation in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp). It contains five manipulation
tasks for a simulated ARX5 arm, reset-bank collectors, SuperMimic and a PPO baseline, and a
complete-episode evaluator.

中文说明：[README.zh-CN.md](README.zh-CN.md)

| Task | Goal | Horizon | Actor / critic | Reset categories |
| --- | --- | --- | --- | --- |
| `box_in_bin` | Put a box in a bin, release it and let it rest | 16 s | 195 / 164 | 4 |
| `lemon` | Put a lemon on a plate and release it | 32 s | 195 / 153 | 4 |
| `square` | Thread a square nut onto its peg, grasping the handle | 16 s | 195 / 153 | 4 |
| `mug_hang` | Hang a mug on a tree by its handle | 16 s | 210 / 240 | 4 |
| `drawer` | Put a pod into an open drawer and push the drawer closed | 60 s | 300 / 81 | 6 |

The arm is driven by an operational-space controller (OSC) at a 20 Hz policy rate and
480 Hz physics. Every task uses the same 7-D action: a relative end-effector pose command in
world axes plus a binary gripper command.

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

Commands never overwrite existing outputs.

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
