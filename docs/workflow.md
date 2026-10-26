# Workflow

Run commands inside the project environment (`source .venv/bin/activate`, or prefix with
`uv run`). Every command writes a new output and refuses to overwrite an existing one.
`clab <command> --help` lists all options.

## 1. Reset banks

Episodes start from a reset bank: physically validated initial states, grouped by reset
category (see [tasks.md](tasks.md)). Each task has its own collector:

```bash
clab resets collect --task box_in_bin --output banks/box_in_bin.pt --split train
clab resets collect --task drawer --output banks/drawer.pt --split train --per-type 2000
```

Collectors sample procedural proposals on CPU worker processes (`--proposal-workers`), solve
inverse kinematics for grasped categories, then simulate every candidate in parallel
(`--num-envs`) and accept only states that stay valid for a hold period under the task's
own physics and controller. Each accepted state is reset once more from its serialized form
and must pass the same checks. Startup physics randomization is off during collection;
training keeps it.

| Option | Meaning |
| --- | --- |
| `--per-type N` | States per category (default 10,000) |
| `--num-envs N` | Parallel simulated candidates per batch |
| `--split` | Label stored in the bank; use separate seeds and splits for training and evaluation banks |
| `--seed S` | Proposal and physics seed |
| `--cuda-graph-mode plain` | Use plain CUDA graphs (needed on drivers older than CUDA 12.4) |

Square and MugHang build their insertion/threading categories in stages
(`--insertion-per-stage`); for Square use it (for example `--per-type 9999
--insertion-per-stage 3333`), since the near-peg and above-peg stages bridge the gap
between the far carries and insertion. Drawer records complete physical trajectories and saves six
snapshots from each, so it is the slowest collector; a few thousand states per category
are enough for training.

The Square and Drawer collectors finish by deriving their pre-grasp category from the
grasped-on-table rows (see [tasks.md](tasks.md)). A bank collected before that category
existed can be completed without recollecting:

```bash
clab resets pregrasp --task square --source banks/square_old.pt --output banks/square.pt
```

A bank records the controller and timing it was collected with. Training refuses a bank
whose controller, physics step or decimation differ from the current task.

## 2. Training

### SuperMimic

```bash
clab train supermimic --task box_in_bin --reset-bank banks/box_in_bin.pt \
  --output logs/supermimic/box_in_bin_s0 --seed 0 --cuda-graph-mode plain
```

Defaults: 4,096 worlds, 10,000 iterations with a matching cosine learning-rate decay,
1,000 world-model-only iterations, a checkpoint every 1,000 iterations. The run directory
holds `params/` (environment, agent, adapter and runtime settings), `model_<iteration>.pt`
and TensorBoard logs. `--smoke` runs a tiny configuration end to end and reloads the final
checkpoint; use it after any task or adapter change. See [supermimic.md](supermimic.md) for
the method and options.

Results vary strongly between seeds; compare configurations over several seeds.

### PPO baseline

```bash
clab train ppo --task box_in_bin --reset-bank banks/box_in_bin.pt --run-name s0 --seed 0
```

Runs RSL-RL PPO on the same environment, observations and native rewards. Logs go to
`logs/rsl_rl/<experiment>/<timestamp>_<run-name>/`; `--gpu-ids` trains on several GPUs.

## 3. Evaluation

```bash
clab eval policy --task box_in_bin --algorithm supermimic \
  --checkpoint logs/supermimic/box_in_bin_s0/model_010000.pt \
  --reset-bank banks/box_in_bin_eval.pt --episodes-per-type 64 --num-envs 64 \
  --seeds 100000 --output reports/box_in_bin_s0.json
```

The evaluator loads only the actor and its frozen observation normalizer, rebuilds the
environment from the run's saved `params/env.yaml`, and checks that observation and action
contracts match. It runs complete episodes from every reset category and reports native
success per category and overall, episode length and returns. Category 0 is the full task.

Success on the training bank is a development diagnostic; report results on a bank
collected with a different seed and split (`--split-label held_out`).
