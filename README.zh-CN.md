# SuperMimic 机械臂操作（中文入口）

本仓库把 SuperMimic（通过可微的学习世界模型反传任务回报来优化策略）用于基于状态的机械臂
操作任务，仿真基于 [mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp）。包含仿真 ARX5
机械臂的五个操作任务、reset bank 采集器、SuperMimic 与 PPO 基线，以及按完整回合统计的评估。

详细文档以英文为准，本页给出概览。

| 任务 | 目标 | 时限 | Actor / Critic | Reset 类别 |
| --- | --- | --- | --- | --- |
| `box_in_bin` | 把箱子放进容器、松手并静止 | 16 s | 195 / 164 | 4 |
| `lemon` | 把柠檬放到盘子上并松手 | 32 s | 195 / 153 | 4 |
| `square` | 把方形螺母套到柱子上并松手 | 16 s | 195 / 153 | 5 |
| `mug_hang` | 把杯子按杯把挂到杯架上 | 16 s | 210 / 240 | 4 |
| `drawer` | 把胶囊放进打开的抽屉并推关抽屉 | 60 s | 300 / 81 | 6 |

机械臂由操作空间控制器（OSC）驱动：策略 20 Hz，物理 480 Hz。所有任务使用同一个 7 维动作：
世界坐标系下的相对末端位姿指令加二值夹爪指令。

## 安装与快速上手

```bash
git lfs pull
uv sync --locked --extra dev
clab tasks
clab resets collect --task box_in_bin --output banks/box_in_bin.pt
clab train supermimic --task box_in_bin --reset-bank banks/box_in_bin.pt \
  --output logs/supermimic/box_in_bin --cuda-graph-mode plain
clab eval policy --task box_in_bin --algorithm supermimic \
  --checkpoint logs/supermimic/box_in_bin/model_010000.pt \
  --reset-bank banks/box_in_bin.pt --output reports/box_in_bin.json
```

## 文档导读

| 文档 | 内容 |
| --- | --- |
| [docs/supermimic.md](docs/supermimic.md) | 方法、各任务适配器（想象中的奖励与成功判据）、训练选项 |
| [docs/tasks.md](docs/tasks.md) | 五个任务的观测、奖励、成功条件和 reset 类别 |
| [docs/workflow.md](docs/workflow.md) | reset bank 采集、训练与评估流程 |
