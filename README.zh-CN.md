# SuperMimic 机械臂操作（中文入口）

本仓库把 [SuperMimic](https://arxiv.org/abs/2610.28480)（通过可微的学习世界模型反传任务回报来优化策略）
用于基于状态的机械臂操作任务，仿真基于 [mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp）。
包含仿真 ARX5 机械臂的五个操作任务、[OmniReset](https://arxiv.org/abs/2603.15789) 式 reset 类别的
reset bank 采集器、SuperMimic 与 PPO 基线，以及按完整回合统计的评估。

详细文档以英文为准（[README.md](README.md) 与 [docs/](docs/)），本页给出概览。

<table>
  <tr>
    <th>Lemon：把柠檬放到盘子上</th>
    <th>BoxInBin：把箱子放进容器</th>
    <th>MugHang：把杯子按杯把挂上杯架</th>
  </tr>
  <tr>
    <td><img src="docs/media/lemon.gif" width="100%"></td>
    <td><img src="docs/media/box_in_bin.gif" width="100%"></td>
    <td><img src="docs/media/mug_hang.gif" width="100%"></td>
  </tr>
  <tr>
    <th>Square：把螺母套到柱子上</th>
    <th>Drawer：把胶囊放进抽屉并推关（2 倍速）</th>
    <th></th>
  </tr>
  <tr>
    <td><img src="docs/media/square.gif" width="100%"></td>
    <td><img src="docs/media/drawer.gif" width="100%"></td>
    <td></td>
  </tr>
</table>

每段视频是训练好的策略的两个回合：机械臂从各关节为 0 的初始位姿出发，物体和目标的摆放来自
独立的测试 reset bank。策略只使用低维状态，画面仅用于观看。从主页姿态出发，Square 策略在
64 个回合中成功 89%，视频是最先的两个回合；Drawer 策略成功 53%，视频是最先的两个成功回合
（共试了 6 个回合），2 倍速播放。

## 结果

以下都是原生任务成功率（仿真中的原始成功判据并保持 0.5 s），确定性策略、完整回合，在与训练
bank 使用不同随机种子和划分采集的独立测试 bank 上评估，每个 reset 类别 64 个回合，没有筛选回合。

| 任务 | 训练轮数 | 训练时长 | 总成功率 | 各 reset 类别 |
| --- | --- | --- | --- | --- |
| `lemon` | 4,000 | 1.8 h | **100.0%** | 100 / 100 / 100 / 100 |
| `box_in_bin` | 4,000 | 1.9 h | **94.9%** | 97 / 91 / 95 / 97 |
| `mug_hang` | 4,000 | 2.8 h | **96.9%** | 94 / 98 / 95 / 100 |
| `square` | 18,000 | 17.5 h | **96.6%** | 98 / 95 / 95 / 98 / 95 |
| `drawer` | 6,000 | 4.5 h | **77.2%** | 50 / 80 / 86 / 92 / 75 / 97 / 61 |

- Lemon、BoxInBin、MugHang 用 `main` 分支发布的第 4,000 轮 checkpoint。Square 和 Drawer 用视频
  中那个 run 在训练 bank 上成功率最高的 checkpoint（每 2,000 轮评估一次，每类 32 个回合）。
  另一个随机种子分别达到 78.1%（Square，第 18,000 轮）和 68.1%（Drawer，第 8,000 轮）。
- Drawer 不同 checkpoint 之间波动很大：上表这个 run 在第 4,000、6,000、8,000、10,000 轮分别为
  59%、77%、22%、61%。
- Square 先训练 10,000 轮，再续训 10,000 轮，续训时余弦学习率调度拉长到 20,000 轮，学习率回到
  初始值的约一半；训练时长是两段之和。
- 训练时长是从启动到该 checkpoint 的实际耗时，包括前 1,000 轮只训练世界模型的阶段，单张
  A100-SXM4-80GB。训练时与其他任务共卡（Square 的两个种子共用一张卡），所以这些时间是上限。
- 类别 0 是完整任务（物体在桌上、夹爪空闲），之后的类别从任务中途开始；Square 和 Drawer 的
  最后一类是张开的手停在物体正上方（见下文）。

**数据量**：每轮迭代从 4,096 个并行环境采集 10,240 条仿真转移：4,000 轮共 4,100 万条（按 20 Hz
约合 24 天仿真时间），Square 1.85 亿条（107 天），Drawer 6,200 万条（36 天）。策略只通过世界
模型训练，没有使用任何策略梯度（PPO）更新。训练 bank 每类 10,000 个初始状态（Square 9,999，
Drawer 1,993），测试 bank 每类 256 个（Drawer 255）。

## 任务

| 任务 | 目标 | 时限 | Actor / Critic | Reset 类别 |
| --- | --- | --- | --- | --- |
| `box_in_bin` | 把箱子放进容器、松手并静止 | 16 s | 195 / 164 | 4 |
| `lemon` | 把柠檬放到盘子上并松手 | 32 s | 195 / 153 | 4 |
| `square` | 把方形螺母套到柱子上并松手 | 16 s | 195 / 153 | 5 |
| `mug_hang` | 把杯子按杯把挂到杯架上 | 16 s | 210 / 240 | 4 |
| `drawer` | 把胶囊放进打开的抽屉并推关抽屉 | 60 s | 265 / 74 | 7 |

机械臂由操作空间控制器（OSC）驱动：策略 20 Hz，物理 480 Hz。所有任务使用同一个 7 维动作：
世界坐标系下的相对末端位姿指令加二值夹爪指令。观测、奖励、成功判据和 reset 类别见
[docs/tasks.md](docs/tasks.md)。

## Square 与 Drawer

**Square**：抓起方形螺母，移到柱子上方，套进柱子后松手。成功条件：柱子穿过螺母的孔，螺母落在
底座上（高度 ±2 mm，倾斜不超过 5°），机器人完全松开、螺母静止，并保持 0.5 s。抓螺母任何位置都
算抓取；接近奖励引向把手，因为抓把手不会挡住孔。五类初始状态：螺母在桌上、夹爪空闲；已抓住
把手，螺母分别在桌上、在空中、部分套入（停在柱子旁、柱子上方或已套上）；以及张开的手停在静置
螺母的把手上方 1–6 cm。

**Drawer**：抓起咖啡胶囊，放进初始拉开 13 cm 的抽屉，在抽屉里松手，再把抽屉推关。原生流程：
稳定的两指抓取保持 0.3 s 并抬起 4 cm → 在抽屉内松手保持 0.3 s → 接触推动使滑轨至少关上
1.5 cm；成功条件是流程完成、滑轨小于 3 cm、胶囊在抽屉内且未被抓住，并保持 0.5 s。除了 60 s
时限，没有别的情况会提前结束回合。reset bank 由每条完整物理轨迹的六个快照（胶囊在桌上、已
抓住、已抬起、举在抽屉上方、已在抽屉内松手、抽屉推到一半）加上张开的手停在胶囊上方组成。

只用各任务共用的迁移方法（[docs/supermimic.md](docs/supermimic.md)）时，这两个任务的完整任务
策略第一步就合上夹爪、再也不张开，已抓住的初始状态则立刻松手。它们还需要：

- **预抓取（pre-grasp）初始类别**：完整任务（夹爪常在半米之外）和第一个已抓住的类别之间没有
  过渡，critic 从未学到"在物体旁张开手"的价值。每个预抓取状态都由"已在桌上抓住"的状态经物理
  仿真得到（张开、抬起、把张开的手停在物体上方 1–6 cm），只保留物体保持静止、未被碰到的状态。
- **不提前结束回合的任务规则**：Square 原来只要两指压在螺母的环上就结束回合，且成功要求先抓过
  把手；现在任何抓取都算。Drawer 原来在胶囊中心还没低于 6 cm 时松手就结束回合，但从"举在抽屉
  上方"的状态松手，256 次全部落进抽屉，于是搬运类别学会把胶囊停在抽屉旁直到超时。现在抓取丢失
  就重新开始抓取，松手的胶囊仍可能落进抽屉或被重新抓起，推关时胶囊被扰动则流程退回松手阶段。
  Drawer 的流程观测只保留对之后回合有影响的历史。
- **从回合起点开始想象**：`--reset-start-fraction 0.25` 让四分之一的想象 rollout 从 reset 状态
  出发，而不是从回放缓存中均匀采样。
- **只在想象中加的引导项**：接近奖励只奖励张开的手或已抓住的状态（Drawer 只到第一次抓住胶囊
  为止），并加一个远距离项，因为原生 5–10 cm 的 tanh 在完整任务起点几乎是平的。Square 想象中
  的松手判据把每根手指的碰撞盒摆到当前位姿，要求离螺母至少 1 mm，与原生判据（最后一次接触结束）
  一致。离柱子远时张手、螺母套上柱子后还抓着不放都要付代价，在柱子上方张手有奖励——否则策略会把
  螺母举在柱子上方悬停到超时，而在那里松手螺母会自己落下并对正。Drawer 想象中的进度和接近奖励
  计入已完成的阶段，完成一个阶段不会让奖励下降；离开抽屉松开胶囊、胶囊进抽屉后还抓着不放都要付
  代价。数据采集、报告的奖励和评估都使用原生任务。

复现这两个 run（Square 种子 1，Drawer 种子 0）的命令见 [README.md](README.md#square-and-drawer)。
一个 Square run 约占 29 GB 显存，一个 Drawer run 约 45 GB，所以一张 80 GB 的卡放不下两个
Drawer run。不同随机种子之间结果差异很大。

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
