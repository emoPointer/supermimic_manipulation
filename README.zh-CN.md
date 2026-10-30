# SuperMimic 机械臂操作（中文入口）

本仓库把 [SuperMimic](https://arxiv.org/abs/2610.28480)（通过可微的学习世界模型反传任务回报来优化策略）
从人形机器人动作跟踪迁移到基于状态的机械臂操作任务，回合的初始状态采用
[OmniReset](https://arxiv.org/abs/2603.15789) 式的 reset 分布。仿真基于
[mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp），机器人为仿真 ARX5 机械臂。包含五个
操作任务及其 reset bank 采集器、SuperMimic 与 PPO 基线，以及按完整回合统计的评估。

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
独立的测试 reset bank。策略只使用低维状态，画面仅用于观看。从初始位姿出发，Square 策略在
64 个回合中成功 94%，Drawer 策略成功 91%；视频都是最先的两个回合，Drawer 为 2 倍速播放。

## 结果

以下都是原生任务成功率（仿真中的原始成功判据并保持 0.5 s），确定性策略、完整回合，没有筛选回合。

**独立测试 reset bank**：每个 reset 类别 64 个回合；测试 bank 与训练 bank 使用不同的随机种子和
划分采集。Lemon、BoxInBin、MugHang 使用第 4,000 轮的 checkpoint（已发布在 Hugging Face，见下文）。
Square 和 Drawer 用视频中那个 run 在训练 bank 上成功率最高的 checkpoint（每 2,000 轮评估一次，
每类 32 个回合；并列时取较早的）。

| 任务 | 训练轮数 | 训练时长 | 总成功率 | 各 reset 类别 |
| --- | --- | --- | --- | --- |
| `lemon` | 4,000 | 1.8 h | **100.0%** | 100 / 100 / 100 / 100 |
| `box_in_bin` | 4,000 | 1.9 h | **94.9%** | 97 / 91 / 95 / 97 |
| `mug_hang` | 4,000 | 2.8 h | **96.9%** | 94 / 98 / 95 / 100 |
| `square` | 16,000 | 10.5 h | **95.0%** | 95 / 98 / 91 / 97 / 94 |
| `drawer` | 4,000 | 3.4 h | **90.2%** | 81 / 97 / 95 / 98 / 77 / 100 / 83 |

训练时长是从启动到该 checkpoint 的实际耗时，包括前 1,000 轮只训练世界模型的阶段，均为单卡
（见[硬件配置](#硬件配置)）。训练时与其他任务共卡，所以这些时间是上限。MugHang 在第 1,000 轮
被中断后从该 checkpoint 继续，时长包含两段。

- Square 和 Drawer 的另一个随机种子分别达到 93.8%（Square，第 14,000 轮）和 75.9%（Drawer，
  第 8,000 轮）。
- Drawer 不同 checkpoint 之间可能波动很大：另一个种子在第 4,000、6,000、8,000、10,000 轮分别为
  78%、24%、76%、69%（上表这个 run：90%、90%、85%、88%）。
- Square 从头训练 20,000 轮，余弦学习率调度覆盖全部 20,000 轮。用默认的 10,000 轮调度时，两个
  种子在第 10,000 轮只有 67% 和 68%。上表这个 run 在第 16,200 轮停止：此时训练 bank 上每一类都
  超过了 90%。

类别 0 是完整任务（物体在桌上、夹爪空闲），之后的类别从任务中途开始；Square 和 Drawer 的
最后一类是张开的手停在物体正上方（见下文）。

**数据量**：每轮迭代从 4,096 个并行环境采集 10,240 条仿真转移：4,000 轮共 4,100 万条（包括
Drawer，按 20 Hz 约合 24 天仿真时间），Square 1.64 亿条（95 天）。策略只通过世界模型训练，没有
使用任何策略梯度（PPO）更新。训练 bank 每类 10,000 个初始状态（Square 9,999，Drawer 1,993），
测试 bank 每类 256 个（Square 和 Drawer 255）。

## 硬件配置

每个训练 run 使用同一台服务器上的一张 GPU：

| 部件 | 配置 |
| --- | --- |
| GPU | 每个 run 1 张 NVIDIA A100-SXM4-80GB（驱动 535.230，CUDA 12.2），每个 run 约占 27 GB 显存（Square 29 GB，Drawer 45 GB） |
| CPU | Intel Xeon Platinum 8336C（112 线程），服务器上所有任务共用 |
| 软件 | PyTorch 2.9.1（CUDA 12.8 版本）、mjlab 1.5.3、MuJoCo 3.10、MuJoCo Warp、Warp 1.17 |
| 仿真 | 4,096 个并行环境；CUDA Graph 用 `--cuda-graph-mode plain`（Lemon、BoxInBin、MugHang）或 `substeps`（Square、Drawer），驱动低于 CUDA 12.4 时都可用 |
| 精度 | MLP 使用 BF16 autocast，参数为 FP32，矩阵乘启用 TF32 |

每轮迭代（采集、世界模型更新、一次 actor/critic 更新）约 1.4–3.5 秒，取决于任务和是否共卡。

## 思路

SuperMimic 原本用于人形机器人动作跟踪，每一步都有稠密的跟踪奖励。机械臂操作不同：决定性的
奖励是一个稀疏的成功判据，而且依赖接触（物体被支撑、已松手、静止）；任务长度也远超策略在
模型里能向前看的 16 步。我们的迁移分四部分：

1. **用初始状态缩短任务（OmniReset）。** 参照 OmniReset，每个任务有一个经过物理验证的
   reset bank，分四类初始状态：完整任务（物体在桌上、夹爪空闲）、物体在桌上已抓住、
   物体在空中已抓住、物体在目标附近已抓住（Square 和 Drawer 的类别更多，见下文）。训练时均匀
   采样各类。靠后的类别让成功判据落在 16 步想象 rollout 的范围内，在那里学到的价值再一步步传回
   完整任务。初始状态由程序生成（抓取状态用逆运动学求解），只保留在任务自身物理和控制器下能
   保持有效的状态。

2. **世界模型只预测物理状态。** 机械臂由操作空间控制器（OSC）驱动，每个动作都以当前末端
   位姿为基准，控制器没有内部记忆。世界模型根据实际执行的控制指令，预测关节以及物体位姿和
   速度的增量（共 28 维，Drawer 另加滑轨），并以该回合随机化的物理参数为条件（各维度的含义见
   [docs/supermimic.md](docs/supermimic.md#world-model)）。想象中的策略观测由预测状态经
   可微正运动学重建，在真实状态上与仿真器的观测误差约 1e-6。

3. **解析奖励和可微的成功判据。** 奖励不靠学习，而是用任务本身的权重从想象状态直接算出。
   依赖接触的成功判据换成几何代理（例如用"两指张开超过阈值"代替"手指不接触物体"）：
   前向是硬判定，反向用 sigmoid 直通梯度。这样 critic 看到的是真实的 +10 成功奖励，actor
   得到可用的梯度，并乘以 0.02 防止它主导优化。0.5 s 的保持计时作为解析计数器传递。

4. **只在想象中加的引导项，补上原生奖励没有梯度的地方。** 物体放好后，接近奖励停止，
   改为奖励张开手指，因此"一直抓着"永远不会比"松手"更划算。MugHang 的原生挂杯奖励在杯把
   没对准横杆时为零，所以想象中额外加了一个把杯把拉向挂点的长程引导。这些项只存在于想象中；
   数据采集、报告的奖励和所有评估都使用未修改的原生任务。

每轮迭代：用当前策略加高斯噪声采集真实转移；用 8 步开环窗口拟合世界模型；从回放状态出发
想象 16 步，通过时间反向传播最大化想象回报加 TD(λ) critic 的价值；再拟合 critic。前 1,000
轮只训练世界模型。详见 [docs/supermimic.md](docs/supermimic.md)。Square 和 Drawer 还需要更多
改动，见 [Square 与 Drawer](#square-与-drawer)。

## 安装

需要 Linux、NVIDIA GPU、[uv](https://docs.astral.sh/uv/) 和 Git LFS。

```bash
git lfs pull
uv sync --locked --extra dev
source .venv/bin/activate        # 或在每条 clab 命令前加 `uv run`
```

下面命令中的 `--cuda-graph-mode plain` 在驱动低于 CUDA 12.4 时需要，新驱动可以去掉。
所有命令都不会覆盖已有输出。

## 训练好的模型与 reset bank

| Hugging Face 仓库 | 内容 |
| --- | --- |
| [emopointer/supermimic-manipulation](https://huggingface.co/emopointer/supermimic-manipulation) | 结果表中 Lemon、BoxInBin、MugHang 的 checkpoint：`<task>/model_004000.pt`（世界模型、actor、critic 与优化器状态），训练设置在 `<task>/params/` |
| [emopointer/supermimic-manipulation-reset-banks](https://huggingface.co/datasets/emopointer/supermimic-manipulation-reset-banks) | 训练用 bank `<task>_train.pt`（每类 10,000 个状态）和独立测试 bank `<task>_test.pt`（每类 256 个） |

`<task>` 为 `lemon`、`box_in_bin` 或 `mug_hang`。用 Hugging Face 命令行下载（通过 `uvx` 运行，
不会往环境里加依赖）：

```bash
uvx --from huggingface_hub hf download emopointer/supermimic-manipulation \
  --local-dir checkpoints
uvx --from huggingface_hub hf download emopointer/supermimic-manipulation-reset-banks \
  --repo-type dataset --local-dir banks
```

Square 和 Drawer 的 checkpoint 与 bank 没有发布；采集、训练和评估它们的命令见
[README.md](README.md#square-and-drawer)。

## 流程 A：直接使用训练好的 checkpoint

按上面下载两个仓库，然后在独立测试 bank 上评估，即可复现结果表（256 个完整回合，每个
reset 类别 64 个）：

```bash
TASK=box_in_bin        # lemon | box_in_bin | mug_hang
clab eval policy --task $TASK --algorithm supermimic \
  --checkpoint checkpoints/$TASK/model_004000.pt \
  --reset-bank banks/${TASK}_test.pt --episodes-per-type 64 --num-envs 64 \
  --seeds 100000 --split-label held_out --cuda-graph-mode plain \
  --output reports/${TASK}_heldout.json
```

报告中有总成功率、各类别成功率、回合长度和回报。`checkpoints/$TASK/params/` 要和
checkpoint 放在一起，评估时会用它检查训练环境与当前环境是否一致。

如果想在训练好的模型上继续训练，用 `--resume`。`--iterations` 表示再训练的轮数，再训
6,000 轮即达到默认的总共 10,000 轮。世界模型、actor、critic、优化器和迭代计数都会恢复；
回放缓存没有保存，会先重新采集。

```bash
clab train supermimic --task $TASK --reset-bank banks/${TASK}_train.pt \
  --resume checkpoints/$TASK/model_004000.pt --iterations 6000 \
  --output logs/supermimic/${TASK}_resumed --cuda-graph-mode plain
```

## 流程 B：从零开始训练

1. **采集 reset bank**：训练 bank 和用不同随机种子、不同划分采集的独立测试 bank
   （也可以直接下载上面发布的 bank，跳过这一步）。

   ```bash
   TASK=box_in_bin        # lemon | box_in_bin | mug_hang
   clab resets collect --task $TASK --output banks/${TASK}_train.pt --split train
   clab resets collect --task $TASK --output banks/${TASK}_test.pt \
     --split test --seed 7 --per-type 256
   ```

2. **训练**：默认 10,000 轮，每 1,000 轮保存一次 checkpoint，结果表用的是
   `model_004000.pt`；前 1,000 轮只训练世界模型。

   ```bash
   clab train supermimic --task $TASK --reset-bank banks/${TASK}_train.pt \
     --output logs/supermimic/$TASK --seed 1 --cuda-graph-mode plain
   ```

   训练过程记录在 TensorBoard（`tensorboard --logdir logs/supermimic`），包括训练回合中
   各类别的成功率。

3. **评估**：在独立测试 bank 上评估。

   ```bash
   clab eval policy --task $TASK --algorithm supermimic \
     --checkpoint logs/supermimic/$TASK/model_004000.pt \
     --reset-bank banks/${TASK}_test.pt --episodes-per-type 64 --num-envs 64 \
     --seeds 100000 --split-label held_out --cuda-graph-mode plain \
     --output reports/${TASK}_heldout.json
   ```

Square 和 Drawer 的 bank、训练和评估设置不同，命令见 [README.md](README.md#square-and-drawer)。
同一环境上的 PPO 基线用 `clab train ppo`（见 [docs/workflow.md](docs/workflow.md)）。

## 任务

| 任务 | 目标 | 时限 | Actor / Critic 输入 | Reset 类别 |
| --- | --- | --- | --- | --- |
| `lemon` | 把柠檬放到盘子上并松手 | 32 s | 195 / 153 | 4 |
| `box_in_bin` | 把箱子放进容器、松手并静止 | 16 s | 195 / 164 | 4 |
| `mug_hang` | 把杯子按杯把挂到杯架上 | 16 s | 210 / 240 | 4 |
| `square` | 把方形螺母套到柱子上并松手 | 16 s | 195 / 153 | 5 |
| `drawer` | 把胶囊放进打开的抽屉并推关抽屉 | 60 s | 265 / 74 | 7 |

所有任务使用同一个 7 维动作（世界坐标系下的相对末端位姿指令加二值夹爪指令），策略 20 Hz，
物理 480 Hz。观测、奖励、成功判据和 reset 类别见 [docs/tasks.md](docs/tasks.md)。

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

只用上面的通用迁移方法时，这两个任务的完整任务策略第一步就合上夹爪、再也不张开，已抓住的
初始状态则立刻松手。它们还需要：

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

复现这两个 run（Square 种子 1，Drawer 种子 1）的命令见 [README.md](README.md#square-and-drawer)。
一个 Square run 约占 29 GB 显存，一个 Drawer run 约 45 GB，所以一张 80 GB 的卡放不下两个
Drawer run。不同随机种子之间结果差异很大。

## 文档导读

| 文档 | 内容 |
| --- | --- |
| [docs/supermimic.md](docs/supermimic.md) | 方法、各任务适配器（想象中的奖励与成功判据）、训练选项 |
| [docs/tasks.md](docs/tasks.md) | 五个任务的观测、奖励、成功条件和 reset 类别 |
| [docs/workflow.md](docs/workflow.md) | reset bank 采集、训练与评估流程 |

## 引用

本项目是独立的迁移实现，与两篇论文的作者无关，也不复现论文中的实验。BibTeX 见
[README.md](README.md#references)。

- **SuperMimic**：Q. Liao 等，*SuperMimic: Scaling Humanoid Whole-Body Control with One
  Differentiable World Model*，[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)，2026。
  算法实现改编自开源复现 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
  （MIT 许可，原许可证保留在 `src/clab/algorithms/supermimic/REFERENCE_LICENSE`）。
- **OmniReset**：P. Yin 等，*Emergent Dexterity via Diverse Resets and Large-Scale
  Reinforcement Learning*，ICLR 2026，[arXiv:2603.15789](https://arxiv.org/abs/2603.15789)，
  [项目主页](https://weirdlabuw.github.io/omnireset/)，
  代码 [UW-Lab/UWLab](https://github.com/UW-Lab/UWLab)。本项目的 reset 类别参照其 reset 分布，
  采集器为我们自己实现。

## 许可

代码以 [MIT 许可](LICENSE) 发布。第三方代码和资产沿用各自的许可：改编自 supermimic_vibed 的代码
（MIT，见 `src/clab/algorithms/supermimic/REFERENCE_LICENSE`）、来自 robosuite 的物体模型
（MIT，见 `src/clab/tasks/lemon/assets/ROBOSUITE_LICENSE`），以及 MugHang 使用的 ShapeNet
杯子模型，须遵守 [ShapeNet 使用条款](https://shapenet.org/terms)（仅限非商业研究与教学）。
