# world-models-study

[English](README.md) · **中文**

两种世界模型，先复现，再拆开验证。

**Dreamer** 通过重建观测来学习潜在动力学，然后让策略完全在这个模型的想象里训练。
**JEPA**——LeCun 一直主张的那条路线——干脆丢掉重建，只在表征空间里预测，规划放到测试时用搜索完成。
这两条路线通常被当作对立的哲学来讨论。这个仓库记录的是：我把其中一条从零写了一遍，把另一条跑通，
然后花时间去试图推翻后者的核心主张。

结论的短版本：**验证 loss 并不能说明模型到底会不会规划。**
我整晚训出来预测 loss 最低的那个 checkpoint，恰恰是规划最差的之一——因为它的 encoder 悄悄塌缩了。
而这正是论文里那个正则项要防的失败模式，所以我没有靠 loss 曲线去推断，而是直接把它测了出来。

---

## 目录

| | |
|---|---|
| [`01_dreamer_from_scratch/`](01_dreamer_from_scratch) | 从零手写的 DreamerV2，单文件。RSSM、reward model、actor-critic、imagination rollout。CartPole 上对比 Double-DQN，3 个 seed。 |
| [`02_lewm_reproduction/`](02_lewm_reproduction) | 跑通 [LeWorldModel](https://github.com/lucas-maes/le-wm) 并复现它发布的 checkpoint。这个目录里大部分笔记是关于安装的——时间实际花在了那儿。 |
| [`03_experiments/`](03_experiments) | 我真正在意的部分。三组针对 LeWorldModel 的受控实验。 |
| [`04_analysis/`](04_analysis) | `summary.csv`——每个 checkpoint、每项指标，一张表——以及画图的脚本。 |
| [`05_dreamerv3_scale/`](05_dreamerv3_scale) | 官方 DreamerV3 在 DMC walker-walk 上跑了一整夜，跑到任务天花板。 |
| [`notes/engineering_log_zh.md`](notes/engineering_log_zh.md) | 什么坏了，各花了多少时间。 |
| [`slides/`](slides) | 讲稿（中英双版），中文版附 PDF。 |

## 第一部分 —— 手写 Dreamer

这部分的意义在于：不能躲在别人的 `train.py` 后面。
Encoder、RSSM（GRU 确定性状态 + 32×16 categorical 随机状态，straight-through 梯度）、
reward head、continue head、基于 λ-return 的 actor-critic，以及 imagination rollout，约 670 行。

CartPole-v1，2 万环境步，3 个 seed，对比 Double-DQN：

| | 全程平均评估回报 | 收敛段（最后 3 次评估） |
|---|---|---|
| DreamerV2 | **240.1** | **248.6** |
| Double-DQN | 113.8 | 162.2 |

![](01_dreamer_from_scratch/results/comparison_multiseed.png)

样本效率的差距是预期内的结果，不是有意思的部分。有意思的是诊断图：
KL 稳定在 0.8 附近（free bits 正在起作用，既没塌缩也没爆炸），
以及**在想象内部**算出的 λ-return 从 0 涨到 70+，与真实表现同步。
后者是在检查一件事——想象出来的轨迹是不是真的对应现实，
还是说模型正舒服地待在自己编造的幻觉里。

![](01_dreamer_from_scratch/results/diagnostics.png)

### 规模验证

CartPole 能证明组件是对的，但产出不了一条像样的曲线。所以官方
[dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) 也在 DMC walker-walk 上跑了一夜：

![](04_analysis/figures/dreamerv3_walker.png)

5k 步时 31 分，45k 时 568，195k 时 951，之后在 930–955 区间走平，天花板约 1000。
细节和它需要的那一处补丁在 [`05_dreamerv3_scale/`](05_dreamerv3_scale)。

## 第二部分 —— 复现 LeWorldModel

发布的 tworoom checkpoint，用 CEM-MPC 在 50 条留出 episode 上规划：**86% 成功率**。
这是第三部分所有数字的参照基准。

那一夜大约四个小时花在装这个东西上，而不是跑它——缺 `swig`、`datasets` 解析到五年前的版本、
`transformers` 大版本重命名了所有 ViT 权重、HDF5 支持在包里但被一个静默的 `ImportError` 关掉了。
全部写进了 [`02_lewm_reproduction/setup_zh.md`](02_lewm_reproduction/setup_zh.md)——
因为这恰恰是复现类文章通常略过、而实际上真正卡住人的部分。

## 第三部分 —— 三组实验

下面每一组都是：同一份数据、同一个 encoder、同一个优化器、同样 900 步梯度更新、
同一个 CEM 规划器、同样 50 条留出 episode。每组只动一个变量。

### 1. predictor 需要多少上下文？

`history_size` ∈ {1, 3, 5}——predictor 能看到几帧过去的 embedding。

![](04_analysis/figures/exp1_horizon.png)

预测误差随上下文单调下降。规划成功率没有跟着走：h=1 → h=3 这一步，
loss 改善了而成功率**掉了**（54% → 52%）。每组 50 条 episode，二项标准误约 7 个百分点，
所以诚实的读法是"h=5 更好，其余是噪声"，而不是一条规律。
而 loss 曲线暗示的东西和评估实际做的事之间这道错位，正是后面两组实验要追的线索。

### 2. 同预算，换架构

PLDM 是 le-wm 对比的 baseline 之一。它的构造函数签名和 LeWM 的 JEPA 一致，
`encode`/`predict` 接口也一样，所以只改模型配置里的 `_target_` 路径，
就能塞进同一个训练循环和同一个规划器。其他什么都不变。

| | pred_loss | CEM 成功率 |
|---|---|---|
| LeWM h=3 | 0.266 | 52% |
| **PLDM** | **0.298** | **66%** |

PLDM 预测更*差*，规划更*好*。单 seed，所以别当定论——但它是一个直接的反例：
loss 这一列不能当排序用。

（一个必须说明的 caveat：这训的是 PLDM 的**架构**配 LeWM 的**训练配方**，不是 PLDM 自己的。
这样做是为了隔离架构变量，但因此它不能被当作对 PLDM 论文的复现来引用。）

### 3. 那个正则项，真的在做论文说的事吗？

LeWorldModel 的卖点是两项 loss 就够：next-embedding 预测，
加一个把 embedding 拉向各向同性高斯、使其无法塌缩的 SIGReg 项。默认权重 0.09。我把它扫到 0.001 和 1.0。

**loss 这一列回答不了这个问题**，这正是陷阱所在——encoder 塌缩会让预测变得*更容易*，
所以 `pred_loss` 反而会降。于是我没有去读 loss，而是写了
[`probe_collapse.py`](03_experiments/exp3_sigreg/probe_collapse.py)：
编码 256 帧真实的留出数据，测量每个维度的标准差，数出有多少维实际上已经死了。

![](04_analysis/figures/exp3_sigreg.png)

权重 0.001 时，**192 维 embedding 里有 69 维在真实数据上方差趋近于零**。
模型找到了作弊解。它的 `pred_loss` 是 0.004——比这里训出来的任何东西低两个数量级，
而且完全没有意义。规划成功率：34%，全场最差。

10 倍权重那一头不会塌缩，但 embedding 被压得太紧、失去区分度——38%。

论文选的默认值正好落在这个倒 U 的顶点。这不是空泛的"正则化有用"，
而是一个具体的、已发表的超参数，被拿它自己声称要防的失败模式、按那个失败模式本身的定义验证了一遍。

### 六个 checkpoint 放一起

![](04_analysis/figures/loss_vs_success.png)

如果验证 loss 是对的代理指标，这张图应该呈现一条向右下的趋势。它没有。
完整数字在 [`04_analysis/summary.csv`](04_analysis/summary.csv)。

---

## 这是什么，不是什么

**是**：两种世界模型的复现，以及三组每次只动一个变量的受控实验，
其中一组通过测量机制本身、而非测量指标，验证了一个已发表的主张。

**不是**可发表级别的结果。每一组 LeWorldModel 实验都是单 seed、900 步梯度更新、50 条评估，
而同一协议下发布的 checkpoint 能到 86%——所以这些模型全都欠训练，小于约 10 个百分点的差距都是噪声。
Dreamer 那一半更规矩（3 个 seed，外加一条跑到天花板的 walker-walk 完整曲线），但 CartPole 规模有限。
这里没有 Dreamer 与 LeWorldModel 的正面对比，因为二者优化的目标不同
（奖励 vs 与目标 embedding 的距离），评估的指标也不同（累计回报 vs 成功率）；
硬把它们塞进同一根轴，需要一个我没有搭建的共同任务定义。

## 复现

每一部分都有自己的运行脚本。第一部分是自包含的（`pip install gymnasium torch`）；
第二、三部分需要一份 [le-wm](https://github.com/lucas-maes/le-wm) 的 checkout——
**先读 [`02_lewm_reproduction/setup_zh.md`](02_lewm_reproduction/setup_zh.md)**，
截至 2026 年 9 月，默认安装方式是跑不通的。

```bash
cd 01_dreamer_from_scratch && ./run_experiments.sh     # 约 20 分钟，单卡
cd 03_experiments/exp1_horizon && ./run.sh             # 约 15 分钟
cd 03_experiments/exp2_pldm   && ./run.sh              # 约 5 分钟
cd 03_experiments/exp3_sigreg && ./run.sh              # 约 10 分钟
cd 04_analysis && python make_figures.py
```

DreamerV3 那条线是独立的整夜任务，见 [`05_dreamerv3_scale/`](05_dreamerv3_scale)。

硬件：单张 RTX 5880 Ada。一组 LeWorldModel 实验在 batch 128、224px 下峰值约 13 GB；
Dreamer 那一半 2 GB 以内。

## 参考

- Hafner et al., *Mastering Atari with Discrete World Models*（DreamerV2）与
  *Mastering Diverse Domains through World Models*（DreamerV3）
- Maes, Le Lidec, Scieur, LeCun, Balestriero, *LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels*, arXiv:2603.19312
- [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)、
  [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)、
  [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
