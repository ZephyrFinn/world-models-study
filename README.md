# world-models-study

[English](README_en.md) · **中文**

两种世界模型，先复现，再拆开验证。

**Dreamer** 通过重建观测来学习潜在动力学，然后让策略完全在这个模型的想象里训练。
**JEPA**——LeCun 一直主张的那条路线——干脆丢掉重建，只在表征空间里预测，规划放到测试时用搜索完成。
这两条路线通常被当作对立的哲学来讨论。这个仓库记录的是：我把其中一条从零写了一遍，把另一条跑通，
然后花时间去试图推翻后者的核心主张。

结论的短版本，分两层。

**表层**：预测 loss 最低的那个 checkpoint，encoder 已经塌缩了——loss 0.004，
比正常模型好 60 倍，而 192 维 embedding 里有 69 维方差趋近于零。
这正是论文里那个正则项要防的失败模式，我直接把它测了出来。

**里层，而且是我没预料到的**：我做了三组受控实验，全部用 CEM 规划成功率当主指标。
第四组实验（不训练，只测已有 checkpoint）发现这个指标被一个**我从没控制、
甚至从没测量过的变量支配**——embedding 的绝对尺度（r = +0.89，t = 5.85）。
而我那三个干预（改上下文长度、换 loss、调正则化权重）**都会顺带改变尺度**。

所以前三组每次以为在测 A，实际测到的是"A 对尺度的副作用"。

![](04_analysis/figures/exp4_scale_vs_success.png)

顺带推翻了两件事：我原本以为"塌缩让表征不再携带可用信息"——
物理量线性探针显示信息几乎完好（R² 0.467 对健康的 0.497），
崩掉的是**信噪比**，不是信息。以及 SIGReg 在整个 1000 倍权重扫描里
**完全没有改变它声称要优化的那个量**（离高斯距离恒为 1206），
它实际维持的是尺度。

这个项目最终讲的不是"我复现了两篇论文"，是**一次完整的自我证伪**：
做了三组实验 → 发现它们测的是同一个混杂变量 → 换低方差指标做诊断 →
找到真正的中介变量 → 顺带发现论文命名的机制和实际生效的机制不是一回事。


---

## 目录

| | |
|---|---|
| [`01_dreamer_from_scratch/`](01_dreamer_from_scratch) | 从零手写的 DreamerV2，单文件。RSSM、reward model、actor-critic、imagination rollout。CartPole 上对比 Double-DQN，3 个 seed。 |
| [`02_lewm_reproduction/`](02_lewm_reproduction) | 跑通 [LeWorldModel](https://github.com/lucas-maes/le-wm) 并复现它发布的 checkpoint。这个目录里大部分笔记是关于安装的——时间实际花在了那儿。 |
| [`03_experiments/`](03_experiments) | 四组实验。前三组测错了东西，第四组查出了原因——**建议从第四组读起**。 |
| [`04_analysis/`](04_analysis) | `summary.csv`——每个 checkpoint、每项指标，一张表——以及画图的脚本。 |
| [`05_dreamerv3_scale/`](05_dreamerv3_scale) | 官方 DreamerV3 在 DMC walker-walk 上跑了一整夜，跑到任务天花板。 |
| [`notes/engineering_log.md`](notes/engineering_log.md) | 什么坏了，各花了多少时间。 |
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
全部写进了 [`02_lewm_reproduction/setup.md`](02_lewm_reproduction/setup.md)——
因为这恰恰是复现类文章通常略过、而实际上真正卡住人的部分。

## 第三部分 —— 四组实验

下面前三组都是：同一份数据、同一个 encoder、同一个优化器、同样 900 步梯度更新、
同一个 CEM 规划器、同样 50 条留出 episode，每组只动一个变量。

**它们的结论后来全被第四组推翻了**，保留在这里是因为推翻的过程本身才是这个项目的内容。

### 1. predictor 需要多少上下文？

`history_size` ∈ {1, 3, 5}——predictor 能看到几帧过去的 embedding。

![](04_analysis/figures/exp1_horizon.png)

预测误差随上下文单调下降。规划成功率没有跟着走：h=1 → h=3 这一步，
loss 改善了而成功率**掉了**（54% → 52%）。每组 50 条 episode，二项标准误约 7 个百分点，
所以诚实的读法是"h=5 更好，其余是噪声"，而不是一条规律。
而 loss 曲线暗示的东西和评估实际做的事之间这道错位，正是后面两组实验要追的线索。

### 2. 换防塌缩机制（做错一次，重做一次）

原计划是「固定预算只换架构」，做法是改模型配置里的 `_target_`。
**这个设计是错的**——JEPA 这一族方法共享架构，PLDM 和 LeWM 的区别在 loss 上：

| | LeWM | PLDM |
|---|---|---|
| 防塌缩 | SIGReg ×1 | VCReg ×4 + 时序对齐 + 逆动力学 |
| 可调权重 | **1** | **6** |

正对上论文摘要那句 "reduces tunable loss hyperparameters **from six to one**
compared to **the only existing end-to-end alternative**"——那个 alternative 就是 PLDM。
只换模型类而沿用 LeWM 的 loss，等于把 PLDM 抽空，两组跑的都是 LeWM。

重做时顺带修了一个真 bug：`train.py` 调 `spt.Manager` 没传 `seed=`，
权重初始化根本不受控——第一遍那 14 个点的差距就是这么来的。

重做结果（每组 3 个 seed）：

| | 成功率 | 均值 | 组内标准差 |
|---|---|---|---|
| LeWM | 36% / 60% / 70% | 55.3% | 17.5pp |
| PLDM | 32% / 36% / 62% | 43.3% | 16.3pp |

Welch t = 0.87，**远不显著**。方向支持 LeWM，但统计上分不开。

顺带发现 **`pred_loss` 跨模型不可比**：PLDM 的预测 loss 稳定低 13–16 倍，
但它的 embedding 尺度也小得多——按尺度归一化后两者持平（2.84 vs 2.66）。
这是塌缩陷阱的弱化版：不用真塌缩，只要把表征压紧，MSE 就无偿变好看。

细节见 [`03_experiments/exp2_pldm/`](03_experiments/exp2_pldm)。

### 3. 那个正则项，真的在做论文说的事吗？

LeWorldModel 的卖点是两项 loss 就够：next-embedding 预测，
加一个把 embedding 拉向各向同性高斯、使其无法塌缩的 SIGReg 项。默认权重 0.09。我把它扫到 0.001 和 1.0。

**loss 这一列回答不了这个问题**，这正是陷阱所在——encoder 塌缩会让预测变得*更容易*，
所以 `pred_loss` 反而会降。于是我没有去读 loss，而是写了
[`probe_collapse.py`](03_experiments/exp3_sigreg/probe_collapse.py)：
编码 256 帧真实的留出数据，测量每个维度的标准差，数出有多少维实际上已经死了。

![](04_analysis/figures/exp3_sigreg.png)

权重 0.001 时，**192 维 embedding 里有 69 维在真实数据上方差趋近于零**，
平均标准差 0.0012 对健康时的 0.325——差 270 倍。模型找到了作弊解。
它的 `pred_loss` 是 0.004，比这里训出来的任何东西低两个数量级，而且完全没有意义。

规划成功率掉到 34%，方向一致。但要说清楚：**塌缩本身是直接测量出来的，不是推断的**，
而 52% → 34% 这个具体幅度 p = 0.064，我的评估规模不足以钉死它。
诚实的表述是：关掉正则项导致表征塌缩（已证实），塌缩方向性地损害了规划（一致但欠功效）。

10 倍权重那一头不会塌缩，但 embedding 被压得太紧、失去区分度——38%。

论文选的默认值正好落在这个倒 U 的顶点。这不是空泛的"正则化有用"，
而是一个具体的、已发表的超参数，被拿它自己声称要防的失败模式、按那个失败模式本身的定义验证了一遍。

### 4. 成功率到底被什么决定？（诊断，也是唯一显著的结论）

不训练任何模型，只在已有的 12 个 checkpoint 上做低方差测量：
表征尺度、有效秩、离高斯的距离、以及"物理量线性探针 R²"（论文自己的 probing 口径）。

与规划成功率的相关（n=11，900 步的那些）：

| 指标 | r | 显著性 |
|---|---|---|
| 物理探针 R²（信息够不够） | +0.24 | 不显著 |
| 有效秩 | −0.22 | 不显著 |
| 离高斯距离 | −0.40 | 不显著 |
| **embedding 尺度** | **+0.89** | **t=5.85，显著** |

![](04_analysis/figures/exp4_confound.png)

左图：实验一和实验三的成功率变化，与它们各自的尺度变化**完全同形**。
右图：SIGReg 权重调 1000 倍，它声称要优化的"离高斯距离"**纹丝不动**（1206/1206/1206），
而真高斯是 0.5、4 维线性流形是 170——训出来的表征离各向同性高斯有 2400 倍远，
**只用了 192 维里约 4 维**。

还有个说明性的例外：官方完整训练的 checkpoint 尺度只有 0.032（比我所有健康模型小一个量级）、
有效秩 1.66（全场最低），却拿到 86%。说明"尺度决定成败"只在**欠训练**模型里成立——
训练充分后 predictor 噪声降下来，紧凑表征反而更好。
900 步的模型只能靠"把表征撑大"来盖过自己的预测噪声。

细节见 [`03_experiments/exp4_representation/`](03_experiments/exp4_representation)。

### 六个 checkpoint 放一起

![](04_analysis/figures/loss_vs_success.png)

如果验证 loss 是对的代理指标，这张图应该呈现一条向右下的趋势。它没有——
但要注意，中间那四个点彼此的差距都在噪声里（见实验二），
真正撑起这个结论的是最左边那个塌缩点：loss 最低、规划几乎最差，
而且它的异常有独立的表征测量佐证。

完整数字在 [`04_analysis/summary.csv`](04_analysis/summary.csv)，
显著性检验在 [`04_analysis/significance.py`](04_analysis/significance.py)。

---

## 这是什么，不是什么

**是**：两种世界模型的复现，加上一条完整的自我纠错链——三组受控实验、
发现它们共享一个未受控的混杂变量、用低方差指标做诊断、找到真正的中介变量，
并在过程中推翻了自己先前的两个机制解释和论文声称的一个机制。

**不是**可发表级别的结果。诚实清单：

- 前三组实验的成功率结论**全部作废**。它们的主指标被 embedding 尺度支配（r=0.89），
  而三个干预都会改变尺度——加 seed 也修不了，那只会得到一个混杂量的精确估计。
- 实验二第一版**变量选错了**（JEPA 家族共享架构，区别在 loss，我只换了架构类）。
- 实验三的塌缩检测成立，但我给的**机制解释是错的**（不是信息没了，是信噪比崩了）。
- 实验四虽然显著（t=5.85），但 n=11、纯观测、非干预。
  要确证"尺度导致规划成败"，得做干预实验——比如在评估时人为缩放 embedding，
  看成功率是否跟着变。这是显然的下一步，我没做。
- 全部实验都在 900 步的欠训练模型上，而官方 checkpoint 在同协议下是 86%。
  实验四本身也显示尺度那条规律**在训练充分的模型上就不成立了**。
- 没有 Dreamer 与 LeWorldModel 的正面对比（目标函数和评估指标不共享语义）。

## 复现

每一部分都有自己的运行脚本。第一部分是自包含的（`pip install gymnasium torch`）；
第二、三部分需要一份 [le-wm](https://github.com/lucas-maes/le-wm) 的 checkout——
**先读 [`02_lewm_reproduction/setup.md`](02_lewm_reproduction/setup.md)**，
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
