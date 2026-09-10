# 实验

[English](README.md) · **中文**

关于 LeWorldModel 的三个问题，每个只动一个变量。

所有实验共用的协议：tworoom 数据集、ViT-tiny encoder、AdamW（5e-5）、batch 128、
**300 batch × 3 epoch = 900 步梯度更新**，然后用 CEM-MPC 在**同样 50 条留出 episode**
上规划。除了被点名的那个变量，其余一律相同。

900 步是刻意的短训练。发布的 checkpoint 在同一协议下是 86%，
这里的一切落在 34%–66% 之间——所以这些都是欠训练的模型互相比较，不是在跟论文比。

| | 问题 | 变量 | 结论 |
|---|---|---|---|
| [`exp1_horizon/`](exp1_horizon) | predictor 需要多少上下文？ | `history_size` ∈ {1,3,5} | h=5 最好；h=1 vs h=3 是噪声 |
| [`exp2_pldm/`](exp2_pldm) | 固定预算下架构有影响吗？ | JEPA vs PLDM | PLDM 预测更差，规划更好 |
| [`exp3_sigreg/`](exp3_sigreg) | 正则项真能防塌缩吗？ | `loss.sigreg.weight` ∈ {0.001, 0.09, 1.0} | 能——0.001 时复现出了塌缩 |

结果汇总到 [`../04_analysis/summary.csv`](../04_analysis/summary.csv)。

## 关于噪声

50 条 episode 是一个二项样本。p≈0.5 时标准误约 7 个百分点，
所以这里任何单个数字的 95% 区间大致是 ±14 个百分点。

**扛得住这个尺度的差距**：sigreg 0.001（34%）对默认 0.09（52%）；
发布的 checkpoint（86%）对这里的一切。

**扛不住的**：h=1（54%）对 h=3（52%）；PLDM（66%）对 h=5（64%）。

SIGReg 那个结果是唯一值得信的，不只是因为差距更大——
它是唯一**同时有一个独立的机制测量**（embedding 方差、死维度计数）指向同一方向的。
