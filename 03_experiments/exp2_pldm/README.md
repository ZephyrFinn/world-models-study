# 实验二 —— 同预算下的 LeWM vs PLDM

[English](README_en.md) · **中文**

PLDM 随 `stable_worldmodel` 一起发布，是 le-wm 对比的 baseline 之一。
它的类接受和 LeWM 的 JEPA 相同的构造参数，也暴露相同的 `encode()` / `predict()`，
因此能塞进同一个训练循环和同一个 CEM 规划器——`pldm.yaml` 里只有 `_target_` 路径与 LeWM 的模型配置不同。

对照 LeWM 的 h=3：同数据、同 900 步、同评估。

| | pred_loss | CEM 成功率 | embedding 平均标准差 |
|---|---|---|---|
| LeWM h=3 | 0.266 | 52% | 0.325 |
| PLDM | 0.298 | **66%** | 0.380 |

PLDM 预测误差更差，规划器更强。两者 embedding 都健康（没有死维度），所以这不是塌缩造成的假象。

两个 caveat，都是真的：

- **单 seed。** 66 对 52 是 14 个百分点，而标准误约 7——是"值得注意"，不是"已确立"。
- **这是 PLDM 的架构配 LeWM 的训练配方**（next-embedding loss + SIGReg），
  不是 PLDM 自己的训练目标。正是这一点隔离出了架构变量，
  但也意味着这个数字**不是对 PLDM 论文的复现**，不应被这样引用。

它的价值在于：它是一个反例，说明 loss 这一列不能拿来排序。
