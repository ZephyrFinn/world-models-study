# 从零手写 DreamerV2

[English](README_en.md) · **中文**

`dreamer.py` 是完整实现，单文件，不依赖框架——所以没有任何地方可以藏一个我没搞懂的组件。

- **Encoder** —— 向量观测用 MLP（64×64 像素的 CNN 分支写了，但这些实验没用到）
- **RSSM** —— GRU 确定性状态 + 32×16 categorical 随机状态，straight-through 梯度。
  `obs_step`（后验，条件于真实观测）和 `img_step`（先验，只有动作）分开；
  `observe` 沿真实序列跑后验，`imagine` 在完全看不到观测的情况下把先验向前滚动。
- **Reward head** 和 **continue head**，都建在潜在状态上
- **Actor-critic** —— 基于 λ-return，配一个缓慢更新的 target critic
- **Imagination rollout** —— actor 完全在从回放的后验状态分叉出去的先验轨迹上训练

`baseline_dqn.py` 是用来对照的 Double-DQN。

## 结果

CartPole-v1，2 万环境步，各 3 个 seed。

| | 全程平均评估回报 | 收敛段（最后 3 次评估） |
|---|---|---|
| DreamerV2 | 240.1 | 248.6 |
| Double-DQN | 113.8 | 162.2 |

![](results/comparison_multiseed.png)

Dreamer 从约 3k 步开始拉开并保持，方差也更小。DQN 更抖，还出现了典型的灾难性遗忘——
有个 seed 中途冲到 500，又跌回 100 左右。

样本效率的差距是教科书结果。值得看的是诊断图：

![](results/diagnostics.png)

- **KL** 在 free bits = 1.0 的约束下稳定在 0.8 附近——约束在起作用，
  说明后验没有塌缩到先验上，也没有爆炸。
- **想象中的 λ-return** 训练过程中从 0 涨到 70+，与真实评估回报同步。
  这是在检查模型想象出来的轨迹是否对应现实：actor 只见过想象状态，
  如果这条曲线涨而真实回报不动，就说明世界模型在舒服地做幻觉、策略在对着幻想做优化。

## 运行

```bash
./run_experiments.sh     # 3 seed ×（Dreamer + DQN），单卡约 20 分钟
python plot_multiseed.py
```

每次运行的原始指标在 `results/runs/*/metrics.csv`；训练权重（`agent.pt`）没有入库。
