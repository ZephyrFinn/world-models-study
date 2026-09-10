# 分析

[English](README_en.md) · **中文**

`summary.csv` —— 每个 checkpoint 一行，收集了全部指标：
训练 loss 的各项、CEM 规划成功率，以及 `probe_collapse.py` 测出的 embedding 健康度诊断。

`make_figures.py` 读它，写出 `figures/`：

| 图 | 内容 |
|---|---|
| `exp1_horizon.png` | 不同 `history_size` 下预测误差与规划成功率的对照 |
| `exp3_sigreg.png` | 倒 U 曲线，以及它下面的塌缩 |
| `loss_vs_success.png` | 六个 checkpoint 放在同一张散点上 |

`significance.py` 对每一组成功率对比做双比例 z 检验。写它是因为我一开始
把两组结果的差距当成了结论，后来才发现六个对比里只有一个是显著的。

`dreamerv3_curve.py` 单独负责 DreamerV3 那条学习曲线，
直接读训练任务的 `metrics.jsonl`，所以随时重跑都能刷新。

```bash
python make_figures.py
python dreamerv3_curve.py
python significance.py
```

那张散点图是整个仓库的总结：横轴是跨了近三个数量级的预测 loss（对数轴），
纵轴是规划成功率，两者没有趋势。

**但要按正确的强度读它**：中间四个点彼此的差距都在噪声里
（经验噪声底约 15 个百分点，来自实验二那次意外测量）。
真正撑起结论的是最左边那个点——loss 最低、规划几乎最差，
而且它的异常有独立的表征测量（270 倍方差差距、69 个死维度）佐证，
不依赖成功率这个噪声很大的指标。
