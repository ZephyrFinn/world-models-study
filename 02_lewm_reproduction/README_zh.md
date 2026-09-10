# 复现 LeWorldModel

[English](README.md) · **中文**

上游：[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) ——
*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*
（Maes, Le Lidec, Scieur, LeCun, Balestriero；arXiv:2603.19312）。

一个 JEPA 世界模型：约 1500 万参数，ViT-tiny encoder，在 embedding 上做自回归的
transformer predictor，全程没有 decoder。它一个像素都不重建。
规划在测试时用 CEM-MPC 完成——采样一批动作序列，在 embedding 空间里向前滚动，
保留那些预测终点最接近目标图像 embedding 的。

## 结果

发布的 tworoom checkpoint，50 条留出 episode，CEM-MPC：

**86% 成功率** —— [`results/pretrained_tworoom.txt`](results/pretrained_tworoom.txt)

```bash
./eval_pretrained.sh
```

这是 [`../03_experiments/`](../03_experiments) 里所有数字的参照基准——
那里同样协议下的 900 步 checkpoint 落在 34% 到 66% 之间。

## 先读 setup_zh.md

[`setup_zh.md`](setup_zh.md) —— 截至 2026 年 9 月，默认安装方式跑不通。
README 与已发布的 `stable-worldmodel` 0.1.1 之间有四处断裂，
外加 checkpoint 的存放格式也变了。那份文档是这个目录里价值最高的东西。

`convert_ckpt.py` 是 README 那段转换代码的 Hydra `instantiate` 版本。
当前发布格式（要的是纯 state dict）用不上它，留着是为了以后万一又改回 object checkpoint。
