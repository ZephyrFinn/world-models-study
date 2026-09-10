# 实验二 —— 防塌缩机制对比（做错一次，重做一次）

[English](README_en.md) · **中文**

> 这个实验做了两遍。第一遍设计是错的，第二遍才测到该测的东西，
> 而第二遍的主要产出是一个让我不太舒服的数字：**这套设置的种子方差有 ±17 个百分点。**

## 第一遍：错在哪

原计划是「固定预算只换架构」，做法是把模型配置里的 `_target_` 从
`jepa.JEPA` 换成 `stable_worldmodel.wm.pldm.pldm.PLDM`。

结果 PLDM 预测 loss 更差却规划更好（0.298/66% 对 0.266/52%），看着像个漂亮的反例。

**它不成立**，因为 JEPA 这一族方法**共享架构，区别在 loss 上**：

```
JEPA（编码 → 表征空间预测 → 无 decoder）
 ├── PLDM   (arXiv 2502.14819)  VCReg×4 + 时序对齐 + 逆动力学 = 6 项
 ├── DINO-WM                    冻结的预训练编码器
 └── LeWM   (arXiv 2603.19312)  SIGReg = 1 项
```

LeWM 论文摘要那句 "reduces tunable loss hyperparameters **from six to one**
compared to **the only existing end-to-end alternative**"——那个 alternative 就是 PLDM，
六项正对得上。

所以只换模型类、沿用 LeWM 的 loss，等于把 PLDM 抽空了：**两组跑的都是 LeWM**。
两组训练日志里都有 `sigreg_loss` 这一项，就是证据——真跑 PLDM 根本不该有它。

还有第二个 bug：`train.py` 调 `spt.Manager` 时没传 `seed=`，权重初始化不受 `cfg.seed` 控制
（库自己会警告 `User didn't specify seed`）。那 14 个百分点的差距，来源就是两次不同的随机初始化。

## 第二遍：换真正的变量

[`exp2_train.py`](exp2_train.py) 保持架构、数据、预算、规划器全部相同，只换防塌缩机制：

| | LeWM | PLDM |
|---|---|---|
| 核心目标 | `pred_loss` | `pred_loss` |
| 防塌缩 | SIGReg ×1 | VCReg ×4（std / std_t / cov / cov_t） |
| 额外正则 | — | 时序对齐 `temp_align` |
| 额外结构 | — | 逆动力学头（我加的两层 MLP） |
| **可调权重** | **1** | **6** |

顺带修了 seed bug：`spt.Manager(..., seed=cfg.seed)`。修完验证过——同 seed 两次的
sanity loss 差 0.0006，不同 seed 差 0.005，差一个数量级，说明种子确实在控制初始化了。
（残余差异来自 bf16 下 GPU 归约的非确定性，要逐位复现还得开
`torch.use_deterministic_algorithms`，代价是速度。）

每组 3 个 seed。

## 结果

| | 成功率（seed 0/1/2） | 均值 | 组内标准差 |
|---|---|---|---|
| LeWM | 36% / 60% / 70% | 55.3% | **17.5pp** |
| PLDM | 32% / 36% / 62% | 43.3% | **16.3pp** |

Welch t 检验：差 +12pp，t = 0.87，95% 区间 [-27, +51]。**远不显著。**

## 三个发现

### 一、种子方差 ±17pp，实验设计欠功效约一个数量级

这是主要产出。同配置、同数据、只换种子，成功率能从 36% 跑到 70%。

按这个方差算，**要以 80% 功效检出 10 个百分点的真实差异，每组需要约 45 个 seed**。
我原本每组只有 1 个。跑一下 [`../../04_analysis/significance.py`](../../04_analysis/significance.py) 看全部数字。

后果是：这个仓库里所有基于成功率的比较——包括实验一的 horizon 扫描、
实验三的 SIGReg 成功率差距——**全部落在噪声里**。

### 二、`pred_loss` 跨模型不可比

PLDM 的预测 loss 稳定低 13–16 倍（0.016–0.021 对 0.25–0.34），看起来碾压。
但它的 embedding 尺度也小得多。`pred_loss` 是 embedding 空间里的 MSE，
量纲随表征尺度的平方缩放，所以要除掉尺度才能比：

| | pred_loss | emb 标准差 | 归一化后 |
|---|---|---|---|
| LeWM | 0.251–0.345 | 0.26–0.41 | 均值 **2.84** |
| PLDM | 0.016–0.021 | 0.07–0.31 | 均值 **2.66** |

**基本持平。** 那个「低 16 倍」完全是尺度造成的假象。

这是实验三那个塌缩陷阱的**弱化版**：不需要真的塌缩，只要正则项把表征压得更紧，
MSE 就无偿变好看。[`../exp3_sigreg/probe_collapse.py`](../exp3_sigreg/probe_collapse.py)
现在会输出 `scale_sq`，就是为了随手做这个归一化。

### 三、PLDM 的训练稳定性看起来更差

`pldm_s2` 的 embedding 平均标准差是 0.306，另两个 seed 是 0.071 和 0.073——
**同配置内差 4 倍**。而它恰好也是 PLDM 里成功率最高的那个（62%）。

暗示六项 loss 会让不同初始化收敛到尺度差异很大的表征上。n=3 不足以下结论，记一笔。

## 对论文声明的回答

LeWM 声称「更简单（1 个超参 vs 6 个）但不输性能」。在 900 步这个预算下：

- **性能**：55.3% 对 43.3%，方向支持 LeWM，但**统计上无法区分**（t=0.87）
- **简洁性**：这一条可以确认。为了跑 PLDM 我得额外写一个逆动力学头，
  还得给六项 loss 定权重——而**论文里的具体取值我没有，只能全取 1.0**。
  这个不确定性本身就是「六个超参数」的成本，不是我的实现瑕疵。

## 复现

```bash
python exp2_train.py data=tworoom +loss_type=lewm seed=0 \
  output_model_name=exp2_lewm_s0 subdir=exp2_lewm_s0 \
  trainer.max_epochs=3 +trainer.limit_train_batches=300 +trainer.limit_val_batches=20 \
  wandb.enabled=false
# loss_type 换成 pldm，seed 换成 0/1/2，共 6 组
```

`results_v2/` 是第二遍的全部日志和评估输出；`results/` 是第一遍的，保留作为对照。

## 教训

1. **想比架构，先 `diff` 实现。** 「构造函数签名一样、能互相替换」不是架构不同的证据。
2. **先搞清楚这一族方法的分类维度是什么。** JEPA 家族的变量是防塌缩机制，不是网络结构——
   我一开始连自己要比什么都没搞对。
3. **评估规模要先于实验设计确定。** 应该先花 20 分钟跑 3 个种子量出方差，
   再决定实验值不值得做。我是反过来的：先做完三组实验，最后才发现它们全都测不出东西。
