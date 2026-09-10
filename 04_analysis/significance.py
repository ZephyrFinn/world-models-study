"""这套设置到底能分辨多大的差异？

结论先写在前面：**分辨不了 10 到 20 个百分点级别的差异**，而我三组实验
设计出来的预期效应量全都在这个区间。

依据来自实验二重做：同一配置、同数据、同预算，只换随机种子，跑 3 次。
成功率的组内标准差约 17 个百分点——比我原本以为"值得讨论"的任何差距都大。

用法：python significance.py
"""
import math
import statistics as st

N_EPISODES = 50

# 实验二重做：同配置 3 个 seed（seed 现在真的生效了，见 exp2_train.py）
SEEDED = {
    "LeWM  (SIGReg, 1 项)":           [36.0, 60.0, 70.0],
    "PLDM  (VCReg+align+IDM, 6 项)":  [32.0, 36.0, 62.0],
}

# 单 seed 的旧对比，保留下来是为了说明它们全部落在上面那个方差里
SINGLE_SEED = [
    ("h=5 64% vs h=3 52%",              64.0, 52.0),
    ("h=1 54% vs h=3 52%",              54.0, 52.0),
    ("SIGReg 默认 52% vs 塌缩 34%",       52.0, 34.0),
    ("SIGReg 默认 52% vs 过强 38%",       52.0, 38.0),
    ("官方 checkpoint 86% vs h=3 52%",   86.0, 52.0),
]


def binom_z(p1, p2, n=N_EPISODES):
    """只考虑评估采样噪声，不含 seed 方差 —— 这是乐观的下界。"""
    p1, p2 = p1 / 100, p2 / 100
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = (p1 - p2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (p1 - p2) * 100, se * 100, z, p


def main():
    print("=" * 74)
    print("一、种子方差（实验二重做，每组 n=3）")
    print("=" * 74)
    for name, vals in SEEDED.items():
        m, sd = st.mean(vals), st.stdev(vals)
        print(f"  {name:<32} {vals}  均值 {m:.1f}%  标准差 {sd:.1f}pp")

    a, b = list(SEEDED.values())
    ma, mb, va, vb, n = st.mean(a), st.mean(b), st.variance(a), st.variance(b), 3
    se = math.sqrt(va / n + vb / n)
    t = (ma - mb) / se
    print(f"\n  两组差值 {ma - mb:+.1f}pp,  Welch t = {t:.2f},  标准误 {se:.1f}pp")
    print(f"  95% 区间约 [{ma - mb - 2.8 * se:+.0f}, {ma - mb + 2.8 * se:+.0f}]pp"
          f"  →  {'显著' if abs(t) > 4.3 else '远不显著'}")

    pooled_sd = math.sqrt((va + vb) / 2)
    need = math.ceil(2 * (1.96 + 0.84) ** 2 * pooled_sd ** 2 / 10 ** 2)
    print(f"\n  合并组内标准差 {pooled_sd:.1f}pp")
    print(f"  → 要以 80% 功效检出 10pp 的真实差异，每组需要约 {need} 个 seed。")
    print(f"    我原本每组只有 1 个。整个实验设计欠功效约一个数量级。")

    print("\n" + "=" * 74)
    print("二、单 seed 旧对比（只算评估采样噪声，已是乐观下界）")
    print("=" * 74)
    print(f"  {'对比':<34}{'差值':>7}{'z':>7}{'p':>8}   结论")
    print("  " + "-" * 70)
    for label, p1, p2 in SINGLE_SEED:
        d, sem, z, p = binom_z(p1, p2)
        verdict = "显著" if p < 0.05 else "不显著"
        print(f"  {label:<34}{d:>+6.0f}pp{z:>7.2f}{p:>8.3f}   {verdict}")

    print(f"""
  注意：上表连 seed 方差都没算进去。把 ±{pooled_sd:.0f}pp 的种子方差叠上之后，
  这些对比全部落进噪声 —— 包括那个看起来显著的官方 checkpoint 对比
  （86% 高于我三次运行的每一次，方向可信，但 n=3 撑不起统计声明）。
""".rstrip())

    print("\n" + "=" * 74)
    print("三、那还剩什么站得住")
    print("=" * 74)
    print("""
  只剩不依赖成功率的直接测量：

    塌缩（SIGReg 权重 0.001）
      embedding 平均标准差  0.0012  对  0.325      —— 差 270 倍
      死维度                69 / 192  对  0 / 192

  这两个数字是拿真实数据编码后量出来的，不是从 loss 推断的，
  也不经过 50 条 episode 的采样噪声。它们是这个项目里唯一
  经得起当前统计强度检验的结论。""".rstrip())


if __name__ == "__main__":
    main()
