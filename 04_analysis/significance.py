"""这套设置到底能分辨多大的差异？

结论先写在前面：**分辨不了 10 到 20 个百分点级别的差异**，而我三组实验
设计出来的预期效应量全都在这个区间。

更糟的是，加 seed 也修不了它们 —— 见第四节：三组实验动的三个旋钮
都改变了同一个中介变量（embedding 尺度，r=0.89），成功率测的是那个变量。
跑 45 个 seed 只会得到一个混杂量的精确估计。

依据来自实验二重做：同一配置、同数据、同预算，只换随机种子，跑 3 次。
成功率的组内标准差约 17 个百分点——比我原本以为"值得讨论"的任何差距都大。

用法：python significance.py
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

N_EPISODES = 50
HERE = Path(__file__).parent
SUMMARY = HERE / "summary.csv"
PROBE = HERE / "probe_physics.json"
# summary.csv 里的 emb_mean_std 来自 probe_collapse.py（256 帧），
# 实验四的四个指标全部来自 probe_physics.py（3000 帧留出数据）。
# 相关性一律用后者，四个指标才在同一批样本上可比。
PROBE_ALIAS = {"lewm_hist3": "lewm_dim192"}

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

    correlations()
    robustness()


def _rows_and_probe():
    probe = {r["checkpoint"].split("/")[0]: r
             for r in json.loads(PROBE.read_text()) if "error" not in r}
    rows = [r for r in csv.DictReader(open(SUMMARY)) if r["train_steps"] == "900"]
    return rows, probe


def pearson(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
    return r, t, n


def correlations():
    """实验四：成功率到底跟哪个表征指标相关？

    只取 900 步的 checkpoint —— 官方 checkpoint 训练充分，
    它恰恰是这条规律的反例（尺度 0.032 却拿 86%），混进来会掩盖问题。
    """
    rows, probe = _rows_and_probe()
    y = [float(r["cem_success_pct"]) for r in rows]

    def col(name):
        return [float(probe[PROBE_ALIAS.get(r["checkpoint"], r["checkpoint"])][name])
                for r in rows]

    print("\n" + "=" * 74)
    print("四、那个成功率指标本身被什么决定？（实验四）")
    print("=" * 74)
    print(f"\n  n = {len(rows)} 个 900 步 checkpoint，t 检验自由度 {len(rows) - 2}\n")
    print(f"  {'指标':<28}{'r':>8}{'t':>8}{'p':>9}   结论")
    print("  " + "-" * 70)

    metrics = [
        ("物理探针 R²（信息够不够）", "probe_r2"),
        ("有效秩", "eff_rank"),
        ("离高斯距离", "gauss_stat"),
        ("embedding 尺度", "emb_mean_std"),
    ]
    for label, name in metrics:
        r, t, n = pearson(col(name), y)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        verdict = "显著" if p < 0.05 else "不显著"
        print(f"  {label:<28}{r:>+8.2f}{t:>8.2f}{p:>9.4f}   {verdict}")

    print("""
  只有尺度显著。也就是说：表征里有多少物理信息几乎不决定它能不能规划，
  表征有多大才决定。而实验一、二、三动的三个旋钮全都会改变尺度 ——
  它们测的是同一个混杂变量。

  注意这是观测不是干预（n 很小，纯相关）。要确证因果，
  得在评估时人为缩放 embedding 看成功率跟不跟着走。那一步我没做。""".rstrip())


def robustness():
    """n=11 的相关性必须做留一法，否则一个离群点就能撑起一个"发现"。

    同时检验机制假说：如果"信号被预测噪声淹没"成立，那么真正该起作用的
    是信噪比 pred_loss / scale²，而不是尺度本身 —— 因为规划器对尺度不变
    （代价是 embedding 空间的 MSE，选精英用 topk 排序，整体缩放不改变排序）。
    """
    rows, probe = _rows_and_probe()
    name = [r["checkpoint"] for r in rows]
    y = [float(r["cem_success_pct"]) for r in rows]
    scale = [float(probe[PROBE_ALIAS.get(r["checkpoint"], r["checkpoint"])]["emb_mean_std"])
             for r in rows]
    snr = [math.log10(float(r["norm_pred_loss"])) for r in rows]

    def pval(t, n):
        return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))

    print("\n" + "=" * 74)
    print("五、实验四的发现有多硬？（留一法 + 机制假说）")
    print("=" * 74)

    for label, xs, sign in [("embedding 尺度", scale, "+"),
                            ("信噪比 log10(pred_loss/scale²)", snr, "−")]:
        r0, t0, n0 = pearson(xs, y)
        rs = []
        weak = []
        for i in range(len(xs)):
            sub_x = [v for j, v in enumerate(xs) if j != i]
            sub_y = [v for j, v in enumerate(y) if j != i]
            r, t, n = pearson(sub_x, sub_y)
            rs.append(r)
            if pval(t, n) >= 0.05:
                weak.append(name[i])
        lo, hi = min(rs), max(rs)
        print(f"\n  {label}")
        print(f"    全部 11 点         r = {r0:+.3f}   p = {pval(t0, n0):.4f}")
        print(f"    留一法 r 的范围     [{lo:+.3f}, {hi:+.3f}]")
        if weak:
            print(f"    删掉这些点后不再显著：{', '.join(weak)}   ← 脆弱")
        else:
            print(f"    任意删掉一个点都仍然显著   ← 稳健")

    print("""
  读法：

    尺度那条稳得住 —— 留一法 r 始终在 0.87 以上，不是被某个离群点撑起来的。

    机制那条撑不住 —— 信噪比方向对（负相关，误差相对信号越大、规划越差），
    但删掉一个点就掉到不显著。它是提示，不是证据。

  所以准确的说法是：**尺度是一个稳健的标记（marker），不是已证实的原因
  （cause）**。而且规划器对尺度严格不变（代价函数是 MSE，选精英是纯排序），
  所以尺度不可能通过代价几何直接起作用 —— 它一定是在代理别的什么东西，
  而那个东西是什么，这个项目还没回答。

  这也意味着"评估时把 embedding 乘个系数看成功率变不变"是个**无效实验**：
  按构造必然得零。见 03_experiments/exp4_representation/README.md 里
  修正后的两个下一步。""".rstrip())


if __name__ == "__main__":
    main()
