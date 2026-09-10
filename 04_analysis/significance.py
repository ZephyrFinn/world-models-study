"""每个成功率对比的双比例 z 检验。

写这个脚本是因为我一开始把两组结果的差距当成了结论，后来才发现它们大多落在
噪声里。50 条 episode、成功率 0.5 附近，单个数字的标准误就有约 7 个百分点，
两个数字相减的标准误接近 10——这个尺度值得算出来摆在明面上，而不是凭感觉说
"差距挺大"。

用法：python significance.py
"""
import math

N = 50  # 每个 arm 的评估 episode 数

# (标签, p1, p2)：p1 - p2 是被讨论的差距
COMPARISONS = [
    ("PLDM 66% vs LeWM h=3 52%",      0.66, 0.52),
    ("h=5 64% vs h=3 52%",            0.64, 0.52),
    ("h=1 54% vs h=3 52%",            0.54, 0.52),
    ("SIGReg 默认 52% vs 塌缩 34%",     0.52, 0.34),
    ("SIGReg 默认 52% vs 过强 38%",     0.52, 0.38),
    ("官方 checkpoint 86% vs h=3 52%", 0.86, 0.52),
]


def z_test(p1, p2, n=N):
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    diff = p1 - p2
    z = diff / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return diff, se, z, p, (diff - 1.96 * se, diff + 1.96 * se)


def main():
    print(f"双比例 z 检验，每组 n = {N}\n")
    print(f"{'对比':<32} {'差值':>7} {'z':>6} {'p':>7}  {'95% CI':>16}  显著性")
    print("-" * 82)
    for label, p1, p2 in COMPARISONS:
        diff, se, z, p, (lo, hi) = z_test(p1, p2)
        sig = "显著" if lo > 0 or hi < 0 else "不显著"
        print(f"{label:<32} {100*diff:>+6.0f}pp {z:>6.2f} {p:>7.3f}  "
              f"[{100*lo:>+5.0f},{100*hi:>+5.0f}]pp  {sig}")

    print("\n" + "-" * 82)
    print("""
只有最后一行是显著的。这不是说其余实验没有信息量，而是说 50 条 episode 的
评估无法把 10-15 个百分点级别的差距和噪声分开。

实际的噪声底可以从实验二读出来：那两个 arm 事后被发现是同一个架构
（stable_worldmodel 里 pldm 和 lewm 的 module.py 逐字节相同），
只是随机初始化不同——它们相差 14 个百分点。所以在这套设置下，
把任何 15 个百分点以内的差距当结论都是不安全的。

唯一不依赖这个尺度的证据是实验三的表征测量：embedding 平均标准差
0.0012 对 0.325（差 270 倍）、死维度 69 对 0。那是直接测量，不是统计推断。
""".strip())


if __name__ == "__main__":
    main()
