"""Figures for the write-up. Reads summary.csv, writes figures/*.png.

Three plots:
  1. horizon sweep      -- pred_loss and success side by side
  2. sigreg sweep       -- the inverted U, with the collapse diagnostic under it
  3. loss vs success    -- every checkpoint on one scatter; the punchline

Usage: python make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, GREY = "#c8402e", "#8a857a"
INK_ = "#17150f"

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


def load():
    with open(HERE / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        for k, v in list(r.items()):
            if k in ("checkpoint", "experiment", "arch", "anti_collapse",
                     "seed", "train_steps"):
                continue
            r[k] = float(v) if v else None
        out[r["checkpoint"]] = r
    return out


def fig_horizon(d):
    hs = [1, 3, 5]
    keys = [f"lewm_hist{h}" for h in hs]
    loss = [d[k]["pred_loss"] for k in keys]
    succ = [d[k]["cem_success_pct"] for k in keys]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    a.plot(hs, loss, "o-", color=BLUE, lw=2, ms=7)
    a.set_xlabel("history_size"); a.set_ylabel("validation pred_loss")
    a.set_title("Prediction error falls with context", fontsize=9.5, loc="left")
    a.set_xticks(hs)
    span = max(loss) - min(loss)
    a.set_ylim(min(loss) - span * 0.25, max(loss) + span * 0.30)
    for x, y in zip(hs, loss):
        a.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                   xytext=(0, 9), ha="center", fontsize=8, color=GREY)

    bars = b.bar([str(h) for h in hs], succ, color=[BLUE, ORANGE, GREEN], width=0.55)
    b.set_xlabel("history_size"); b.set_ylabel("CEM success (%)")
    b.set_ylim(0, 100)
    b.set_title("Planning success does not follow it cleanly", fontsize=9.5, loc="left")
    for bar, y in zip(bars, succ):
        b.annotate(f"{y:.0f}%", (bar.get_x() + bar.get_width() / 2, y),
                   textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "exp1_horizon.png", bbox_inches="tight")
    plt.close(fig)


def fig_sigreg(d):
    order = ["lewm_sigreg_low", "lewm_hist3", "lewm_sigreg_high"]
    labels = ["0.001\n(off)", "0.09\n(paper default)", "1.0\n(10x)"]
    succ = [d[k]["cem_success_pct"] for k in order]
    dead = [d[k]["dead_dims"] for k in order]
    spread = [d[k]["emb_mean_std"] for k in order]
    colors = [RED, GREEN, YELLOW]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    bars = a.bar(labels, succ, color=colors, width=0.55)
    a.set_ylabel("CEM success (%)"); a.set_ylim(0, 100)
    a.set_title("Both extremes lose to the published default", fontsize=9.5, loc="left")
    for bar, y in zip(bars, succ):
        a.annotate(f"{y:.0f}%", (bar.get_x() + bar.get_width() / 2, y),
                   textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)

    b.bar(labels, spread, color=colors, width=0.55)
    b.set_yscale("log")
    b.set_ylabel("mean embedding std (log)")
    b.set_title("...because at 0.001 the embedding collapses", fontsize=9.5, loc="left")
    for i, (s, dd) in enumerate(zip(spread, dead)):
        note = f"{s:.4f}\n{int(dd)}/192 dead" if dd else f"{s:.3f}\nno dead dims"
        b.annotate(note, (i, s), textcoords="offset points", xytext=(0, 6),
                   ha="center", fontsize=8, color=RED if dd else GREY)
    b.set_ylim(top=max(spread) * 6)

    fig.tight_layout()
    fig.savefig(OUT / "exp3_sigreg.png", bbox_inches="tight")
    plt.close(fig)


def fig_scatter(d):
    pts = [
        ("SIGReg x0.01", "lewm_sigreg_low", RED),
        ("h=1", "lewm_hist1", BLUE),
        ("h=3", "lewm_hist3", ORANGE),
        ("h=5", "lewm_hist5", GREEN),
        ("PLDM (v2, s0)", "exp2_pldm_s0", YELLOW),
        ("SIGReg x11", "lewm_sigreg_high", "#b3760a"),
    ]
    # hand-placed so nothing collides: (dx, dy, ha)
    offsets = {
        "SIGReg x0.01": (12, -16, "left"),
        "h=1": (13, 4, "left"),
        "h=3": (0, -17, "center"),
        "h=5": (-13, -3, "right"),
        "PLDM (v2, s0)": (6, 9, "left"),
        "SIGReg x11": (0, -17, "center"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, key, color in pts:
        x, y = d[key]["pred_loss"], d[key]["cem_success_pct"]
        ax.scatter(x, y, s=95, color=color, zorder=3,
                   edgecolor="white", linewidth=1.2)
        dx, dy, ha = offsets[label]
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel("validation pred_loss (log scale)  -- lower is 'better'")
    ax.set_ylabel("CEM planning success (%)")
    ax.set_ylim(20, 80)
    ax.set_title("Six checkpoints, no trend line",
                 fontsize=10, loc="left")
    ax.annotate("lowest loss of the night,\nnear-worst planner",
                xy=(0.0043, 35.5), xytext=(0.011, 48),
                fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))

    fig.tight_layout()
    fig.savefig(OUT / "loss_vs_success.png", bbox_inches="tight")
    plt.close(fig)


def fig_seed_variance(_d=None):
    """实验二重做：种子方差有多大 —— 这张图是整个仓库的尺子。"""
    lewm = [36.0, 60.0, 70.0]
    pldm = [32.0, 36.0, 62.0]
    import statistics as st

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                               gridspec_kw={"width_ratios": [1, 1.25]})

    # 左：每个 seed 的点 + 均值线
    for i, (vals, color, name) in enumerate([(lewm, BLUE, "LeWM\n(SIGReg, 1 term)"),
                                             (pldm, ORANGE, "PLDM\n(6 terms)")]):
        xs = [i + (j - 1) * 0.13 for j in range(3)]
        a.scatter(xs, vals, s=70, color=color, zorder=3, edgecolor="white", linewidth=1.2)
        a.hlines(st.mean(vals), i - 0.26, i + 0.26, color=color, lw=2.5, zorder=2)
        a.annotate(f"mean {st.mean(vals):.0f}%\nSD {st.stdev(vals):.0f}pp",
                   (i, st.mean(vals)), textcoords="offset points", xytext=(30, -6),
                   fontsize=8, color=GREY)
    a.set_xticks([0, 1]); a.set_xticklabels(["LeWM\n(1 term)", "PLDM\n(6 terms)"], fontsize=8.5)
    a.set_ylabel("CEM success (%)"); a.set_ylim(20, 85)
    a.set_title("3 seeds each, everything else fixed", fontsize=9.5, loc="left")

    # 右：噪声带 vs 本仓库所有被讨论过的差距
    pooled = (st.stdev(lewm) + st.stdev(pldm)) / 2
    gaps = [("h=1 vs h=3", 2), ("h=5 vs h=3", 12), ("LeWM vs PLDM", 12),
            ("SIGReg on/off", 18), ("released vs mine", 34)]
    ys = range(len(gaps))
    b.barh(list(ys), [g[1] for g in gaps], color=GREY, height=0.55, zorder=2)
    b.axvspan(0, pooled, color=RED, alpha=0.13, zorder=1)
    b.axvline(pooled, color=RED, lw=1.5, ls="--", zorder=3)
    b.annotate(f"seed noise ±{pooled:.0f}pp", (pooled, -0.75),
               textcoords="offset points", xytext=(4, 0), fontsize=8,
               color=RED, ha="left", va="center")
    b.set_yticks(list(ys)); b.set_yticklabels([g[0] for g in gaps], fontsize=8.5)
    b.set_xlabel("gap being claimed (pp)"); b.set_xlim(0, 40)
    b.set_ylim(-1.2, len(gaps) - 0.4)
    b.set_title("Only the last one clears the noise", fontsize=9.5, loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "exp2_seed_variance.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 实验四：表征测量
# --------------------------------------------------------------------------
import json as _json


def _probe():
    return {r["checkpoint"].split("/")[0]: r
            for r in _json.load(open(HERE / "probe_physics.json"))}


SUCC = {"lewm_sigreg_low": 34, "lewm_dim192": 52, "lewm_sigreg_high": 38,
        "lewm_hist1": 54, "lewm_hist5": 64,
        "exp2_lewm_s0": 36, "exp2_lewm_s1": 60, "exp2_lewm_s2": 70,
        "exp2_pldm_s0": 32, "exp2_pldm_s1": 36, "exp2_pldm_s2": 62,
        "tworoom": 86}


def fig_scale_vs_success():
    """主发现：决定规划成败的是 embedding 尺度，不是表征里有多少信息。"""
    import statistics as st
    p = _probe()
    groups = {
        "SIGReg sweep": (["lewm_sigreg_low", "lewm_dim192", "lewm_sigreg_high"], RED),
        "horizon sweep": (["lewm_hist1", "lewm_hist5"], BLUE),
        "LeWM (3 seeds)": (["exp2_lewm_s0", "exp2_lewm_s1", "exp2_lewm_s2"], GREEN),
        "PLDM (3 seeds)": (["exp2_pldm_s0", "exp2_pldm_s1", "exp2_pldm_s2"], ORANGE),
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.5))

    xs, ys = [], []
    for label, (keys, c) in groups.items():
        gx = [p[k]["emb_mean_std"] for k in keys]
        gy = [SUCC[k] for k in keys]
        a.scatter(gx, gy, s=80, color=c, label=label, zorder=3,
                  edgecolor="white", linewidth=1.2)
        xs += gx; ys += gy

    # 拟合线（只用 900 步的 11 个点）
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    slope = sum((u - mx) * (v - my) for u, v in zip(xs, ys)) / sum((u - mx) ** 2 for u in xs)
    r = (sum((u - mx) * (v - my) for u, v in zip(xs, ys))
         / ((n - 1) * st.stdev(xs) * st.stdev(ys)))
    lo, hi = min(xs) * 0.5, max(xs) * 1.08
    a.plot([lo, hi], [my + slope * (lo - mx), my + slope * (hi - mx)],
           color=GREY, lw=1.4, ls="--", zorder=2)
    a.annotate(f"r = {r:+.2f}  (n={n} at 900 steps, t=5.9)", (0.03, 0.90),
               xycoords="axes fraction", fontsize=9, color=GREY)

    # 官方 checkpoint：例外，单独标
    a.scatter([p["tworoom"]["emb_mean_std"]], [SUCC["tworoom"]], s=110,
              facecolor="none", edgecolor=INK_, linewidth=1.8, zorder=4)
    a.annotate("released ckpt\n(fully trained,\nbreaks the trend)",
               (p["tworoom"]["emb_mean_std"], SUCC["tworoom"]),
               textcoords="offset points", xytext=(-12, -30), fontsize=7.5,
               color=INK_, ha="right")

    a.set_xscale("log")
    a.set_xlabel("embedding scale (mean per-dim SD, log)")
    a.set_ylabel("CEM success (%)")
    a.set_title("Scale predicts planning. Everything else doesn't.",
                fontsize=9.5, loc="left")
    a.legend(fontsize=7.5, loc="lower right", framealpha=0.95,
             borderpad=0.4, handletextpad=0.4)
    a.set_ylim(22, 97)
    a.set_xlim(6e-4, 0.9)

    # 右：其他候选指标全部不相关
    cands = [("physics probe R²", "probe_r2", 0.244),
             ("effective rank", "eff_rank", -0.220),
             ("distance from Gaussian", "gauss_stat", -0.305),
             ("embedding scale", "emb_mean_std", 0.890)]
    names = [c[0] for c in cands]
    vals = [c[2] for c in cands]
    colors = [GREY] * 3 + [GREEN]
    b.barh(names, vals, color=colors, height=0.55)
    b.axvline(0, color=INK_, lw=0.8)
    for i, v in enumerate(vals):
        b.annotate(f"{v:+.2f}", (v, i), textcoords="offset points",
                   xytext=(6 if v > 0 else -6, 0), va="center",
                   ha="left" if v > 0 else "right", fontsize=8.5)
    b.set_xlim(-0.6, 1.15)
    b.set_xlabel("correlation with planning success (n=11)")
    b.set_title("Only one of them is significant", fontsize=9.5, loc="left")
    b.tick_params(axis="y", labelsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "exp4_scale_vs_success.png", bbox_inches="tight")
    plt.close(fig)


def fig_confound():
    """诊断：前三组实验以为在测 A，其实测到的是 A 对尺度的副作用。"""
    p = _probe()
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.3))

    # 左：三组实验里，成功率如何跟着尺度走
    panels = [
        ("horizon", [("h=3", "lewm_dim192"), ("h=1", "lewm_hist1"), ("h=5", "lewm_hist5")], BLUE),
        ("SIGReg λ", [("0.001", "lewm_sigreg_low"), ("1.0", "lewm_sigreg_high"),
                      ("0.09", "lewm_dim192")], RED),
    ]
    for label, items, c in panels:
        xs = [p[k]["emb_mean_std"] for _, k in items]
        ys = [SUCC[k] for _, k in items]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        a.plot([xs[i] for i in order], [ys[i] for i in order], "o-",
               color=c, lw=1.8, ms=7, label=label)
        for (nm, k), x, y in zip(items, xs, ys):
            dy = 10 if label == "horizon" else -14
            a.annotate(nm, (x, y), textcoords="offset points", xytext=(0, dy),
                       ha="center", fontsize=7.5, color=c)
    a.set_xscale("log")
    a.set_xlabel("embedding scale (log)"); a.set_ylabel("CEM success (%)")
    a.set_title("Both sweeps just moved the scale", fontsize=9.5, loc="left")
    a.legend(fontsize=8); a.set_ylim(25, 75)

    # 右：SIGReg 声称要优化的量，在整个扫描里纹丝不动
    lam = ["0.001", "0.09", "1.0"]
    keys = ["lewm_sigreg_low", "lewm_dim192", "lewm_sigreg_high"]
    gs = [p[k]["gauss_stat"] for k in keys]
    b.bar(lam, gs, color=RED, width=0.5, zorder=3)
    for i, v in enumerate(gs):
        b.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                   xytext=(0, 4), ha="center", fontsize=9)
    b.axhline(170, color=GREY, ls="--", lw=1.2, zorder=2)
    b.annotate("a 4-D linear manifold → 170", (0.985, 170), fontsize=8,
               color=GREY, ha="right", va="bottom",
               xycoords=("axes fraction", "data"),
               bbox=dict(fc="white", ec="none", pad=1.2))
    b.axhline(0.5, color=GREEN, ls="--", lw=1.2, zorder=2)
    b.annotate("true isotropic Gaussian → 0.5", (0.985, 0.55), fontsize=8,
               color=GREEN, ha="right", va="bottom",
               xycoords=("axes fraction", "data"),
               bbox=dict(fc="white", ec="none", pad=1.2))
    b.set_yscale("symlog")
    b.set_xlabel("SIGReg weight λ (1000x range)")
    b.set_ylabel("distance from Gaussian (log)")
    b.set_title("...and never moved what it optimises", fontsize=9.5, loc="left")
    b.set_ylim(0.2, 6000)

    fig.tight_layout()
    fig.savefig(OUT / "exp4_confound.png", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    d = load()
    fig_horizon(d)
    fig_sigreg(d)
    fig_scatter(d)
    fig_seed_variance()
    fig_scale_vs_success()
    fig_confound()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
