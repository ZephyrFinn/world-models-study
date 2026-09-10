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


if __name__ == "__main__":
    d = load()
    fig_horizon(d)
    fig_sigreg(d)
    fig_scatter(d)
    fig_seed_variance()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
