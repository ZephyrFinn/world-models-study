"""DreamerV3 learning curve on DMC walker-walk.

Reads the metrics.jsonl that dreamerv3-torch writes as it trains and plots
eval_return against environment steps. Run it again as training continues --
the run targets 1e6 steps and the curve is still climbing well past 1e5.

Usage:
    python dreamerv3_curve.py [path/to/metrics.jsonl]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DEFAULT = Path.home() / "workspace/WAM/dreamerv3-torch/logdir/dmc_walker_walk/metrics.jsonl"
BLUE, GREY = "#2a78d6", "#8a857a"


def read_evals(path):
    steps, returns = [], []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # partial line at the tail
            if "eval_return" in rec:
                steps.append(rec["step"])
                returns.append(rec["eval_return"])
    return steps, returns


def main(path):
    steps, returns = read_evals(path)
    if not steps:
        sys.exit(f"no eval_return records in {path}")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(steps, returns, "o-", color=BLUE, lw=2, ms=5)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval return (10 episodes)")
    ax.set_title("DreamerV3 on DMC walker-walk", fontsize=10, loc="left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(1080, max(returns) * 1.2))

    ax.axhline(1000, color=GREY, ls=":", lw=1)
    ax.annotate("task ceiling ~1000", (steps[0], 1000), fontsize=8,
                color=GREY, va="bottom", ha="left",
                textcoords="offset points", xytext=(2, 3))

    last = returns[-1]
    # describe the tail honestly: still rising, or flat near the ceiling?
    tail = returns[-8:]
    plateaued = len(tail) >= 8 and (max(tail) - min(tail)) < 0.08 * max(tail)
    shape = ("plateaued near ceiling" if plateaued else "still climbing")
    ax.annotate(f"{last:.0f} @ {steps[-1]/1000:.0f}k steps\n({shape}, run targets 1M)",
                (steps[-1], last), textcoords="offset points", xytext=(-10, -40),
                ha="right", fontsize=8, color=BLUE)

    fig.tight_layout()
    out = HERE / "figures" / "dreamerv3_walker.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"{len(steps)} eval points, latest {last:.1f} at step {steps[-1]} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
