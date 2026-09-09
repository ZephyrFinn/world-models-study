"""Plot reward curves and sample-efficiency comparison from metrics.csv files.

Usage: python plot.py --runs runs/dreamer_cartpole:Dreamer runs/dqn_cartpole:DQN --out out.png
Each --runs entry is "path:label". Reads metrics.csv (env_steps, eval_return, ...) from each dir.
"""
import argparse, csv, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(pathlib.Path(path) / "metrics.csv")))
    steps = [int(float(r["env_steps"])) for r in rows]
    ev = [float(r["eval_return"]) for r in rows]
    ev_std = [float(r.get("eval_std", 0) or 0) for r in rows]
    return steps, ev, ev_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="path:label pairs")
    p.add_argument("--out", default="comparison.png")
    p.add_argument("--title", default="Sample efficiency")
    cfg = p.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in cfg.runs:
        path, _, label = spec.partition(":")
        label = label or path
        steps, ev, ev_std = load(path)
        ax.plot(steps, ev, label=label, linewidth=2)
        lo = [m - s for m, s in zip(ev, ev_std)]
        hi = [m + s for m, s in zip(ev, ev_std)]
        ax.fill_between(steps, lo, hi, alpha=0.15)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval return")
    ax.set_title(cfg.title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.out, dpi=150)
    print("saved", cfg.out)


if __name__ == "__main__":
    main()
