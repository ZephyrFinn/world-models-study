"""Aggregate multiple seeds per algorithm into mean +/- std eval-return curves."""
import argparse, csv, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(pathlib.Path(path) / "metrics.csv")))
    steps = np.array([int(float(r["env_steps"])) for r in rows])
    ev = np.array([float(r["eval_return"]) for r in rows])
    return steps, ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", action="append", nargs="+", required=True,
                    help='label followed by run dirs, e.g. --group DreamerV2 runs/dreamer_cartpole runs/dreamer_cartpole_s1 runs/dreamer_cartpole_s2')
    p.add_argument("--out", default="comparison_multiseed.png")
    p.add_argument("--title", default="Sample efficiency (mean +/- std over seeds)")
    cfg = p.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for group in cfg.group:
        label, dirs = group[0], group[1:]
        curves = [load(d) for d in dirs]
        steps = curves[0][0]
        for s, _ in curves:
            assert np.array_equal(s, steps), "eval step grids must match across seeds"
        evs = np.stack([e for _, e in curves], 0)  # [n_seeds, T]
        mean, std = evs.mean(0), evs.std(0)
        ax.plot(steps, mean, label=f"{label} (n={len(dirs)})", linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
        for s, e in curves:
            ax.plot(s, e, linewidth=0.6, alpha=0.35, color=ax.lines[-1].get_color() if False else None)
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
