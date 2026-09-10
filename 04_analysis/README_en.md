# Analysis

**English** · [中文](README.md)

`summary.csv` — one row per checkpoint, every metric collected: training loss
terms, CEM planning success, and the embedding-health diagnostics from
`probe_collapse.py`.

`make_figures.py` reads it and writes `figures/`:

| figure | what it shows |
|---|---|
| `exp1_horizon.png` | prediction error vs. planning success across `history_size` |
| `exp3_sigreg.png` | the inverted U, and the collapse underneath it |
| `loss_vs_success.png` | all six checkpoints on one scatter |

`significance.py` runs a two-proportion z-test on every success-rate
comparison. It exists because I initially read the gaps between arms as
results; only one of the six turns out to be significant.

`dreamerv3_curve.py` draws the DreamerV3 learning curve straight from the
training run's `metrics.jsonl`, so it refreshes on any rerun.

```bash
python make_figures.py
python dreamerv3_curve.py
python significance.py
```

The scatter is the summary of the whole repo: on a log x-axis spanning nearly
three orders of magnitude of prediction loss, planning success does not trend.

**Read it at the right strength, though.** The four middle points are within
noise of each other — the empirical floor is ~15 points, measured accidentally
in experiment 2. What carries the conclusion is the leftmost point: lowest
loss, near-worst planning, and an anomaly backed by a direct measurement of the
representation (270x variance gap, 69 dead dimensions) rather than by the noisy
success-rate metric.
