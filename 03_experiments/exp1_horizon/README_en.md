# Experiment 1 — imagination horizon

**English** · [中文](README.md)

`history_size` is the number of past embeddings the predictor conditions on
before forecasting the next one. Swept over {1, 3, 5}; 3 is the published
default.

| history_size | pred_loss | CEM success | dead dims |
|---|---|---|---|
| 1 | 0.345 | 54% | 0 / 192 |
| 3 (default) | 0.266 | 52% | 0 / 192 |
| 5 | **0.254** | **64%** | 0 / 192 |

Prediction error falls with more context, which is what you would expect —
more frames to disambiguate motion from. Planning success does not follow it:
h=1 → h=3 improves loss and *loses* two points of success rate.

⚠️ **The experiment-2 redo forces a further retraction here.**

Measuring seed variance directly with three seeds gave **±17 points** — larger
than the biggest gap in this table (12 points, h=5 against h=3). All three arms
here are single-seed, so **h=5's 64% is inside the noise too and is not a
result.**

The honest statement: at current statistical power this experiment measured
nothing. Making it meaningful needs roughly 45 seeds per arm — see
[`../../04_analysis/significance.py`](../../04_analysis/significance.py).

`rollout()` reads the window size back off `predictor.num_frames`, so each
checkpoint plans with the window it trained on without an eval-side override.

`./run.sh` reproduces both halves. Raw logs and eval dumps in `results/`.
