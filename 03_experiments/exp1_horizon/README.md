# Experiment 1 — imagination horizon

**English** · [中文](README_zh.md)

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

At 50 episodes per arm, ±7pp standard error, that 54 vs 52 is nothing. The
defensible claim is "h=5 helps"; the rest is noise. Worth another point at
h=7-8 before believing the trend continues, and worth more seeds before
believing any of it.

`rollout()` reads the window size back off `predictor.num_frames`, so each
checkpoint plans with the window it trained on without an eval-side override.

`./run.sh` reproduces both halves. Raw logs and eval dumps in `results/`.
