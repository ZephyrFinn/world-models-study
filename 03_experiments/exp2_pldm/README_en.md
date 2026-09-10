# Experiment 2 — a failed architecture comparison, and what it measured instead

**English** · [中文](README.md)

> **This experiment did not do what it set out to do.** It is kept because what
> it accidentally measured turned out to be more useful than the plan, and
> because the mistake is worth writing down.

## The plan

PLDM is one of the baselines le-wm reports against, and it ships inside
`stable_worldmodel`. Its class takes the same constructor arguments as LeWM's
JEPA and exposes the same `encode()` / `predict()`, so swapping two `_target_`
paths in the model config drops it into the identical training loop and
planner. The intent: hold the budget fixed, change only the architecture.

| | pred_loss | CEM success | mean emb. std | dead dims |
|---|---|---|---|---|
| LeWM h=3 | 0.266 | 52% | 0.325 | 0 / 192 |
| PLDM | 0.298 | 66% | 0.380 | 0 / 192 |

It looks like a clean counterexample: predicts worse, plans better. **It isn't.**

## Why it doesn't hold

**One: the two "architectures" are the same architecture.**

```bash
$ diff stable_worldmodel/wm/pldm/module.py stable_worldmodel/wm/lewm/module.py
$ echo $?
0
```

Byte-identical. Same `Predictor`, `Embedder`, `MLP`, `Transformer`. The two
top-level files, `pldm.py` and `lewm.py`, are 153 and 154 lines and differ only
in the `rollout()` inference path — dtype handling, caching and detaching the
initial embedding, batching the action encoding. Nothing structural.

So this was never an architecture comparison.

**Two: the two runs got different random initialisations, unseeded.**

`train.py` constructs `spt.Manager(...)` without passing `seed=`, and `Manager`
only calls `pl.seed_everything` when it receives one — it even prints
`User didn't specify seed, runs won't be exactly reproducible!`. The `cfg.seed`
value only feeds the `torch.Generator` used for the dataset split; it never
reaches weight initialisation.

The evidence: the sanity-check validation loss differs *before training starts* —
5.147 for LeWM against 5.001 for PLDM. Not one gradient step in. Different
initial weights.

**Three: 14 points is not significant.**

Two-proportion z-test (`../../04_analysis/significance.py`): z = 1.44,
p = 0.15, 95% CI [-5, +33] points. It crosses zero.

## What it actually measured

**Same architecture, same data, same recipe, same 900 steps, different random
init: pred_loss moves 0.266 → 0.298 (12%), success rate moves 52% → 66%
(14 points).**

That makes this an accidental run-to-run variance measurement — which is
precisely the number this repo needed and did not otherwise have: **the noise
floor**.

With it, the other experiments get a ruler:

| comparison | gap | against the noise floor |
|---|---|---|
| h=1 vs h=3 | 2pp | well inside |
| h=5 vs h=3 | 12pp | inside |
| SIGReg default vs collapsed | 18pp | just above, still not significant (p=0.064) |
| released ckpt vs my h=3 | 34pp | significant (p<0.001) |

Conclusion: **under this setup, no success-rate gap under ~15 points can be
treated as a result** — including the one this experiment was meant to produce.

## What I take from it

"Same constructor signature, drops in as a replacement" is not evidence of a
different architecture. `diff` the implementations first. I only checked after
being asked why my result disagreed with the paper's claim, by which point the
number was already in the README and in a talk.

The second lesson: **fix the evaluation size before designing the experiments.**
A 50-episode eval can only separate gaps above roughly 30 points, while all
three experiments here were designed around expected effects of 10–20. Even if
the architectures had genuinely differed, this setup could not have shown it.

## Reproducing

`./run.sh` reruns training and evaluation. To turn it into a *deliberate*
variance measurement, run one config five times with different seeds — which is
what should have happened.
