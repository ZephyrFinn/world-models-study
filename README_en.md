# world-models-study

**English** · [中文](README.md)

Two families of world model, reproduced and then poked at.

**Dreamer** learns a latent dynamics model by reconstructing what it sees, and
trains a policy entirely inside that model's imagination. **JEPA** — the line
Yann LeCun has been arguing for — drops the reconstruction entirely and only
predicts in representation space, then plans by search at test time. They are
usually discussed as rival philosophies. This repo is what happened when I
built one from scratch, ran the other, and then spent the time trying to break
the second one's central claim.

The short version of what came out of it: **validation loss turned out to be a
bad predictor of whether either model could actually plan.** The
lowest-prediction-loss checkpoint I trained all night was also one of the worst
planners, because its encoder had quietly collapsed — which is exactly the
failure mode the paper's regularizer exists to prevent, so I went and measured
it directly rather than inferring it from the loss curve.

---

## Layout

| | |
|---|---|
| [`01_dreamer_from_scratch/`](01_dreamer_from_scratch) | DreamerV2 written from scratch, single file. RSSM, reward head, actor-critic, imagination rollout. vs. Double-DQN on CartPole, 3 seeds. |
| [`02_lewm_reproduction/`](02_lewm_reproduction) | Getting [LeWorldModel](https://github.com/lucas-maes/le-wm) running and reproducing its released checkpoint. Most of the notes here are about the install, which is where the time actually went. |
| [`03_experiments/`](03_experiments) | The part I care about. Three controlled experiments on LeWorldModel. |
| [`04_analysis/`](04_analysis) | `summary.csv` — every checkpoint, every metric, one table — and the scripts that draw the figures. |
| [`05_dreamerv3_scale/`](05_dreamerv3_scale) | The official DreamerV3 on DMC walker-walk, run overnight to the task ceiling. |
| [`notes/engineering_log_en.md`](notes/engineering_log_en.md) | What broke, and how long each thing cost. |

## Part 1 — Dreamer, written out by hand

Point of this part was to not be able to hide behind someone's `train.py`.
Encoder, RSSM (GRU deterministic state + 32x16 categorical stochastic state
with straight-through gradients), reward head, continue head, actor-critic on
λ-returns, and the imagination rollout, in ~670 lines.

CartPole-v1, 20k env steps, 3 seeds, against a Double-DQN baseline:

| | mean eval return (all evals) | converged (last 3 evals) |
|---|---|---|
| DreamerV2 | **240.1** | **248.6** |
| Double-DQN | 113.8 | 162.2 |

![](01_dreamer_from_scratch/results/comparison_multiseed.png)

The sample-efficiency gap is the expected result and is not the interesting
part. The diagnostics are: KL sits around 0.8 (free bits binding, no collapse
and no blow-up), and the λ-return computed *inside imagination* climbs from 0
to 70+ over training, in step with real performance. That last one is the check
that the imagined rollouts are tracking something real rather than drifting off
into a comfortable hallucination.

![](01_dreamer_from_scratch/results/diagnostics.png)

### At benchmark scale

CartPole proves the components work; it does not produce a curve anyone wants
to look at. So the official
[dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) also ran overnight
on DMC walker-walk:

![](04_analysis/figures/dreamerv3_walker.png)

31 return at 5k steps, 568 at 45k, 951 at 195k, then flat in the 930–955 band
against a ceiling near 1000. Details and the one patch it needed in
[`05_dreamerv3_scale/`](05_dreamerv3_scale).

## Part 2 — LeWorldModel, reproduced

The released tworoom checkpoint, planned with CEM-MPC over 50 held-out
episodes: **86% success**. That is the reference number everything in part 3
gets read against.

Roughly four hours of the night went into installing the thing rather than
running it — a missing `swig`, a `datasets` resolution five years stale, a
`transformers` major version that renamed every ViT weight, and HDF5 support
that is present in the package but disabled by a silent `ImportError`. All
written up in [`02_lewm_reproduction/setup_en.md`](02_lewm_reproduction/setup_en.md),
because that is the part a reproduction write-up usually omits and the part
that actually stops people.

## Part 3 — Three experiments

Every arm below: same dataset, same encoder, same optimizer, same 900 gradient
steps, same CEM planner, same 50 held-out episodes. One thing changes per
experiment.

### 1. How much context does the predictor need?

`history_size` ∈ {1, 3, 5} — the window of past embeddings the predictor sees.

![](04_analysis/figures/exp1_horizon.png)

Prediction error falls monotonically with more context. Planning success does
not: h=1 → h=3 sees loss improve while success *drops* (54% → 52%). At 50
episodes per arm that is inside binomial noise (±~7pp), so the honest reading
is "h=5 is better; the rest is noise" — not a law. That mismatch between what
the loss curve implied and what the eval actually did is what the next two
experiments chase.

### 2. Same budget, different architecture

PLDM — one of the baselines le-wm reports against — takes the same constructor
arguments as LeWM's JEPA and exposes the same `encode`/`predict`, so it drops
into the identical training loop and planner by swapping `_target_` paths in
the model config. Nothing else changes.

| | pred_loss | CEM success |
|---|---|---|
| LeWM h=3 | 0.266 | 52% |
| **PLDM** | **0.298** | **66%** |

PLDM predicts *worse* and plans *better*. One seed, so treat it lightly — but
it is a direct counterexample to reading the loss column as a ranking.

(Caveat worth stating: this trains PLDM's architecture under LeWM's recipe, not
PLDM's own. That isolates the architecture, which is the point, but it is not a
faithful reproduction of the PLDM paper.)

### 3. Does the regularizer do what the paper says?

LeWorldModel's pitch is that two loss terms are enough: next-embedding
prediction, plus a SIGReg term pulling embeddings toward an isotropic Gaussian
so they cannot collapse. Default weight 0.09. I swept it to 0.001 and 1.0.

The loss column alone cannot answer this, and that is the trap — a collapsed
encoder makes prediction *easier*, so `pred_loss` goes down. So instead of
reading the loss I wrote [`probe_collapse.py`](03_experiments/exp3_sigreg/probe_collapse.py):
encode 256 real held-out frames, measure the per-dimension spread of the
resulting embedding, count how many dimensions are effectively dead.

![](04_analysis/figures/exp3_sigreg.png)

At weight 0.001, **69 of 192 embedding dimensions have essentially zero
variance** across real data. The model found the trivial solution. Its
`pred_loss` is 0.004 — two orders of magnitude below anything else trained
here, and completely meaningless. Planning success: 34%, the worst of the
night.

At 10× the default the embedding doesn't collapse, but gets squeezed too
tightly around the target distribution to stay discriminative — 38%.

The published default sits at the top of the inverted U. That is a specific
hyperparameter, independently checked against the failure mode it is named for,
on that failure mode's own terms.

### All six checkpoints

![](04_analysis/figures/loss_vs_success.png)

If validation loss were the right proxy this would trend down-and-to-the-right.
It doesn't. Full numbers in [`04_analysis/summary.csv`](04_analysis/summary.csv).

---

## What this is and isn't

It is: two world-model families reproduced, and three controlled experiments
where exactly one variable moves, including one that verifies a published
claim by measuring the mechanism rather than the metric.

It isn't publication-grade. Every LeWorldModel arm is a single seed at 900
gradient steps with a 50-episode eval, against a released checkpoint that
reaches 86% on the same protocol — so all of these models are undertrained, and
differences under ~10pp are noise. The Dreamer half is better behaved — 3 seeds on CartPole, plus a full-scale
walker-walk run that reaches the task ceiling. Nothing here compares Dreamer against LeWorldModel
head-to-head, because they optimize different objectives (reward vs.
distance-to-goal-embedding) and evaluate on different metrics (return vs.
success rate); forcing them onto one axis would need a shared task definition I
did not build.

## Reproducing

Each part has its own runner. Part 1 is self-contained (`pip install
gymnasium torch`); parts 2 and 3 need a checkout of
[le-wm](https://github.com/lucas-maes/le-wm) — see
[`02_lewm_reproduction/setup_en.md`](02_lewm_reproduction/setup_en.md) first, the
default install does not work as of September 2026.

```bash
cd 01_dreamer_from_scratch && ./run_experiments.sh     # ~20 min, 1 GPU
cd 03_experiments/exp1_horizon && ./run.sh             # ~15 min
cd 03_experiments/exp2_pldm   && ./run.sh              # ~5 min
cd 03_experiments/exp3_sigreg && ./run.sh              # ~10 min
cd 04_analysis && python make_figures.py
```

The DreamerV3 run is a separate overnight job — see
[`05_dreamerv3_scale/`](05_dreamerv3_scale).

Hardware: one RTX 5880 Ada. Peak ~13 GB for a LeWorldModel arm at batch 128,
224px; the Dreamer half fits in under 2 GB.

## References

- Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2), and
  *Mastering Diverse Domains through World Models* (DreamerV3)
- Maes, Le Lidec, Scieur, LeCun, Balestriero, *LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels*, arXiv:2603.19312
- [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch),
  [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm),
  [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
