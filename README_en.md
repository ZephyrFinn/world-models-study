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

The short version: **the checkpoint with the lowest prediction loss had a
collapsed encoder.** Loss of 0.004, sixty times better than normal, and 69 of
192 embedding dimensions with near-zero variance — exactly the failure mode the
paper's regularizer exists to prevent. I measured that directly rather than
inferring it from the loss curve.

And a less flattering one: **I mistook noise for a finding.** Experiment 2 was
supposed to compare architectures. It took a while to work out that methods in
the JEPA family share the architecture and differ in the loss — I swapped the
model class but kept LeWM's loss, so both arms were the same thing.

Redone properly (actually swapping the anti-collapse mechanism, three seeds per
arm) it produced this project's most useful and least fun number: **same config,
seed alone, planning success ranges from 36% to 70% — a within-arm SD of ±17
points.** Detecting a true 10-point difference at that variance needs roughly 45
seeds per arm. I had one.

![](04_analysis/figures/exp2_seed_variance.png)

So **every success-rate comparison in this repo sits inside the noise.** What
survives is the direct measurement of representation collapse.

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

### 2. Swapping the anti-collapse mechanism (wrong once, then redone)

The plan was "hold the budget, change only the architecture", implemented by
editing `_target_` in the model config. **That design was wrong** — methods in
the JEPA family share the architecture and differ in the loss:

| | LeWM | PLDM |
|---|---|---|
| anti-collapse | SIGReg x1 | VCReg x4 + temporal alignment + inverse dynamics |
| tunable weights | **1** | **6** |

Which is exactly the abstract's "reduces tunable loss hyperparameters **from six
to one** compared to **the only existing end-to-end alternative**" — that
alternative is PLDM. Swapping the class while keeping LeWM's loss hollowed PLDM
out; both arms were LeWM.

The redo also fixed a real bug: `train.py` never passes `seed=` to
`spt.Manager`, so weight init was uncontrolled — which is where the first
attempt's 14-point gap came from.

Redone, three seeds per arm:

| | success | mean | within-arm SD |
|---|---|---|---|
| LeWM | 36% / 60% / 70% | 55.3% | 17.5pp |
| PLDM | 32% / 36% / 62% | 43.3% | 16.3pp |

Welch t = 0.87, **nowhere near significant**. Direction favours LeWM; statistics
cannot separate them.

It also surfaced that **`pred_loss` is not comparable across models**: PLDM's is
consistently 13–16x lower, but its embedding scale is far smaller — normalise by
scale and they level out (2.84 vs 2.66). A milder version of the collapse trap:
nothing has to collapse, a regulariser that merely squeezes the representation
makes the MSE look better for free.

Details in [`03_experiments/exp2_pldm/`](03_experiments/exp2_pldm).

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
variance** across real data — mean std 0.0012 against 0.325 when healthy, a
270x difference. The model found the trivial solution. Its `pred_loss` is
0.004, two orders of magnitude below anything else trained here, and completely
meaningless.

Planning success drops to 34%, consistent in direction. To be precise about
what that supports: **the collapse is measured, not inferred**, while the
52% → 34% magnitude sits at p = 0.064 — the evaluation is too small to pin it.
The honest statement is that disabling the regularizer causes collapse
(established) and that the collapse degrades planning (consistent but
underpowered).

At 10× the default the embedding doesn't collapse, but gets squeezed too
tightly around the target distribution to stay discriminative — 38%.

The published default sits at the top of the inverted U. That is a specific
hyperparameter, independently checked against the failure mode it is named for,
on that failure mode's own terms.

### All six checkpoints

![](04_analysis/figures/loss_vs_success.png)

If validation loss were the right proxy this would trend down-and-to-the-right.
It doesn't — though note that the four middle points are within noise of each
other (see experiment 2). What carries the conclusion is the leftmost point:
lowest loss, near-worst planning, and an anomaly corroborated by an independent
measurement of the representation.

Full numbers in [`04_analysis/summary.csv`](04_analysis/summary.csv), tests in
[`04_analysis/significance.py`](04_analysis/significance.py).

---

## What this is and isn't

It is: two world-model families reproduced, and three controlled experiments
where exactly one variable moves, including one that verifies a published
claim by measuring the mechanism rather than the metric.

It isn't publication-grade, and the weakness is worse than I first thought.

The seed variance measured in the experiment-2 redo is **±17 points** — same
config, seed alone, a 34-point spread. At that variance, resolving a true
10-point difference needs about 45 seeds per arm; experiments 1 and 3 have one
each. **So every success-rate comparison in this repo sits inside the noise**,
including the horizon sweep and the SIGReg success-rate gap.

What survives is the measurement that does not route through success rate: under
collapse, embedding SD is 0.0012 against 0.325 (270x) and 69 of 192 dimensions
are dead. Those come from encoding real data, not from a 50-episode sample.

There is also no head-to-head Dreamer vs LeWorldModel comparison (different
objectives, different metrics), and experiment 2's first design was simply
wrong — written up in its own README. The Dreamer half is better behaved — 3 seeds on CartPole, plus a full-scale
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
