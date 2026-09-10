# Experiment 2 — anti-collapse mechanisms (done wrong once, then again)

**English** · [中文](README.md)

> This ran twice. The first design was wrong; the second measured the right
> thing, and its main output is a number I did not enjoy finding:
> **the seed variance of this setup is ±17 percentage points.**

## First attempt: what was wrong with it

The plan was "hold the budget, change only the architecture", implemented by
swapping `_target_` in the model config from `jepa.JEPA` to
`stable_worldmodel.wm.pldm.pldm.PLDM`.

PLDM came out with worse prediction loss but better planning (0.298/66% against
0.266/52%), which looked like a clean counterexample.

**It wasn't**, because methods in the JEPA family **share the architecture and
differ in the loss**:

```
JEPA (encode -> predict in representation space -> no decoder)
 ├── PLDM   (arXiv 2502.14819)  VCReg x4 + temporal alignment + inverse dynamics = 6 terms
 ├── DINO-WM                    frozen pretrained encoder
 └── LeWM   (arXiv 2603.19312)  SIGReg = 1 term
```

The LeWM abstract says it "reduces tunable loss hyperparameters **from six to
one** compared to **the only existing end-to-end alternative**" — that
alternative is PLDM, and the six terms line up exactly.

So swapping the model class while keeping LeWM's loss hollowed PLDM out:
**both arms were LeWM**. The proof is in the training logs — both report a
`sigreg_loss` term, which a real PLDM run would not have.

A second bug compounded it: `train.py` never passes `seed=` to `spt.Manager`,
so weight initialisation is outside `cfg.seed` (the library warns
`User didn't specify seed`). Those 14 points were two different random inits.

## Second attempt: change the actual variable

[`exp2_train.py`](exp2_train.py) holds architecture, data, budget and planner
fixed and swaps only the anti-collapse mechanism:

| | LeWM | PLDM |
|---|---|---|
| core objective | `pred_loss` | `pred_loss` |
| anti-collapse | SIGReg x1 | VCReg x4 (std / std_t / cov / cov_t) |
| extra regulariser | — | temporal alignment |
| extra structure | — | inverse-dynamics head (added by me) |
| **tunable weights** | **1** | **6** |

The seed bug is fixed alongside: `spt.Manager(..., seed=cfg.seed)`. Verified —
two runs at the same seed differ by 0.0006 in sanity loss, two different seeds
by 0.005, an order of magnitude apart. (The residual comes from
non-deterministic GPU reductions under bf16; bit-exactness would additionally
need `torch.use_deterministic_algorithms`, at a speed cost.)

Three seeds per arm.

## Results

| | success (seed 0/1/2) | mean | within-arm SD |
|---|---|---|---|
| LeWM | 36% / 60% / 70% | 55.3% | **17.5pp** |
| PLDM | 32% / 36% / 62% | 43.3% | **16.3pp** |

Welch t-test: +12pp, t = 0.87, 95% CI [-27, +51]. **Nowhere near significant.**

## Three findings

### 1. Seed variance is ±17pp, and the whole design was underpowered by ~10x

This is the main output. Same config, same data, seed alone: 36% to 70%.

At that variance, **detecting a true 10-point difference at 80% power needs
about 45 seeds per arm.** I had one. Run
[`../../04_analysis/significance.py`](../../04_analysis/significance.py) for
the full set.

The consequence: every success-rate comparison in this repo — the horizon sweep
in experiment 1, the SIGReg success-rate gap in experiment 3 — **is inside the
noise**.

### 2. `pred_loss` is not comparable across models

PLDM's prediction loss is consistently 13–16x lower (0.016–0.021 against
0.25–0.34), which looks decisive. But its embedding scale is also far smaller.
`pred_loss` is an MSE in embedding space, so it scales with the square of the
representation's scale; divide that out before comparing:

| | pred_loss | emb SD | normalised |
|---|---|---|---|
| LeWM | 0.251–0.345 | 0.26–0.41 | mean **2.84** |
| PLDM | 0.016–0.021 | 0.07–0.31 | mean **2.66** |

**Effectively level.** The "16x better" was entirely scale.

This is a milder version of the collapse trap from experiment 3: nothing has to
collapse — a regulariser that merely squeezes the representation makes the MSE
look better for free.
[`../exp3_sigreg/probe_collapse.py`](../exp3_sigreg/probe_collapse.py) now
reports `scale_sq` so the normalisation is one division away.

### 3. PLDM looks less stable across seeds

`pldm_s2` has a mean embedding SD of 0.306 against 0.071 and 0.073 for the
other two seeds — **4x apart within one configuration**. It is also PLDM's best
run by success rate (62%).

Suggests six loss terms let different inits converge to representations of very
different scale. n=3 cannot establish it; noted for the record.

## Answering the paper's claim

LeWM claims to be simpler (1 hyperparameter against 6) without losing
performance. At this 900-step budget:

- **Performance**: 55.3% against 43.3%. Direction favours LeWM,
  **statistically indistinguishable** (t=0.87).
- **Simplicity**: confirmed. Running PLDM required writing an inverse-dynamics
  head and choosing weights for six loss terms — and **the paper's values were
  not available to me, so all six are 1.0**. That uncertainty is itself the
  cost of "six hyperparameters", not a flaw in the implementation.

## Reproducing

```bash
python exp2_train.py data=tworoom +loss_type=lewm seed=0 \
  output_model_name=exp2_lewm_s0 subdir=exp2_lewm_s0 \
  trainer.max_epochs=3 +trainer.limit_train_batches=300 +trainer.limit_val_batches=20 \
  wandb.enabled=false
# swap loss_type to pldm, seed over 0/1/2 — six runs
```

`results_v2/` holds every log and eval from the second attempt; `results/` keeps
the first for comparison.

## Lessons

1. **`diff` the implementations before claiming an architecture comparison.**
   Matching constructor signatures are not evidence of different architectures.
2. **Work out what the axis of variation in a family of methods actually is.**
   For JEPAs it is the anti-collapse mechanism, not the network — I had not
   established what I was comparing before comparing it.
3. **Fix the evaluation size before designing experiments.** Twenty minutes
   spent running three seeds to measure the variance would have told me none of
   these experiments could resolve their expected effects. I did it in the
   opposite order.
