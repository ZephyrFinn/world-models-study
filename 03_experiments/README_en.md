# Experiments

**English** · [中文](README.md)

Three questions about LeWorldModel, one variable each.

Common protocol for every arm: tworoom dataset, ViT-tiny encoder, AdamW at
5e-5, batch 128, **300 batches × 3 epochs = 900 gradient steps**, then CEM-MPC
planning over the **same 50 held-out episodes** with the same seed. Anything
not named as the variable is identical across arms.

900 steps is short on purpose. The released checkpoint reaches 86% on this
protocol; everything here lands between 34% and 66%, so these are all
undertrained models being compared to each other, not to the paper.

| | question | variable | verdict |
|---|---|---|---|
| [`exp1_horizon/`](exp1_horizon) | how much context does the predictor need? | `history_size` ∈ {1,3,5} | single seed, all inside ±17pp noise — no conclusion |
| [`exp2_pldm/`](exp2_pldm) | does the anti-collapse mechanism matter? | SIGReg (1 term) vs VCReg+align+IDM (6) | 3 seeds, t=0.87, not significant; **measured seed variance ±17pp** |
| [`exp3_sigreg/`](exp3_sigreg) | does the regularizer prevent collapse? | `loss.sigreg.weight` ∈ {0.001, 0.09, 1.0} | **yes** — 69/192 dead dims measured directly (the one conclusion that holds) |

Results roll up into [`../04_analysis/summary.csv`](../04_analysis/summary.csv).

## On noise (read this before any number below)

The experiment-2 redo measured it directly: **same config, same data, same
budget, seed alone, three runs.**

| | success | within-arm SD |
|---|---|---|
| LeWM | 36% / 60% / 70% | 17.5pp |
| PLDM | 32% / 36% / 62% | 16.3pp |

![](../04_analysis/figures/exp2_seed_variance.png)

**Seed variance is about ±17 points.** At that variance, resolving a true
10-point difference at 80% power needs roughly **45 seeds per arm**. Experiments
1 and 3 have **one** each.

The consequence is blunt: **every success-rate comparison here sits inside the
noise** — the 12 points between h=5 and h=3, the 18 points across the SIGReg
switch. Run [`../04_analysis/significance.py`](../04_analysis/significance.py)
for the full set.

## What survives

Only the measurements that do not route through success rate:

**Collapse (SIGReg weight 0.001)**: mean embedding SD 0.0012 against 0.325, a
270x difference, and 69/192 dead dimensions against 0/192.

Those come from encoding real held-out data. They are not inferred from a loss
curve and do not pass through a 50-episode sample. They are the only conclusion
here that the current statistical power supports.

The precise statement: **disabling the regulariser causes representation
collapse (established); the collapse degrades planning (consistent, but not
established at this power).**
