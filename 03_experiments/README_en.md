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
| [`exp1_horizon/`](exp1_horizon) | how much context does the predictor need? | `history_size` ∈ {1,3,5} | h=5 best; h=1 vs h=3 is noise |
| [`exp2_pldm/`](exp2_pldm) | does the architecture matter at fixed budget? | ~~JEPA vs PLDM~~ | **didn't work** — same architecture; measured the noise floor instead |
| [`exp3_sigreg/`](exp3_sigreg) | does the regularizer prevent collapse? | `loss.sigreg.weight` ∈ {0.001, 0.09, 1.0} | yes — collapse reproduced at 0.001 |

Results roll up into [`../04_analysis/summary.csv`](../04_analysis/summary.csv).

## On noise

50 episodes is a binomial sample. At p≈0.5 the standard error is ~7pp, so a
95% interval on any single number here is roughly ±14pp. Differences that
survive that: sigreg 0.001 (34%) vs. the 0.09 default (52%), and the released
checkpoint (86%) vs. everything. Differences that do not: h=1 (54%) vs. h=3
(52%), and PLDM (66%) vs. h=5 (64%).

The SIGReg result is the one worth trusting, and not only because the gap is
larger — it is the only one with an independent mechanistic measurement
(embedding variance, dead dimension count) pointing the same direction as the
success rate.
