# Experiment 3 — does SIGReg actually prevent collapse?

**English** · [中文](README.md)

LeWorldModel's central claim is that a JEPA can be trained stably end-to-end
from pixels with only two loss terms: next-embedding prediction, plus SIGReg
regularising the embedding distribution toward an isotropic Gaussian. Weight
defaults to 0.09.

Swept to 0.001 (effectively off) and 1.0 (10x), against the h=3 arm as the
0.09 point.

| weight | pred_loss | mean emb. std | dead dims | CEM success |
|---|---|---|---|---|
| 0.001 | **0.004** | 0.0012 | **69 / 192** | **34%** |
| 0.09 (default) | 0.266 | 0.325 | 0 / 192 | **52%** |
| 1.0 | 1.074 | 0.115 | 0 / 192 | 38% |

## Why the loss column cannot answer this

At weight 0.001 the prediction loss is 0.004 — sixty times better than the
default, and the best number produced by any run in this repo. It is also
completely meaningless. A collapsed encoder emits a near-constant vector, and
predicting a constant is trivial, so the objective is minimised by destroying
the representation. Read the loss alone and you would ship this checkpoint.

## Measuring it instead

`probe_collapse.py` loads a checkpoint, encodes 256 real held-out frames, and
reports the per-dimension standard deviation of the resulting embeddings plus
a count of dimensions with std < 1e-3.

```
$ python probe_collapse.py lewm_sigreg_low/weights_epoch_3.pt
{
  "mean_std": 0.00115,
  "dead_dims": 69,
  "dead_frac": 0.359
}
```

36% of the embedding is flat across real data. Planning success drops to 34%,
the worst of any checkpoint here.

At the other end, 10x the default does not collapse anything (0 dead dims) but
compresses the spread to 0.115 — the embeddings are pushed so hard toward the
target distribution that they stop separating states well enough to plan with.
38%.

The published default sits at the peak of the resulting inverted U. Which is
the point: the paper names a failure mode and picks a constant to avoid it,
and the constant holds up when you go and check for the failure mode directly.
