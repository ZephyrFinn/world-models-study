# Experiment 2 — LeWM vs PLDM at matched budget

**English** · [中文](README_zh.md)

PLDM ships inside `stable_worldmodel` and is one of the baselines le-wm reports
against. Its class takes the same constructor arguments as LeWM's JEPA and
exposes the same `encode()` / `predict()`, so it drops into the same training
loop and the same CEM planner — only the `_target_` paths in `pldm.yaml`
differ from the LeWM model config.

Trained against the h=3 LeWM arm: same data, same 900 steps, same eval.

| | pred_loss | CEM success | mean emb. std |
|---|---|---|---|
| LeWM h=3 | 0.266 | 52% | 0.325 |
| PLDM | 0.298 | **66%** | 0.380 |

PLDM has the worse prediction loss and the better planner. Both have healthy
embeddings (no dead dimensions), so this is not a collapse artifact.

Two caveats, both real:

- One seed. 66 vs 52 is a 14pp gap against a ~7pp standard error — suggestive,
  not established.
- This is PLDM's **architecture** under **LeWM's recipe** (next-embedding loss
  + SIGReg), not PLDM's own training objective. That is what isolates the
  architecture, but it means the number is not a reproduction of the PLDM
  paper and should not be cited as one.

What it is good for: a counterexample to reading the loss column as a ranking.
