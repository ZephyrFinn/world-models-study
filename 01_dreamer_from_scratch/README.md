# DreamerV2 from scratch

`dreamer.py` is the whole thing in one file — no framework, so there is nowhere
to hide a component I did not understand.

- **Encoder** — MLP for vector observations (CNN path written for 64x64 pixels
  but not used in these runs)
- **RSSM** — GRU deterministic state + 32x16 categorical stochastic state with
  straight-through gradients. `obs_step` (posterior, conditioned on a real
  observation) and `img_step` (prior, action only) kept separate; `observe`
  runs the posterior along a real sequence, `imagine` rolls the prior forward
  without ever seeing an observation.
- **Reward head** and **continue head** on the latent state
- **Actor-critic** on λ-returns with a slow-moving target critic
- **Imagination rollout** — the actor is trained entirely on prior rollouts
  branched off posterior states from replay

`baseline_dqn.py` is a Double-DQN for comparison.

## Results

CartPole-v1, 20k env steps, 3 seeds each.

| | mean eval return | converged (last 3 evals) |
|---|---|---|
| DreamerV2 | 240.1 | 248.6 |
| Double-DQN | 113.8 | 162.2 |

![](results/comparison_multiseed.png)

Dreamer pulls ahead from ~3k steps and stays there with lower variance. DQN is
spikier and shows the classic catastrophic-forgetting dip — one seed reaches
500 mid-run and falls back to ~100.

The sample-efficiency gap is the textbook result. The diagnostics are the part
worth looking at:

![](results/diagnostics.png)

- **KL** settles around 0.8 with free bits at 1.0 — binding, so the posterior
  is not collapsing onto the prior, and not exploding either.
- **Imagination λ-return** climbs 0 → 70+ over training, tracking real eval
  return. This is the check that the model's imagined rollouts correspond to
  something: the actor only ever sees imagined states, so if this number rose
  while real return stayed flat, the world model would be hallucinating
  comfortably and the policy would be optimising against a fantasy.

## Running

```bash
./run_experiments.sh     # 3 seeds x (Dreamer + DQN), ~20 min on one GPU
python plot_multiseed.py
```

Raw per-run metrics in `results/runs/*/metrics.csv`; the trained weights
(`agent.pt`) are not committed.
