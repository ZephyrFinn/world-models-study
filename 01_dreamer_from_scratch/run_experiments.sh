#!/usr/bin/env bash
# DreamerV2 vs Double-DQN on CartPole-v1, 3 seeds each.
#
# Dreamer runs on GPU (the RSSM + imagination rollout is where the time goes);
# DQN is small enough that CPU is faster than paying the transfer overhead.
# ~20 min total on one card.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
STEPS=20000

for seed in 0 1 2; do
  # seed 0 writes to the un-suffixed dir, matching how the first run was launched
  if [ "$seed" = "0" ]; then sfx=""; else sfx="_s${seed}"; fi

  $PY dreamer.py \
    --env CartPole-v1 \
    --logdir "runs/dreamer_cartpole${sfx}" \
    --tag "dreamer${sfx:-_s0}" \
    --device cuda --seed "$seed" \
    --total_steps $STEPS --prefill 1000 \
    --eval_every 1000 --eval_episodes 5 \
    --batch_size 16 --seq_len 32 --horizon 15 --train_every 5 \
    2>&1 | tee "runs_dreamer${sfx}.log"

  $PY baseline_dqn.py \
    --env CartPole-v1 \
    --logdir "runs/dqn_cartpole${sfx}" \
    --tag "dqn${sfx:-_s0}" \
    --device cpu --seed "$seed" \
    --total_steps $STEPS --prefill 1000 \
    --eval_every 1000 --eval_episodes 5 \
    2>&1 | tee "runs_dqn${sfx}.log"
done

# reward curves + the RSSM diagnostics (KL, imagination lambda-return, model loss)
$PY plot.py
$PY plot_multiseed.py
