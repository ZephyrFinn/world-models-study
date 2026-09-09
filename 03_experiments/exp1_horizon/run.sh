#!/usr/bin/env bash
# Experiment 1 — how much context does the predictor need?
#
# LeWorldModel forecasts the next latent embedding from a window of past
# embeddings. `history_size` is that window. Everything else is held fixed:
# same dataset, same encoder, same optimizer, same 900 gradient steps, same
# evaluation episodes. Only the window changes.
#
# Run from a checkout of https://github.com/lucas-maes/le-wm with its venv
# active and the tworoom dataset in place (see 02_lewm_reproduction/setup.md).
set -euo pipefail

LEWM=${LEWM:-$HOME/workspace/WAM/le-wm}
PY=${PY:-$LEWM/.venv/bin/python}
cd "$LEWM"

unset PYTHONPATH            # a ROS install on PATH otherwise shadows the venv
export MUJOCO_GL=egl        # headless render through the NVIDIA driver

# --- train -------------------------------------------------------------
# 300 batches x 3 epochs. Short on purpose: the point is the comparison
# between arms, not to match the paper's full schedule.
for h in 1 3 5; do
  $PY train.py \
    data=tworoom \
    history_size=$h \
    output_model_name=lewm_hist$h subdir=hist${h}_ablation \
    trainer.max_epochs=3 \
    +trainer.limit_train_batches=300 +trainer.limit_val_batches=20 \
    wandb.enabled=false \
    2>&1 | tee results/hist${h}_train.log
done

# --- plan --------------------------------------------------------------
# Same CEM planner, same 50 held-out episodes, for every checkpoint.
# rollout() reads history_size back off predictor.num_frames, so each
# checkpoint plans with the window it was trained on. No override needed.
for h in 1 3 5; do
  $PY eval.py --config-name=tworoom.yaml \
    policy=lewm_hist$h/weights_epoch_3.pt \
    output.filename=hist${h}_eval.txt
done
