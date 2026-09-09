#!/usr/bin/env bash
# Experiment 2 — same budget, different architecture.
#
# PLDM is one of the baselines le-wm reports against. Its class takes the same
# constructor arguments as LeWM's JEPA (encoder / predictor / action_encoder /
# projector / pred_proj) and exposes the same encode() and predict(), so the
# existing training loop and CEM planner accept it unchanged -- only the
# _target_ paths in the model config differ. See pldm.yaml.
#
# Note on interpretation: this trains PLDM's *architecture* under LeWM's
# training recipe (next-embedding loss + SIGReg), not PLDM's original recipe.
# That is deliberate -- it isolates the architecture and nothing else -- but it
# means the number below is not a faithful reproduction of the PLDM paper.
set -euo pipefail

LEWM=${LEWM:-$HOME/workspace/WAM/le-wm}
PY=${PY:-$LEWM/.venv/bin/python}

cp "$(dirname "$0")/pldm.yaml" "$LEWM/config/train/model/pldm.yaml"
cd "$LEWM"

unset PYTHONPATH
export MUJOCO_GL=egl

# Matched to the h=3 LeWM arm: same data, same 300x3 batches, same seed.
$PY train.py \
  data=tworoom model=pldm \
  output_model_name=pldm_baseline subdir=pldm_baseline \
  trainer.max_epochs=3 \
  +trainer.limit_train_batches=300 +trainer.limit_val_batches=20 \
  wandb.enabled=false \
  2>&1 | tee results/pldm_train.log

$PY eval.py --config-name=tworoom.yaml \
  policy=pldm_baseline/weights_epoch_3.pt \
  output.filename=pldm_baseline_results.txt
