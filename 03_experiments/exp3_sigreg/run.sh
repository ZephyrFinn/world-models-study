#!/usr/bin/env bash
# Experiment 3 — does the regularizer do what the paper says it does?
#
# LeWorldModel's claim is that two loss terms suffice: next-embedding
# prediction, plus SIGReg pulling the embedding distribution toward an
# isotropic Gaussian so it cannot collapse. Weight defaults to 0.09.
#
# Sweep it at 0.001 (effectively off) and 1.0 (10x), against the h=3 arm from
# experiment 1 as the 0.09 point -- same data, same steps, same eval.
#
# The loss number alone will not tell you whether collapse happened: a
# collapsed encoder makes prediction *easier*, so pred_loss goes DOWN. Hence
# probe_collapse.py, which encodes real frames and measures the per-dimension
# spread of the embedding directly.
set -euo pipefail

LEWM=${LEWM:-$HOME/workspace/WAM/le-wm}
PY=${PY:-$LEWM/.venv/bin/python}
HERE="$(cd "$(dirname "$0")" && pwd)"

cp "$HERE/probe_collapse.py" "$LEWM/probe_collapse.py"
cd "$LEWM"

unset PYTHONPATH
export MUJOCO_GL=egl

train_one () {  # $1 = weight, $2 = run name
  $PY train.py \
    data=tworoom \
    loss.sigreg.weight=$1 \
    output_model_name=$2 subdir=$2 \
    trainer.max_epochs=3 \
    +trainer.limit_train_batches=300 +trainer.limit_val_batches=20 \
    wandb.enabled=false \
    2>&1 | tee "$HERE/results/$2.log"
}

train_one 0.001 lewm_sigreg_low
train_one 1.0   lewm_sigreg_high
# the 0.09 point is lewm_hist3 from experiment 1 -- not retrained here

# --- plan, then look inside --------------------------------------------
for ck in lewm_sigreg_low lewm_sigreg_high lewm_hist3; do
  $PY eval.py --config-name=tworoom.yaml \
    policy=$ck/weights_epoch_3.pt \
    output.filename=${ck}_eval.txt
  $PY probe_collapse.py $ck/weights_epoch_3.pt | tee "$HERE/results/${ck}_probe.json"
done
