#!/usr/bin/env bash
# Reproduce the reference number: the released LeWorldModel checkpoint,
# planning on tworoom. 50 held-out episodes, CEM-MPC, ~100 s.
# Expect 86% success. See setup.md for how the checkpoint has to be laid out.
set -euo pipefail

LEWM=${LEWM:-$HOME/workspace/WAM/le-wm}
PY=${PY:-$LEWM/.venv/bin/python}
cd "$LEWM"

unset PYTHONPATH
export MUJOCO_GL=egl

$PY eval.py --config-name=tworoom.yaml \
  policy=tworoom/lewm.pt \
  output.filename=pretrained_tworoom.txt
