#!/usr/bin/env bash
# DreamerV3 on DMC walker-walk (pixel/vision), full 1M-step official benchmark run.
set -euo pipefail
cd /home/tz/workspace/WAM/dreamerv3-torch
unset PYTHONPATH
source /home/tz/workspace/WAM/.venv/bin/activate
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk \
  --logdir ./logdir/dmc_walker_walk "$@"
