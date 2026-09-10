#!/usr/bin/env bash
# 实验四：在已有 checkpoint 上做低方差表征测量。不需要重新训练。
#
# 三个指标都在 3000 帧留出数据上计算，方差远小于 50 条 episode 的规划成功率
# （后者噪声 ±19pp，比前三组实验的效应量还大）：
#
#   gauss_stat  Epps-Pulley 正态性统计量 —— SIGReg 声称要优化的东西
#   probe_r2    岭回归 embedding -> 智能体位置，留出集 R² —— 论文的 probing 口径
#   eff_rank    协方差特征值参与比 —— 实际被用起来的维度数
set -euo pipefail
LEWM=${LEWM:-$HOME/workspace/WAM/le-wm}
PY=${PY:-$LEWM/.venv/bin/python}
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/probe_physics.py" "$LEWM/"
cd "$LEWM"
unset PYTHONPATH
export MUJOCO_GL=egl

$PY probe_physics.py --all | tee "$HERE/results/probe_physics.log"
