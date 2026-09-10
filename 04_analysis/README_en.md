# Analysis

**English** · [中文](README.md)

`summary.csv` — one row per checkpoint, every metric collected: training loss
terms, CEM planning success, and the embedding-health diagnostics from
`probe_collapse.py`.

`make_figures.py` reads it and writes `figures/`:

| figure | what it shows |
|---|---|
| `exp1_horizon.png` | prediction error vs. planning success across `history_size` |
| `exp3_sigreg.png` | the inverted U, and the collapse underneath it |
| `loss_vs_success.png` | all six checkpoints on one scatter |

```bash
python make_figures.py
```

The scatter is the summary of the whole repo: on a log x-axis spanning nearly
three orders of magnitude of prediction loss, planning success does not trend.
The best loss belongs to a collapsed encoder that plans badly; the best planner
is a different architecture with middling loss.
