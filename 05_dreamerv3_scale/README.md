# DreamerV3 at benchmark scale

**English** · [中文](README_zh.md)

Part 1 shows the components are understood by building them. This part runs
the official implementation on a real benchmark to get a curve worth showing.

[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) on DMC
walker-walk, `dmc_vision` config, 64x64 pixels, action_repeat 2, left running
overnight on one RTX 5880 Ada.

| step | eval return |
|---|---|
| 5k | 31 |
| 45k | 568 |
| 195k | 951 |
| 395k | 942 |

Task ceiling is ~1000. The curve rises steeply to ~700 by 85k, reaches the
930-955 band around 195k, and stays there — see
`../04_analysis/figures/dreamerv3_walker.png`, regenerate with
`python ../04_analysis/dreamerv3_curve.py`.

`results/eval_curve.jsonl` is the extracted eval points (40 of them, 10
episodes each), copied out of the run's `metrics.jsonl` so the curve survives
independently of the training directory.

## One patch was needed

`dreamer.py` line 7 hardcodes `os.environ["MUJOCO_GL"] = "osmesa"` before any
import, so it cannot be overridden from the environment. Their Dockerfile
`apt install`s `libosmesa6`; a bare-metal box with an NVIDIA card generally has
EGL instead, which is also GPU-accelerated rather than software rendering:

```python
os.environ.setdefault("MUJOCO_GL", "egl")
```

Reusing an existing torch 2.4.1 install also surfaced four `torch._inductor`
runtime dependencies that the pinned requirements do not pull — `sympy`,
`functorch`, `filelock`, `jinja2`. `torch.compile` is on by default in this
config, so it dies on the first compiled forward. Worth testing
`torch.compile` on a two-line function before waiting through a 2500-step
prefill to discover them one crash at a time.
