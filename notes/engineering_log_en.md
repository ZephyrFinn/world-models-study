# Engineering log

**English** · [中文](engineering_log.md)

Kept because reproduction write-ups usually skip this part, and it is the part
that actually costs the days. Rough order, one night.

---

**Network.** Machine sits behind a proxy that reaches GitHub and PyPI proper
but not the domestic mirrors, and the mirrors are the only fast path for large
wheels. So: proxy on for `git`/HuggingFace, proxy explicitly *unset* for
`uv pip`. `aria2c` additionally refuses the `socks://` form in `ALL_PROXY` and
dies with an unhelpful `unrecognized proxy format`; it needs the variable
cleared and `--https-proxy` passed as a flag. With `-x16` the HF datasets came
down at ~2 MB/s instead of ~380 KB/s single-stream.

**`PYTHONPATH` pollution.** A ROS Jazzy install exports
`/opt/ros/jazzy/lib/python3.12/site-packages` on the global `PYTHONPATH`, which
shadows every venv on the box with Python 3.12 packages. Every script in this
repo starts with `unset PYTHONPATH` for that reason. Cost about 20 minutes of
confusion before it showed up in `sys.path`.

**Don't symlink into someone else's venv.** To avoid downloading torch three
times I symlinked `torch/`, `torchvision/`, `nvidia_*` from an existing project
venv into the new ones. This works right up until `uv` decides an unpinned
`torch>=2` wants upgrading — at which point it writes *through* the symlink and
silently upgrades the other project's torch in place (2.7.0+cu128 →
2.14.0+cu130), leaving that project's `dist-info` claiming the old version. The
other project's long-running job survived only because Linux keeps unlinked
inodes alive for open file handles.

Recovering it was its own detour: `uv sync` reported `Checked in 0.00ms` and
did nothing, even against a deliberately emptied venv, even with `--reinstall`.
The cause is that the project is a uv *workspace* whose root declares no
dependencies — torch lives in a workspace member's optional group. Plain
`uv sync` syncs only the root. `uv sync --all-packages --extra skrl-torch`
restored it correctly.

Lesson taken: copy, or use the package manager's own cache. Never a bare
symlink into a venv something else is using.

**`libosmesa` isn't installed, and dreamerv3-torch hardcodes it.**
`dreamer.py` line 7 sets `os.environ["MUJOCO_GL"] = "osmesa"` before any
import, so it cannot be overridden from outside. Their Dockerfile
`apt install`s `libosmesa6`; a bare metal box with an NVIDIA card usually has
EGL instead, which is also GPU-accelerated rather than software. One-line
change to `os.environ.setdefault("MUJOCO_GL", "egl")`.

**`torch.compile` needs packages that pinned installs don't pull.** After
reusing a torch 2.4.1 install, DreamerV3 got as far as the first compiled
forward and then died on `No module named 'sympy'`, then `functorch`, then
`filelock`, then `jinja2` — all runtime dependencies of `torch._inductor` that
were not in the requirements file. Faster to test `torch.compile` on a
two-line function than to rediscover them one crash at a time through a
2500-step prefill.

**Package drift in le-wm.** The repo's README and the published
`stable-worldmodel` 0.1.1 have diverged. Four separate breaks, all in
[`02_lewm_reproduction/setup_en.md`](../02_lewm_reproduction/setup_en.md): `swig`
missing for `box2d-py`, `datasets` resolving to 1.1.1 against a modern
`pyarrow`, `transformers` 5.x renaming every ViT weight so the released
checkpoints won't load, and HDF5 support hidden behind a silent
`import hdf5plugin` failure. Also: `load_pretrained` wants a state dict in a
`.pt` under `checkpoints/<name>/` with a sibling `config.json`, not the
`*_object.ckpt` pickled module the README's snippet builds.

The Google Drive link for the baseline checkpoint suite (PLDM, DINO-WM, IQL,
GCBC, ...) returns 404. That is why experiment 2 trains PLDM from the
implementation inside the package instead of loading a published baseline —
which turned out better anyway, since training it under the identical recipe
and budget is a cleaner comparison than borrowing someone's checkpoint trained
on an unknown schedule.

**`embed_dim` is not a free hyperparameter.** First planned ablation was latent
dimension. It crashes:
`mat1 and mat2 shapes cannot be multiplied (512x192 and 96x2048)`. The encoder
is ViT-tiny with a fixed 192-d hidden size, but the projector config sets its
`input_dim` from `${embed_dim}` — so the default only works because 192 happens
to equal ViT-tiny's width. Changing the "latent dimension" alone desynchronises
them. To sweep it properly you would have to co-scale the backbone, which
changes parameter count and stops being a clean single-variable ablation.
Switched to `history_size`, which is wired through consistently
(`predictor.num_frames: ${history_size}`).

**GPU scheduling.** Three LeWorldModel arms in parallel plus DreamerV3 hit OOM
on a 48 GB card — a single arm peaks near 13 GB at batch 128 / 224px, and the
end-of-epoch validation spikes line up across runs that started together. Ran
them two at a time after that.

**stdout buffering hid two hours of progress.** DreamerV3's console log sat at
step 45000 for over an hour while `metrics.jsonl` — written with an explicit
flush — showed it had passed 80000. Checked `nvidia-smi` and process state
first and concluded "healthy but slow"; the log file was simply stale. Trust
the metrics file, not the redirected stdout.

**Mistook noise for a finding, and then found the problem ran deeper.**
Experiment 2 was meant to compare architectures: drop PLDM into LeWM's training
loop at matched budget. PLDM came out predicting worse and planning better,
which I wrote up as a counterexample to reading the loss column as a ranking.

Being asked why that disagreed with the paper's own claim surfaced the first
error: **methods in the JEPA family share the architecture and differ in the
loss.** PLDM uses VCReg x4 + temporal alignment + inverse dynamics (six terms),
LeWM uses SIGReg (one) — exactly the abstract's "from six to one compared to
the only existing end-to-end alternative". Swapping only the model class while
keeping LeWM's loss hollowed PLDM out. Both training logs report a
`sigreg_loss` term, which settles it.

A second bug alongside: `train.py` never passes `seed=` to `spt.Manager`, so
weight init sits outside `cfg.seed` — the library warns about it. Sanity loss
differs before training starts: 5.147 against 5.001.

The redo fixed both: `exp2_train.py` swaps the actual variable (the
anti-collapse mechanism) and passes the seed through. Three seeds per arm.

**The redo's result is more uncomfortable than the original mistake.** Same
config, seed alone: 36% / 60% / 70%, a within-arm SD of ±17 points. At that
variance, resolving a true 10-point difference needs about 45 seeds per arm.
Experiments 1 and 3 have one each — **the whole design is underpowered by
roughly an order of magnitude**, and I only learned that after running all
three.

It also exposed a third thing: **`pred_loss` is not comparable across models.**
PLDM's is consistently 13-16x lower, but its embedding scale is far smaller;
normalise by scale squared and they level out (2.84 against 2.66). A milder
version of the collapse trap — nothing has to collapse, a regulariser that
merely squeezes the representation makes the MSE look better for free.
`probe_collapse.py` now reports `scale_sq` so that division is at hand.

Three lessons: `diff` the implementations before claiming an architecture
comparison; establish what the axis of variation in a family of methods
actually is; and **fix the evaluation size before designing experiments** —
twenty minutes measuring seed variance up front would have shown that none of
the three could resolve their expected effects.

---

Naming sediment worth knowing about if you read the raw logs: the `history_size=3`
arm is recorded as `lewm_dim192` in
[`03_experiments/exp1_horizon/results/hist3_train.log`](../03_experiments/exp1_horizon/results/hist3_train.log)
and in the Hydra overrides dumped into its eval file. That run started life as
the control arm of the abandoned `embed_dim` ablation and was reused as the
h=3 baseline rather than retrained. Filenames here say `hist3`; log contents
are unedited.
