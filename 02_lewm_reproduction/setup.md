# Getting le-wm to run

Notes from actually doing this on a fresh box, September 2026. The README in
the upstream repo is accurate about intent but has drifted from the published
`stable-worldmodel` package in a few places; most of the time here went into
that gap rather than into anything conceptual.

Hardware: single RTX 5880 Ada (48 GB), driver 595, CUDA 13.2. Nothing here
needs that much card — the model is ~15M parameters and peaks around 13 GB
with batch 128 at 224px.

## Environment

```bash
uv venv --python=3.10
uv pip install "stable-worldmodel[train,env]"
```

Four things that bite:

**`box2d-py` fails to build.** It needs `swig`, which is not a Python
dependency and is not on most machines. `uv pip install swig` gets a
prebuilt binary into `.venv/bin` — no apt, no sudo. Do this *before* the
main install, and make sure `.venv/bin` is on `PATH` when the build runs.

**`datasets` resolves to 1.1.1.** A five-year-old version, which calls
`pyarrow.PyExtensionType` — removed in pyarrow 14+. Pin forward:
`uv pip install -U "datasets>=2.14"`.

**`transformers` 5.x renames every ViT weight.** The published checkpoints
were saved under the classic HuggingFace ViT layout
(`encoder.layer.N.attention.attention.query.weight`); transformers 5.x
serves the same architecture as `layers.N.attention.q_proj.weight`, so
`load_state_dict` fails with a wall of missing/unexpected keys. `transformers==4.57.6`
is the last release with the old naming and still satisfies the package's
`>=4.50.0` floor.

**HDF5 support is silently absent.** `stable_worldmodel.data` exposes
`lance / folder / lerobot / video` and no `hdf5`, which is awkward because
the released tworoom dataset *is* an `.h5`. The reader is actually there —
`data/formats/hdf5.py` — behind a `try: import hdf5plugin` that fails
quietly. `uv pip install hdf5plugin` and `HDF5Dataset` reappears.

## Data and checkpoints

Both from the [HF collection](https://huggingface.co/collections/quentinll/lewm).
The Google Drive folder the README links for the baseline suite (PLDM,
DINO-WM, IQL, ...) returns 404 as of this writing, which is why experiment 2
trains PLDM from the implementation shipped inside the package rather than
loading a published baseline checkpoint.

```bash
export STABLEWM_HOME=/path/with/room     # defaults to ~/.stable_worldmodel
# tworoom is the smallest: 3.4 GB compressed, 12.8 GB as .h5
aria2c -x16 -s16 https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst
tar --zstd -xvf tworoom.tar.zst
mkdir -p $STABLEWM_HOME/datasets && mv tworoom.h5 $STABLEWM_HOME/datasets/
```

Layout matters and the README predates it. Datasets are looked up under
`$STABLEWM_HOME/datasets/`, checkpoints under `$STABLEWM_HOME/checkpoints/`,
and `load_pretrained` wants a **state dict in a `.pt`** with a sibling
`config.json` — not the `*_object.ckpt` pickled module the README's
conversion snippet produces. For the released weights that means no
conversion at all:

```bash
hf download quentinll/lewm-tworooms --local-dir $STABLEWM_HOME/hf_tworooms
mkdir -p $STABLEWM_HOME/checkpoints/tworoom
cp $STABLEWM_HOME/hf_tworooms/weights.pt   $STABLEWM_HOME/checkpoints/tworoom/lewm.pt
cp $STABLEWM_HOME/hf_tworooms/config.json  $STABLEWM_HOME/checkpoints/tworoom/
```

`convert_ckpt.py` in this directory is the Hydra-instantiate version of the
README's snippet, kept because it is the thing to reach for if a future
release goes back to object checkpoints.

## Evaluating the released checkpoint

```bash
unset PYTHONPATH                 # ROS on PATH will shadow the venv
export MUJOCO_GL=egl
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm.pt
```

50 held-out episodes, CEM planning against a goal image, roughly 100 s.
Result is in `results/pretrained_tworoom.txt`: **86% success**. That is the
reference every number in `03_experiments/` should be read against — those
runs stop at 900 gradient steps and land between 34% and 66%.
