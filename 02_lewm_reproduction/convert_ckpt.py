"""Convert HF weights.pt + config.json into the *_object.ckpt format eval.py expects.
Usage: python convert_ckpt.py <env>   where <env> in {pusht, tworooms, cube, reacher}

Note: the config.json shipped alongside the HF weights is a full Hydra config
(with _target_ keys pointing at stable_worldmodel.wm.lewm.*), not the flat
kwargs the README's inline snippet assumed. We instantiate it directly via
hydra.utils.instantiate instead of manually reconstructing jepa.JEPA.
"""
import json
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
import hydra.utils

import stable_worldmodel as swm

env = sys.argv[1] if len(sys.argv) > 1 else "tworooms"
# le-wm README uses "pusht/lewm" etc as the checkpoint stem (singular dir names)
out_env = {"tworooms": "tworoom"}.get(env, env)

src = Path(swm.data.utils.get_cache_dir(), f"hf_{env}")
out = Path(swm.data.utils.get_cache_dir(), out_env, "lewm_object.ckpt")

cfg = OmegaConf.create(json.loads((src / "config.json").read_text()))
model = hydra.utils.instantiate(cfg)

sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print(f"saved -> {out}")
