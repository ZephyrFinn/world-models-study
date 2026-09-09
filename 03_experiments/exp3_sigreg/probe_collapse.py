"""Diagnostic: does the learned embedding space actually spread out, or collapse?

Loads a trained checkpoint, encodes a batch of *real* tworoom frames, and reports
per-dimension embedding std plus the fraction of "dead" dimensions (std < 1e-3).
A collapsed representation (e.g. SIGReg weight -> 0) should show near-zero std
across most of the 192 embedding dimensions -- the model can then trivially
minimize next-embedding prediction loss by predicting a near-constant vector,
which is exactly the failure mode SIGReg is supposed to prevent.

Usage: python probe_collapse.py <checkpoint_name>
  e.g. python probe_collapse.py lewm_sigreg_low/weights_epoch_3.pt
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Dataset

name = sys.argv[1]
model = swm.wm.utils.load_pretrained(name)
model = model.to("cuda").eval()
model.requires_grad_(False)
model.interpolate_pos_encoding = True

cache_dir = Path(swm.data.utils.get_cache_dir())
dataset = HDF5Dataset("tworoom", keys_to_cache=["action", "proprio"], cache_dir=cache_dir)

# grab a batch of real frames spread across the dataset (not just the first episode)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(dataset), size=256, replace=False))
rows = dataset.get_row_data(idx)
pixels = torch.from_numpy(np.array(rows["pixels"])).float()  # (B, H, W, C) or (B, C, H, W)
if pixels.shape[-1] in (1, 3):
    pixels = pixels.permute(0, 3, 1, 2)
pixels = torch.nn.functional.interpolate(pixels, size=(224, 224), mode="bilinear", align_corners=False)
pixels = pixels.to("cuda")

with torch.no_grad():
    info = {"pixels": pixels.unsqueeze(1)}  # add fake time dim (B, T=1, C, H, W)
    out = model.encode(info)
    emb = out["emb"].squeeze(1)  # (B, D)

std = emb.std(dim=0).cpu().numpy()
dead = (std < 1e-3).sum()
result = {
    "checkpoint": name,
    "n_samples": int(pixels.shape[0]),
    "embed_dim": int(emb.shape[-1]),
    "mean_std": float(std.mean()),
    "median_std": float(np.median(std)),
    "min_std": float(std.min()),
    "max_std": float(std.max()),
    "dead_dims": int(dead),
    "dead_frac": float(dead / emb.shape[-1]),
}
print(json.dumps(result, indent=2))
