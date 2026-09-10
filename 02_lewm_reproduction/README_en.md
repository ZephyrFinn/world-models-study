# Reproducing LeWorldModel

**English** · [中文](README.md)

Upstream: [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) —
*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from
Pixels* (Maes, Le Lidec, Scieur, LeCun, Balestriero; arXiv:2603.19312).

A JEPA world model: ~15M parameters, ViT-tiny encoder, autoregressive
transformer predictor over embeddings, no decoder anywhere. It never
reconstructs a pixel. Planning is CEM-MPC at test time — sample action
sequences, roll them forward in embedding space, keep the ones whose predicted
final embedding lands nearest the goal image's embedding.

## Result

Released tworoom checkpoint, 50 held-out episodes, CEM-MPC:

**86% success** — [`results/pretrained_tworoom.txt`](results/pretrained_tworoom.txt)

```bash
./eval_pretrained.sh
```

This is the reference number for everything in `../03_experiments/`, where the
same protocol on 900-step checkpoints lands between 34% and 66%.

## Read setup_en.md first

[`setup_en.md`](setup_en.md) — the default install does not work as of September
2026. Four breaks between the README and the published `stable-worldmodel`
0.1.1, plus a checkpoint layout that has changed shape. That document is most
of the value in this directory.

`convert_ckpt.py` builds the model via Hydra `instantiate` from the shipped
`config.json`. Not needed for the current release format (which wants a plain
state dict), kept for when it is.
