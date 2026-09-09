# Deck

`build_slides.py` writes `slides.html` — one self-contained file, figures
inlined as data URIs. Arrow keys or space to advance; Ctrl+P prints to PDF.

```bash
python build_slides.py
```

Every slide footer carries the repo path it corresponds to, so you can switch
from the deck to the actual code mid-talk.

The DreamerV3 slide reads the current eval point out of the training run's
`metrics.jsonl` at build time rather than hardcoding it — rebuild after the
run advances and the number and the curve both update.
