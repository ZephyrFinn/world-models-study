# Deck

**English** · [中文](README.md)

Two builds off the same data and layout:

| | |
|---|---|
| `build_slides.py` | English, `slides.html` |
| `build_slides_zh.py` | Chinese, fixed 1280×720 so it prints one slide per page |

```bash
python build_slides.py                 # -> slides.html
python build_slides_zh.py --pdf        # -> slides_zh.html + slides_zh.pdf
```

Arrow keys or space to advance. The Chinese build passes `--print-to-pdf`
through headless Chrome; `slides_zh.pdf` is committed since that is the
artifact you actually hand to someone.

Every slide footer carries the repo path it corresponds to, so you can switch
from the deck to the code mid-talk.

The DreamerV3 slide reads the training run's `metrics.jsonl` at build time
instead of hardcoding a number, and checks whether its last eight eval points
are flat before claiming the curve is still climbing.
