# Deck

**English** · [中文](README.md)

Two builds off the same data and layout:

| | |
|---|---|
| `build_slides.py` | English, `slides.html` |
| `build_slides_zh.py` | Chinese web deck, fixed 1280×720, exports to PDF |
| `build_pptx.py` | Chinese, **native .pptx** with editable text |

```bash
python build_slides.py                 # -> slides.html
python build_slides_zh.py --pdf        # -> slides_zh.html + slides_zh.pdf
python build_pptx.py                   # -> 世界模型的两条路线.pptx
```

Arrow keys or space to advance. The Chinese build passes `--print-to-pdf`
through headless Chrome; `slides_zh.pdf` is committed since that is the
artifact you actually hand to someone.

Every slide footer carries the repo path it corresponds to, so you can switch
from the deck to the code mid-talk.

The DreamerV3 slide reads the training run's `metrics.jsonl` at build time
instead of hardcoding a number, and checks whether its last eight eval points
are flat before claiming the curve is still climbing.

`build_pptx.py` rebuilds the layout with python-pptx rather than pasting screenshots: text is real text and tables are native tables, editable in PowerPoint, WPS or Keynote. Only the charts are PNGs, since they already are images.

Every slide carries **speaker notes** (visible only on the presenter's screen) with timing, the point to land, and the follow-up questions that slide tends to draw.
