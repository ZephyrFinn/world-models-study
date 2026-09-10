# 讲稿

[English](README_en.md) · **中文**

同一套数据、同一套版式，两个构建：

| | |
|---|---|
| `build_slides.py` | 英文版，输出 `slides.html` |
| `build_slides_zh.py` | 中文版网页，固定 1280×720，可导出 PDF |
| `build_pptx.py` | 中文版 **原生 .pptx**，文字可编辑 |

```bash
python build_slides.py                 # -> slides.html
python build_slides_zh.py --pdf        # -> slides_zh.html + slides_zh.pdf
python build_pptx.py                   # -> 世界模型的两条路线.pptx
```

方向键或空格翻页。中文版通过 headless Chrome 传 `--print-to-pdf` 导出；
`slides_zh.pdf` 入库了，因为那才是真正会递给别人的东西，生成的 HTML 不入库。

每页页脚标着它对应的仓库路径，讲的时候可以直接从幻灯片切到代码。

DreamerV3 那一页在构建时读训练任务的 `metrics.jsonl`，而不是把数字写死，
并且会检查最后八个评估点是否已经走平，再决定要不要说"曲线仍在上升"。

`build_pptx.py` 用 python-pptx 重新搭版式，不是把网页截图贴进去——文字是真文字，表格是原生表格，在 PowerPoint / WPS / Keynote 里都能直接改。只有图表是 PNG，因为它们本来就是图。
