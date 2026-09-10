# DreamerV3，基准规模

[English](README_en.md) · **中文**

第一部分靠亲手搭建来证明组件被理解了。这一部分跑官方实现、在真实基准上拿一条拿得出手的曲线。

[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) 在 DMC walker-walk 上，
`dmc_vision` 配置，64×64 像素，action_repeat 2，单张 RTX 5880 Ada 跑了一夜。

| 步数 | 评估回报 |
|---|---|
| 5k | 31 |
| 45k | 568 |
| 195k | 951 |
| 395k | 942 |

任务天花板约 1000。曲线陡升到 85k 时的约 700，在 195k 附近进入 930–955 区间并稳定下来——
见 `../04_analysis/figures/dreamerv3_walker.png`，用
`python ../04_analysis/dreamerv3_curve.py` 重新生成。

`results/eval_curve.jsonl` 是抽出来的评估点（40 个，每个 10 条 episode），
从训练任务的 `metrics.jsonl` 里复制出来，这样曲线不依赖训练目录而独立存在。

## 需要打一个补丁

`dreamer.py` 第 7 行在任何 import 之前硬编码了
`os.environ["MUJOCO_GL"] = "osmesa"`，所以没法从环境变量覆盖。
它的 Dockerfile 会 `apt install libosmesa6`；而一台带 NVIDIA 卡的裸机通常有的是 EGL，
后者还是 GPU 硬件加速的，不是软件渲染：

```python
os.environ.setdefault("MUJOCO_GL", "egl")
```

另外，复用一份已有的 torch 2.4.1 安装会暴露出四个 `torch._inductor` 的运行时依赖，
而它们不在 pinned requirements 里——`sympy`、`functorch`、`filelock`、`jinja2`。
这个配置默认开着 `torch.compile`，所以会死在第一次编译的前向上。
与其等过 2500 步的 prefill 再一次崩一个地发现它们，不如先拿两行函数试一下 `torch.compile`。
