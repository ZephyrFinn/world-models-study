# 工程日志

[English](engineering_log_en.md) · **中文**

记下来是因为复现类文章通常跳过这一段，而这一段才是真正吃掉时间的地方。大致按发生顺序，一个晚上。

---

**网络。** 机器在一个代理后面，能到 GitHub 和 PyPI 官方源，但到不了国内镜像，
而镜像又是大 wheel 唯一的快路径。所以：`git`/HuggingFace 走代理，`uv pip` 明确**取消**代理。
`aria2c` 还额外拒绝 `ALL_PROXY` 里的 `socks://` 形式，会以一句没什么帮助的
`unrecognized proxy format` 直接退出；得把变量清掉，用 `--https-proxy` 参数传进去。
开 `-x16` 之后 HF 数据集从单线程约 380 KB/s 提到约 2 MB/s。

**`PYTHONPATH` 污染。** 一个 ROS Jazzy 安装把
`/opt/ros/jazzy/lib/python3.12/site-packages` 导出到了全局 `PYTHONPATH`，
于是机器上每一个 venv 都被 Python 3.12 的包盖住。
这个仓库里所有脚本开头都有 `unset PYTHONPATH` 就是为了它。
在它出现在 `sys.path` 里之前，大约困惑了 20 分钟。

**不要把软链指进别人的 venv。** 为了避免下载三遍 torch，
我把 `torch/`、`torchvision/`、`nvidia_*` 从一个已有项目的 venv 软链进了新环境。
这一直没问题——直到 `uv` 判定一个没锁版本的 `torch>=2` 需要升级，
于是它**穿过**软链写入，悄无声息地把另一个项目的 torch 就地升级了
（2.7.0+cu128 → 2.14.0+cu130），而那个项目的 `dist-info` 还声称是旧版本。
那个项目当时有个跑了一天的任务还活着——它没崩，只是因为 Linux 会为已打开的文件句柄保留被 unlink 的 inode。

恢复过程本身又是一段插曲：`uv sync` 报告 `Checked in 0.00ms` 然后什么也不做，
即使对着一个故意清空的 venv，即使加了 `--reinstall`。
原因是那个项目是一个 uv **workspace**，而它的根不声明任何依赖——
torch 住在某个 workspace 成员的可选依赖组里。裸 `uv sync` 只同步根。
`uv sync --all-packages --extra skrl-torch` 才正确地把它恢复回来。

得到的教训：要么拷贝，要么用包管理器自己的缓存。
永远不要用一个裸软链指向别人正在使用的 venv。

**`libosmesa` 没装，而 dreamerv3-torch 把它写死了。**
`dreamer.py` 第 7 行在任何 import 之前设置 `os.environ["MUJOCO_GL"] = "osmesa"`，
所以没法从外部覆盖。它的 Dockerfile 会 `apt install libosmesa6`；
而一台带 NVIDIA 卡的裸机通常有的是 EGL，后者还是 GPU 加速而非软件渲染。
改成 `os.environ.setdefault("MUJOCO_GL", "egl")`，一行的事。

**`torch.compile` 需要一些 pinned 安装不会带来的包。**
复用一份 torch 2.4.1 之后，DreamerV3 走到第一次编译的前向就死了：
先是 `No module named 'sympy'`，然后 `functorch`，然后 `filelock`，然后 `jinja2`——
全都是 `torch._inductor` 的运行时依赖，而它们不在 requirements 文件里。
与其穿过 2500 步的 prefill 一次崩一个地重新发现它们，
不如先拿一个两行的函数试一下 `torch.compile`。

**le-wm 的包漂移。** 仓库 README 和已发布的 `stable-worldmodel` 0.1.1 已经分叉。
四处独立的断裂，全部写在
[`02_lewm_reproduction/setup.md`](../02_lewm_reproduction/setup.md) 里：
`box2d-py` 缺 `swig`、`datasets` 在现代 `pyarrow` 下解析到 1.1.1、
`transformers` 5.x 重命名了每一个 ViT 权重导致发布的 checkpoint 加载不了、
以及 HDF5 支持被一句静默的 `import hdf5plugin` 失败挡住。
另外：`load_pretrained` 要的是 `checkpoints/<name>/` 下一个 `.pt` 里的 state dict 加同级 `config.json`，
不是 README 那段代码构建的 `*_object.ckpt` pickle 模块。

那个指向 baseline checkpoint 套件（PLDM、DINO-WM、IQL、GCBC……）的 Google Drive 链接返回 404。
这就是为什么实验二用包内自带的实现训练 PLDM，而不是加载一个已发布的 baseline——
结果反而更好，因为用完全相同的配方和预算训练出来，比借用一个训练调度未知的 checkpoint
是更干净的对照。

**`embed_dim` 不是自由超参数。** 原计划的第一个消融是潜在维度。它崩了：
`mat1 and mat2 shapes cannot be multiplied (512x192 and 96x2048)`。
encoder 是 ViT-tiny，隐藏维度固定 192，而 projector 的配置从 `${embed_dim}` 取它的 `input_dim`——
所以默认值能跑通，只是因为 192 恰好等于 ViT-tiny 的宽度。
单独改"潜在维度"会让两者失配。要正经扫它，就必须同步缩放骨干网络，
那会改变参数量，也就不再是干净的单变量消融了。
于是改成 `history_size`，它是被一致接线的（`predictor.num_frames: ${history_size}`）。

**GPU 调度。** 三个 LeWorldModel 实验并行再加上 DreamerV3，在一张 48 GB 卡上会 OOM——
单个实验在 batch 128 / 224px 下峰值接近 13 GB，而同时启动的几个运行，
它们 epoch 末尾的验证峰值会撞在一起。之后改成一次跑两个。

**stdout 缓冲藏起了两小时进度。** DreamerV3 的终端日志在第 45000 步停了一个多小时，
而带显式 flush 写入的 `metrics.jsonl` 显示它已经过了 80000 步。
我先查了 `nvidia-smi` 和进程状态，得出"健康但慢"的结论；实际上只是日志文件陈旧了。
信指标文件，别信重定向的 stdout。

**把噪声当成了发现。** 实验二原本是要比架构：把 PLDM 塞进 LeWM 的训练循环，
同预算跑一遍。结果 PLDM 预测更差却规划更好，我当成了"loss 不能给模型排序"的直接反例，
写进了 README 和讲稿。

后来被问到"官方不是宣称 LeWM 比 PLDM 好吗"，我才去 `diff` 了一下实现——
`stable_worldmodel` 里 `pldm/module.py` 和 `lewm/module.py` **逐字节相同**，
两个主文件只差 `rollout()` 里的推理优化。这从来就不是一次架构对比。

而且 `train.py` 调 `spt.Manager` 时没传 `seed=`，模型权重初始化不受 `cfg.seed` 控制
（库自己会警告 `User didn't specify seed`）。证据是训练开始前的 sanity check loss
就已经不同：5.147 对 5.001。

所以那 14 个百分点是随机初始化造成的重跑方差。双比例 z 检验 p = 0.15，CI 跨过 0。

保留这个实验是因为它意外给出了这套设置的噪声底（约 15 个百分点），
而那把尺子反过来说明：我设计的三组实验，预期效应量全都在 10-20 个百分点，
**50 条 episode 的评估规模从一开始就不足以分辨它们**。
教训有两条：想比架构先 `diff` 实现，别信"接口一样能互换"；
评估规模要先于实验设计确定，而不是跑完再补统计。

---

如果你去读原始日志，有一处命名沉积值得知道：`history_size=3` 那一组在
[`03_experiments/exp1_horizon/results/hist3_train.log`](../03_experiments/exp1_horizon/results/hist3_train.log)
以及它 eval 文件里 dump 出来的 Hydra 覆盖参数中，被记作 `lewm_dim192`。
那次运行原本是那个被放弃的 `embed_dim` 消融的对照组，后来直接复用成了 h=3 baseline，没有重训。
这里的文件名写的是 `hist3`；日志内容未经改动。
