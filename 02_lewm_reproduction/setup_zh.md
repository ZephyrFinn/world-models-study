# 把 le-wm 跑起来

[English](setup.md) · **中文**

2026 年 9 月在一台干净机器上实际操作后的笔记。上游 README 在意图上是准确的，
但它和已发布的 `stable-worldmodel` 包之间已经产生了几处漂移；
这里绝大部分时间花在了填这些缝，而不是花在任何概念性的东西上。

硬件：单张 RTX 5880 Ada（48 GB），驱动 595，CUDA 13.2。
其实用不着这么大的卡——模型约 1500 万参数，batch 128、224px 下峰值约 13 GB。

## 环境

```bash
uv venv --python=3.10
uv pip install "stable-worldmodel[train,env]"
```

四个会咬人的地方：

**`box2d-py` 编译失败。** 它需要 `swig`，而 swig 不是 Python 依赖，多数机器上也没有。
`uv pip install swig` 会把一个预编译二进制装进 `.venv/bin`——不需要 apt，不需要 sudo。
这一步要在主安装**之前**做，并确保构建时 `.venv/bin` 在 `PATH` 上。

**`datasets` 会解析到 1.1.1。** 一个五年前的版本，它调用 `pyarrow.PyExtensionType`——
这个 API 在 pyarrow 14+ 里已被移除。往前钉住：`uv pip install -U "datasets>=2.14"`。

**`transformers` 5.x 重命名了每一个 ViT 权重。** 发布的 checkpoint 是按经典
HuggingFace ViT 布局保存的（`encoder.layer.N.attention.attention.query.weight`），
而 transformers 5.x 把同一套架构按 `layers.N.attention.q_proj.weight` 提供，
于是 `load_state_dict` 会以一大片 missing/unexpected key 失败。
`transformers==4.57.6` 是最后一个用旧命名、同时仍满足包里 `>=4.50.0` 下限的版本。

**HDF5 支持是静默缺失的。** `stable_worldmodel.data` 暴露出
`lance / folder / lerobot / video`，没有 `hdf5`——这很尴尬，因为发布的 tworoom 数据集**就是** `.h5`。
读取器其实在那儿（`data/formats/hdf5.py`），只是被一句 `try: import hdf5plugin` 悄悄挡住了。
`uv pip install hdf5plugin`，`HDF5Dataset` 就回来了。

## 数据和 checkpoint

都来自 [HF collection](https://huggingface.co/collections/quentinll/lewm)。
README 里指向 baseline 套件（PLDM、DINO-WM、IQL……）的那个 Google Drive 链接，
截至撰写时返回 404——这也是为什么实验二是用包里自带的实现训练 PLDM，
而不是加载一个已发布的 baseline checkpoint。

```bash
export STABLEWM_HOME=/有空间的路径     # 默认 ~/.stable_worldmodel
# tworoom 最小：压缩 3.4 GB，解开成 .h5 是 12.8 GB
aria2c -x16 -s16 https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst
tar --zstd -xvf tworoom.tar.zst
mkdir -p $STABLEWM_HOME/datasets && mv tworoom.h5 $STABLEWM_HOME/datasets/
```

目录结构是有讲究的，而 README 写在这次改动之前。
数据集在 `$STABLEWM_HOME/datasets/` 下查找，checkpoint 在 `$STABLEWM_HOME/checkpoints/` 下，
而且 `load_pretrained` 要的是**一个 `.pt` 里的 state dict**加一个同级的 `config.json`——
不是 README 那段转换代码产出的 `*_object.ckpt` pickle 模块。
对已发布的权重来说，这意味着根本不需要转换：

```bash
hf download quentinll/lewm-tworooms --local-dir $STABLEWM_HOME/hf_tworooms
mkdir -p $STABLEWM_HOME/checkpoints/tworoom
cp $STABLEWM_HOME/hf_tworooms/weights.pt   $STABLEWM_HOME/checkpoints/tworoom/lewm.pt
cp $STABLEWM_HOME/hf_tworooms/config.json  $STABLEWM_HOME/checkpoints/tworoom/
```

这个目录里的 `convert_ckpt.py` 是 README 那段代码的 Hydra `instantiate` 版本，
留着是为了以后哪个版本又改回 object checkpoint 时能派上用场。

## 评估已发布的 checkpoint

```bash
unset PYTHONPATH                 # PATH 上的 ROS 会盖掉 venv
export MUJOCO_GL=egl
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm.pt
```

50 条留出 episode，针对一张目标图像做 CEM 规划，约 100 秒。
结果在 `results/pretrained_tworoom.txt`：**86% 成功率**。
这是 `03_experiments/` 里每个数字都应该对照的参照——
那些运行停在 900 步梯度更新，落在 34% 到 66% 之间。
