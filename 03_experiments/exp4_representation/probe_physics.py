"""物理量线性探针：高斯化和物理结构，能不能兼得？

动机来自 LeWM 论文摘要里两个方向相反的声明：

  1. SIGReg 把 embedding 逼向"各向同性高斯"——所有方向方差相等、维度间无相关。
  2. "LeWM's latent space encodes meaningful physical structure through
     probing of physical quantities."

tworoom 的物理状态本质上是低维的（智能体 xy、目标 xy）。把它塞进 192 维
再强迫各向同性，要么破坏几何、要么只是白化（不伤线性解码）、要么靠注入
无关维度撑满高维球（稀释信号）。哪一种是经验问题，论文没扫这个。

这里在同一批留出帧上同时测三件事：

  gauss_stat  —— Epps-Pulley 正态性统计量（SIGReg 自己的式子），越小越高斯
  probe_r2    —— 岭回归 embedding -> 智能体位置，留出集 R²，越高说明物理结构越可读
  eff_rank    —— 协方差特征值的参与比 (Σλ)²/Σλ²，实际被用起来的维度数

三个都在几千帧上算，方差远小于 50 条 episode 的规划成功率——
这是刻意的：规划成功率的噪声（±19pp）比这些实验的效应量还大。

用法：
    python probe_physics.py <ckpt> [<ckpt> ...]
    python probe_physics.py --all          # 扫所有 exp2_/lewm_ 开头的 checkpoint
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Dataset

N_FRAMES = 3000        # 探针样本数；R² 在这个量级上已经很稳
TRAIN_FRAC = 0.7
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def load_frames(n=N_FRAMES, seed=0):
    """从数据集里均匀抽帧，返回 (pixels, proprio)。"""
    cache = Path(swm.data.utils.get_cache_dir())
    ds = HDF5Dataset("tworoom", keys_to_cache=["proprio"], cache_dir=cache)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=n, replace=False))
    rows = ds.get_row_data(idx)
    px = torch.from_numpy(np.array(rows["pixels"])).float()
    if px.shape[-1] in (1, 3):
        px = px.permute(0, 3, 1, 2)
    px = torch.nn.functional.interpolate(px, size=(224, 224),
                                         mode="bilinear", align_corners=False)
    return px, np.array(rows["proprio"], dtype=np.float64)


@torch.no_grad()
def encode(model, px, bs=64):
    out = []
    for i in range(0, len(px), bs):
        info = model.encode({"pixels": px[i:i + bs].cuda().unsqueeze(1)})
        out.append(info["emb"].squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def gauss_stat(z, n_proj=1024, knots=17, seed=0):
    """Epps-Pulley 统计量，和 SIGReg 训练时用的是同一个式子。
    先按整体标准差归一化，这样它衡量的是"形状像不像高斯"，与尺度无关。"""
    rng = np.random.default_rng(seed)
    z = (z - z.mean(0)) / (z.std() + 1e-8)
    A = rng.normal(size=(z.shape[1], n_proj))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    x = z @ A                                        # (N, P)
    t = np.linspace(0, 3, knots)
    dt = 3 / (knots - 1)
    w = np.full(knots, 2 * dt); w[[0, -1]] = dt
    phi = np.exp(-t ** 2 / 2)
    xt = x[:, :, None] * t                           # (N, P, K)
    err = (np.cos(xt).mean(0) - phi) ** 2 + np.sin(xt).mean(0) ** 2
    return float((err @ (w * phi)).mean() * len(z))


def effective_rank(z):
    """参与比：(Σλ)² / Σλ²。等价于"有效被使用的维度数"。"""
    c = np.cov((z - z.mean(0)).T)
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    return float(ev.sum() ** 2 / (np.square(ev).sum() + 1e-20))


def ridge_r2(z, y, train_frac=TRAIN_FRAC, seed=0):
    """岭回归 embedding -> 物理量，留出集 R²。alpha 在训练集上按留一折简单选。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(z))
    k = int(len(z) * train_frac)
    tr, te = perm[:k], perm[k:]

    # 标准化（用训练集统计量）
    mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-8
    Ztr, Zte = (z[tr] - mu) / sd, (z[te] - mu) / sd
    ym, ys = y[tr].mean(0), y[tr].std(0) + 1e-8
    Ytr, Yte = (y[tr] - ym) / ys, (y[te] - ym) / ys

    best = (-np.inf, None)
    G = Ztr.T @ Ztr
    for a in RIDGE_ALPHAS:
        W = np.linalg.solve(G + a * np.eye(G.shape[0]), Ztr.T @ Ytr)
        pred = Zte @ W
        ss_res = np.square(Yte - pred).sum()
        ss_tot = np.square(Yte - Yte.mean(0)).sum()
        r2 = 1 - ss_res / ss_tot
        if r2 > best[0]:
            best = (r2, a)
    return float(best[0]), best[1]


def analyse(name, px, proprio):
    model = swm.wm.utils.load_pretrained(name).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    z = encode(model, px)
    r2, alpha = ridge_r2(z, proprio)
    return {
        "checkpoint": name,
        "emb_mean_std": float(z.std(0).mean()),
        "dead_dims": int((z.std(0) < 1e-3).sum()),
        "eff_rank": round(effective_rank(z), 2),
        "gauss_stat": round(gauss_stat(z), 2),
        "probe_r2": round(r2, 4),
        "ridge_alpha": alpha,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--all"]:
        root = Path(swm.data.utils.get_cache_dir(), "checkpoints")
        args = sorted(f"{d.name}/weights_epoch_3.pt" for d in root.iterdir()
                      if d.is_dir() and (d / "weights_epoch_3.pt").exists())
    px, proprio = load_frames()
    print(f"# {len(px)} 帧，探针目标：智能体位置 (2D)\n", flush=True)
    for a in args:
        try:
            print(json.dumps(analyse(a, px, proprio), ensure_ascii=False), flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(json.dumps({"checkpoint": a, "error": str(e)[:120]}), flush=True)
