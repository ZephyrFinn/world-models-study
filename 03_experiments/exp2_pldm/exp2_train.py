"""实验二（重做）：同架构、同数据、同预算，只换防塌缩机制。

第一次做这个实验时我只换了模型类的 `_target_` 路径，以为那是在换架构。
不是——JEPA 这一族方法共享架构，PLDM 和 LeWM 的区别全在 loss 上。
换了个同构的类、却沿用 LeWM 的 loss，等于把 PLDM 抽空，两组跑的都是 LeWM。

这一版换的是真正的变量：

    LeWM :  pred_loss + 0.09 * SIGReg                       (1 个可调权重)
    PLDM :  pred_loss + std + std_t + cov + cov_t
                      + temp_align + idm                     (6 项)

对应 LeWorldModel 论文摘要那句 "reduces tunable loss hyperparameters
from six to one compared to the only existing end-to-end alternative"——
PLDM 就是那个 alternative（arXiv 2502.14819）。

另外修了一个 bug：原 train.py 调 spt.Manager 时没传 seed，
模型权重初始化不受 cfg.seed 控制（库自己会警告），导致"跑多个 seed"根本无效。
这里显式传进去，多 seed 才有意义。

用法：
    python exp2_train.py loss_type=lewm seed=0
    python exp2_train.py loss_type=pldm seed=0
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from torch import nn

from module import SIGReg
from stable_worldmodel.wm.loss import PLDMLoss
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


class InverseDynamics(nn.Module):
    """从相邻两个 embedding 预测中间的动作。PLDM 的 IDM 项需要它，
    LeWM 不需要——这本身就是"多一项 loss 就多一块结构"的代价之一。"""

    def __init__(self, emb_dim, action_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, action_dim)
        )

    def forward(self, z):  # z: (B, T, D) -> (B, T-1, A)
        pair = torch.cat([z[:, :-1], z[:, 1:]], dim=-1)
        return self.net(pair)


def forward_fn(self, batch, stage, cfg):
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]            # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # 两组共有的核心目标：下一步 embedding 预测
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    if cfg.loss_type == "lewm":
        # 一个防塌缩项，一个可调权重
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]

    elif cfg.loss_type == "pldm":
        # 六项。权重全取 1.0：PLDM 论文的具体取值我没有，
        # 而"要调六个权重"恰恰是 LeWM 声称要消除的成本，这里如实记录。
        a_pred = self.idm(emb)
        a_target = batch["action"][:, 1:emb.size(1)]
        if a_target.shape[1] != a_pred.shape[1]:      # 对齐时间维
            n = min(a_target.shape[1], a_pred.shape[1])
            a_pred, a_target = a_pred[:, :n], a_target[:, :n]
        terms = self.pldm_loss(emb, a_pred=a_pred, a_target=a_target)
        for k, v in terms.items():
            output[k] = v
        output["loss"] = output["pred_loss"] + sum(terms.values())
    else:
        raise ValueError(cfg.loss_type)

    losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # 本脚本新增的两个键，允许命令行覆盖
    with open_dict(cfg):
        cfg.setdefault("loss_type", "lewm")

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR", None), **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.model.action_encoder.input_dim = action_dim

    dataset.transform = spt.data.transforms.Compose(*transforms)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True,
                                        drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    world_model = hydra.utils.instantiate(cfg.model)

    extras = {}
    if cfg.loss_type == "lewm":
        extras["sigreg"] = SIGReg(**cfg.loss.sigreg.kwargs)
    else:
        extras["pldm_loss"] = PLDMLoss()
        extras["idm"] = InverseDynamics(cfg.embed_dim, action_dim)

    module = spt.Module(
        model=world_model,
        forward=partial(forward_fn, cfg=cfg),
        optim={"model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }},
        **extras,
    )

    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1)],
        num_sanity_val_steps=1,
        logger=None,
        enable_checkpointing=True,
    )
    # 关键修复：显式传 seed，否则权重初始化不受控，多 seed 实验无意义
    manager = spt.Manager(trainer=trainer, module=module,
                          data=spt.data.DataModule(train=train, val=val),
                          seed=cfg.seed)
    manager()


if __name__ == "__main__":
    run()
