"""Model-free DQN baseline on the same env/budget, for the sample-efficiency comparison.

Deliberately a strong-but-standard DQN (double Q, target net, eps-greedy) so the
comparison isolates "learning in imagination" vs "learning from replayed transitions".
"""
from __future__ import annotations

import argparse, csv, json, pathlib, random, time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


def evaluate(q, env, episodes, device):
    rets = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs, dtype=torch.float32, device=device)).argmax())
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets)), float(np.std(rets))


def main(cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = torch.device(cfg.device)
    env = gym.make(cfg.env); env.reset(seed=cfg.seed); env.action_space.seed(cfg.seed)
    eval_env = gym.make(cfg.env); eval_env.reset(seed=cfg.seed + 10_000)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n

    q = mlp(obs_dim, 256, act_dim).to(device)
    qt = mlp(obs_dim, 256, act_dim).to(device); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf = deque(maxlen=cfg.capacity)

    outdir = pathlib.Path(cfg.logdir); outdir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(cfg), open(outdir / "config.json", "w"), indent=2)
    f = open(outdir / "metrics.csv", "w", newline=""); w = csv.writer(f)
    w.writerow(["env_steps", "grad_steps", "eval_return", "eval_std", "train_return",
                "episodes", "wall_time"]); f.flush()

    obs, _ = env.reset()
    ret, rets, grad_steps, t0 = 0.0, [], 0, time.time()
    for step in range(1, cfg.total_steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - step / cfg.eps_decay)
        if step < cfg.prefill or random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs, dtype=torch.float32, device=device)).argmax())
        nobs, r, term, trunc, _ = env.step(a)
        buf.append((obs, a, r, nobs, float(term)))
        obs, ret = nobs, ret + r
        if term or trunc:
            rets.append(ret); obs, _ = env.reset(); ret = 0.0

        if step >= cfg.prefill and step % cfg.train_every == 0:
            batch = random.sample(buf, cfg.batch_size)
            o, ac, rw, no, dn = map(np.array, zip(*batch))
            o = torch.as_tensor(o, dtype=torch.float32, device=device)
            no = torch.as_tensor(no, dtype=torch.float32, device=device)
            ac = torch.as_tensor(ac, dtype=torch.int64, device=device)
            rw = torch.as_tensor(rw, dtype=torch.float32, device=device)
            dn = torch.as_tensor(dn, dtype=torch.float32, device=device)
            with torch.no_grad():
                na = q(no).argmax(-1)                                   # double DQN
                tgt = rw + cfg.discount * (1 - dn) * qt(no).gather(1, na[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q(o).gather(1, ac[:, None]).squeeze(1), tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            grad_steps += 1
            if grad_steps % cfg.target_every == 0:
                qt.load_state_dict(q.state_dict())

        if step % cfg.eval_every == 0:
            m, s = evaluate(q, eval_env, cfg.eval_episodes, device)
            w.writerow([step, grad_steps, m, s, float(np.mean(rets[-10:])) if rets else 0.0,
                        len(rets), time.time() - t0]); f.flush()
            print(f"[{cfg.tag}] steps={step:6d} eval={m:6.1f}+-{s:5.1f} eps={eps:.2f}", flush=True)
    f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--logdir", default="runs/dqn"); p.add_argument("--tag", default="dqn")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--capacity", type=int, default=100_000)
    p.add_argument("--prefill", type=int, default=1000)
    p.add_argument("--train_every", type=int, default=1)
    p.add_argument("--target_every", type=int, default=500)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=float, default=10_000)
    p.add_argument("--total_steps", type=int, default=60_000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_episodes", type=int, default=5)
    main(p.parse_args())
