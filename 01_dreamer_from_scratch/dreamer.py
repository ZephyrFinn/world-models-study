"""DreamerV2-style world model + latent-imagination actor-critic. Single file.

Components (Phase 1 checklist):
  Encoder            -> MLP (vector obs) or CNN (64x64 pixels)
  RSSM               -> deterministic GRU + categorical stochastic latent (straight-through)
  Decoder            -> reconstruct obs (Gaussian with unit std == MSE)
  Reward model       -> r_hat(s)
  Continue model     -> gamma_hat(s)  (episode-termination head)
  Actor-Critic       -> trained purely inside imagination on lambda-returns
  Imagination rollout-> H-step latent rollout from replayed posterior states

Reference: Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as td


# --------------------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------------------
def mlp(in_dim, hidden, out_dim, layers=2, act=nn.SiLU):
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
        d = hidden
    mods += [nn.Linear(d, out_dim)]
    return nn.Sequential(*mods)


class OneHotDist(td.OneHotCategorical):
    """Categorical latent with straight-through gradients (DreamerV2 Sec. 3)."""

    def mode(self):
        return F.one_hot(self.probs.argmax(-1), self.probs.shape[-1]).to(self.probs.dtype)

    def sample(self, sample_shape=torch.Size(), seed=None):
        sample = super().sample(sample_shape).to(self.probs.dtype)
        # straight-through: value from the hard sample, gradient from the probs
        return sample + (self.probs - self.probs.detach())


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


# --------------------------------------------------------------------------------------
# encoder / decoder
# --------------------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, obs_shape, embed_dim, hidden=256, layers=2):
        super().__init__()
        self.net = mlp(int(np.prod(obs_shape)), hidden, embed_dim, layers)
        self.embed_dim = embed_dim

    def forward(self, obs):  # [..., obs_dim]
        return self.net(obs)


class MLPDecoder(nn.Module):
    def __init__(self, feat_dim, obs_shape, hidden=256, layers=2):
        super().__init__()
        self.shape = obs_shape
        self.net = mlp(feat_dim, hidden, int(np.prod(obs_shape)), layers)

    def forward(self, feat):
        mean = self.net(feat).reshape(*feat.shape[:-1], *self.shape)
        return td.Independent(td.Normal(mean, 1.0), len(self.shape))


class ConvEncoder(nn.Module):
    """DreamerV2 conv encoder: 64x64 -> 1024-d embedding."""

    def __init__(self, obs_shape, depth=32, act=nn.SiLU):
        super().__init__()
        c = obs_shape[0]
        self.net = nn.Sequential(
            nn.Conv2d(c, depth, 4, 2), act(),           # 31
            nn.Conv2d(depth, 2 * depth, 4, 2), act(),   # 14
            nn.Conv2d(2 * depth, 4 * depth, 4, 2), act(),  # 6
            nn.Conv2d(4 * depth, 8 * depth, 4, 2), act(),  # 2
        )
        self.embed_dim = 8 * depth * 2 * 2

    def forward(self, obs):
        lead, img = obs.shape[:-3], obs.shape[-3:]
        x = self.net(obs.reshape(-1, *img))
        return x.reshape(*lead, -1)


class ConvDecoder(nn.Module):
    def __init__(self, feat_dim, obs_shape, depth=32, act=nn.SiLU):
        super().__init__()
        self.shape = obs_shape
        self.depth = depth
        self.fc = nn.Linear(feat_dim, 32 * depth)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, 2), act(),   # 5
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, 2), act(),    # 13
            nn.ConvTranspose2d(2 * depth, depth, 6, 2), act(),        # 30
            nn.ConvTranspose2d(depth, obs_shape[0], 6, 2),            # 64
        )

    def forward(self, feat):
        lead = feat.shape[:-1]
        x = self.fc(feat).reshape(-1, 32 * self.depth, 1, 1)
        mean = self.net(x).reshape(*lead, *self.shape)
        return td.Independent(td.Normal(mean, 1.0), 3)


# --------------------------------------------------------------------------------------
# RSSM
# --------------------------------------------------------------------------------------
class RSSM(nn.Module):
    """Recurrent State-Space Model.

    state = {deter h_t (GRU), stoch z_t (categorical)}
    prior  p(z_t | h_t)            <- imagination path
    post   q(z_t | h_t, e_t)       <- uses the encoded observation
    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    """

    def __init__(self, act_dim, embed_dim, deter=256, stoch=16, classes=16, hidden=256):
        super().__init__()
        self.stoch, self.classes, self.deter = stoch, classes, deter
        self.stoch_dim = stoch * classes
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_dim + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.cell = nn.GRUCell(hidden, deter)
        self.img_out = mlp(deter, hidden, self.stoch_dim, layers=1)
        self.obs_out = mlp(deter + embed_dim, hidden, self.stoch_dim, layers=1)

    @property
    def feat_dim(self):
        return self.stoch_dim + self.deter

    def initial(self, batch, device):
        return dict(
            logit=torch.zeros(batch, self.stoch, self.classes, device=device),
            stoch=torch.zeros(batch, self.stoch, self.classes, device=device),
            deter=torch.zeros(batch, self.deter, device=device),
        )

    def get_feat(self, state):
        stoch = state["stoch"].reshape(*state["stoch"].shape[:-2], self.stoch_dim)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state):
        return td.Independent(OneHotDist(logits=state["logit"]), 1)

    def _stats(self, head, x):
        logit = head(x).reshape(*x.shape[:-1], self.stoch, self.classes)
        return dict(logit=logit)

    def img_step(self, prev_state, prev_action, sample=True):
        """One step of the latent dynamics prior (no observation)."""
        prev_stoch = prev_state["stoch"].reshape(*prev_state["stoch"].shape[:-2], self.stoch_dim)
        x = self.img_in(torch.cat([prev_stoch, prev_action], -1))
        deter = self.cell(x, prev_state["deter"])
        stats = self._stats(self.img_out, deter)
        dist = td.Independent(OneHotDist(logits=stats["logit"]), 1)
        stoch = dist.sample() if sample else dist.mode()
        return dict(logit=stats["logit"], stoch=stoch, deter=deter)

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """Prior step, then correct it with the encoded observation -> posterior."""
        if is_first is not None:
            mask = (1.0 - is_first)[..., None]
            prev_action = prev_action * mask
            init = self.initial(is_first.shape[0], is_first.device)
            prev_state = {
                k: v * mask if v.dim() == 2 else v * mask[..., None]
                for k, v in prev_state.items()
            }
            prev_state = {k: v + init[k] * 0 for k, v in prev_state.items()}
        prior = self.img_step(prev_state, prev_action, sample)
        stats = self._stats(self.obs_out, torch.cat([prior["deter"], embed], -1))
        dist = td.Independent(OneHotDist(logits=stats["logit"]), 1)
        stoch = dist.sample() if sample else dist.mode()
        post = dict(logit=stats["logit"], stoch=stoch, deter=prior["deter"])
        return post, prior

    def observe(self, embed, action, is_first, state=None):
        """Filter a whole batch of sequences. embed/action/is_first are [B, T, ...]."""
        B, T = embed.shape[0], embed.shape[1]
        if state is None:
            state = self.initial(B, embed.device)
        posts, priors = [], []
        for t in range(T):
            state, prior = self.obs_step(state, action[:, t], embed[:, t], is_first[:, t])
            posts.append(state)
            priors.append(prior)
        stack = lambda seq: {k: torch.stack([s[k] for s in seq], 1) for k in seq[0]}
        return stack(posts), stack(priors)

    def imagine(self, state, actor, horizon):
        """H-step latent rollout using the actor. Returns feats/actions/logprobs."""
        state = {k: v.detach() for k, v in state.items()}
        states, actions, logprobs, entropies = [state], [], [], []
        for _ in range(horizon):
            feat = self.get_feat(states[-1])
            dist = actor(feat.detach())
            action = dist.rsample() if hasattr(dist, "rsample_st") else dist.sample()
            logprobs.append(dist.log_prob(action.detach()))
            entropies.append(dist.entropy())
            actions.append(action)
            states.append(self.img_step(states[-1], action))
        feats = torch.stack([self.get_feat(s) for s in states], 0)  # [H+1, B, F]
        return feats, torch.stack(actions, 0), torch.stack(logprobs, 0), torch.stack(entropies, 0)

    def kl_loss(self, post, prior, balance=0.8, free=1.0):
        sg = lambda d: {k: v.detach() for k, v in d.items()}
        lhs = td.kl_divergence(self.get_dist(sg(post)), self.get_dist(prior))
        rhs = td.kl_divergence(self.get_dist(post), self.get_dist(sg(prior)))
        loss = balance * lhs.mean().clamp(min=free) + (1 - balance) * rhs.mean().clamp(min=free)
        return loss, rhs.mean().detach()


# --------------------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------------------
class ScalarHead(nn.Module):
    def __init__(self, feat_dim, hidden=256, layers=2, use_symlog=False):
        super().__init__()
        self.net = mlp(feat_dim, hidden, 1, layers)
        self.use_symlog = use_symlog

    def forward(self, feat):
        return self.net(feat).squeeze(-1)

    def dist(self, feat):
        return td.Normal(self(feat), 1.0)

    def pred(self, feat):
        out = self(feat)
        return symexp(out) if self.use_symlog else out


class BernoulliHead(nn.Module):
    def __init__(self, feat_dim, hidden=256, layers=2):
        super().__init__()
        self.net = mlp(feat_dim, hidden, 1, layers)

    def dist(self, feat):
        return td.Bernoulli(logits=self.net(feat).squeeze(-1))


class Actor(nn.Module):
    def __init__(self, feat_dim, act_dim, hidden=256, layers=3):
        super().__init__()
        self.net = mlp(feat_dim, hidden, act_dim, layers)

    def forward(self, feat):
        return OneHotDist(logits=self.net(feat))


# --------------------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------------------
class SequenceReplay:
    """Stores whole episodes; samples fixed-length chunks (DreamerV2 style)."""

    def __init__(self, capacity_steps, obs_shape, act_dim, seq_len):
        self.capacity = capacity_steps
        self.seq_len = seq_len
        self.obs_shape, self.act_dim = obs_shape, act_dim
        self.episodes = []
        self.n_steps = 0

    def add_episode(self, ep):
        ep = {k: np.asarray(v) for k, v in ep.items()}
        self.episodes.append(ep)
        self.n_steps += len(ep["reward"])
        while self.n_steps > self.capacity and len(self.episodes) > 1:
            self.n_steps -= len(self.episodes.pop(0)["reward"])

    def sample(self, batch_size, rng):
        lens = np.array([len(e["reward"]) for e in self.episodes], dtype=np.float64)
        probs = lens / lens.sum()
        out = defaultdict(list)
        for _ in range(batch_size):
            ei = rng.choice(len(self.episodes), p=probs)
            ep = self.episodes[ei]
            T = len(ep["reward"])
            if T <= self.seq_len:
                idx = np.arange(T)
                pad = self.seq_len - T
            else:
                start = rng.integers(0, T - self.seq_len + 1)
                idx = np.arange(start, start + self.seq_len)
                pad = 0
            for k, v in ep.items():
                chunk = v[idx]
                if pad:  # repeat last frame; weight=0 masks it out of the loss
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, 0)], 0)
                out[k].append(chunk)
            w = np.ones(self.seq_len, dtype=np.float32)
            if pad:
                w[T:] = 0.0
            out["weight"].append(w)
        return {k: np.stack(v, 0) for k, v in out.items()}


# --------------------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------------------
class Dreamer(nn.Module):
    def __init__(self, obs_shape, act_dim, cfg, device):
        super().__init__()
        self.cfg, self.device, self.act_dim = cfg, device, act_dim
        pixels = len(obs_shape) == 3
        self.encoder = (ConvEncoder(obs_shape, cfg.cnn_depth) if pixels
                        else MLPEncoder(obs_shape, cfg.embed_dim, cfg.hidden))
        self.rssm = RSSM(act_dim, self.encoder.embed_dim, cfg.deter, cfg.stoch,
                         cfg.classes, cfg.hidden)
        F_ = self.rssm.feat_dim
        self.decoder = (ConvDecoder(F_, obs_shape, cfg.cnn_depth) if pixels
                        else MLPDecoder(F_, obs_shape, cfg.hidden))
        self.reward_head = ScalarHead(F_, cfg.hidden, use_symlog=cfg.symlog)
        self.cont_head = BernoulliHead(F_, cfg.hidden)
        self.actor = Actor(F_, act_dim, cfg.hidden)
        self.critic = ScalarHead(F_, cfg.hidden, use_symlog=cfg.symlog)
        self.target_critic = ScalarHead(F_, cfg.hidden, use_symlog=cfg.symlog)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.to(device)

        wm_params = list(self.encoder.parameters()) + list(self.rssm.parameters()) + \
            list(self.decoder.parameters()) + list(self.reward_head.parameters()) + \
            list(self.cont_head.parameters())
        opt = lambda ps, lr: torch.optim.Adam(ps, lr=lr, eps=1e-5, weight_decay=1e-6)
        self.wm_opt = opt(wm_params, cfg.model_lr)
        self.actor_opt = opt(self.actor.parameters(), cfg.actor_lr)
        self.critic_opt = opt(self.critic.parameters(), cfg.critic_lr)
        self.wm_params = wm_params
        self.updates = 0

    # ---------------- acting ----------------
    @torch.no_grad()
    def policy(self, obs, prev_state, prev_action, is_first, mode="train"):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None]
        is_first = torch.as_tensor([float(is_first)], device=self.device)
        embed = self.encoder(obs)
        state, _ = self.rssm.obs_step(prev_state, prev_action, embed, is_first)
        feat = self.rssm.get_feat(state)
        dist = self.actor(feat)
        if mode == "eval":
            action = dist.mode()
        else:
            action = dist.sample()
            if self.cfg.expl_eps > 0 and np.random.rand() < self.cfg.expl_eps:
                action = F.one_hot(torch.randint(self.act_dim, (1,), device=self.device),
                                   self.act_dim).float()
        return state, action

    # ---------------- world model ----------------
    def train_world_model(self, data):
        cfg = self.cfg
        obs, action = data["obs"], data["action"]
        reward, is_first, is_term, weight = (data["reward"], data["is_first"],
                                             data["is_terminal"], data["weight"])
        embed = self.encoder(obs)
        post, prior = self.rssm.observe(embed, action, is_first)
        feat = self.rssm.get_feat(post)

        recon = -self.decoder(feat).log_prob(obs)
        rew_target = symlog(reward) if cfg.symlog else reward
        rew_loss = -self.reward_head.dist(feat).log_prob(rew_target)
        cont_loss = -self.cont_head.dist(feat).log_prob(1.0 - is_term)
        kl, kl_val = self.rssm.kl_loss(post, prior, cfg.kl_balance, cfg.kl_free)

        w = weight
        loss = ((recon + rew_loss + cont_loss) * w).mean() / w.mean().clamp(min=1e-8) \
            + cfg.kl_scale * kl

        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.wm_params, cfg.grad_clip)
        self.wm_opt.step()

        metrics = dict(wm_loss=loss.item(), recon=recon.mean().item(),
                       rew_loss=rew_loss.mean().item(), cont_loss=cont_loss.mean().item(),
                       kl=kl_val.item(), wm_gradnorm=float(gn))
        return {k: v.detach() for k, v in post.items()}, metrics

    # ---------------- behaviour in imagination ----------------
    def train_actor_critic(self, post, weight):
        cfg = self.cfg
        # flatten [B, T] posterior states into a batch of imagination start states
        start = {k: v.reshape(-1, *v.shape[2:]) for k, v in post.items()}
        feats, actions, logprobs, entropies = self.rssm.imagine(start, self.actor, cfg.horizon)

        reward = self.reward_head.pred(feats)                       # [H+1, B]
        cont = self.cont_head.dist(feats).mean                      # [H+1, B]
        disc = cfg.discount * cont
        with torch.no_grad():
            value_t = self.target_critic.pred(feats)                # [H+1, B]

        target = lambda_return(reward[1:], value_t[1:], disc[1:], cfg.lam)   # [H, B]
        # discount weights: state s_0 always valid, later states weighted by survival prob
        w = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[1:cfg.horizon]], 0), 0)
        w = w.detach()

        # ---- actor ----
        adv = (target - value_t[:-1]).detach()
        if cfg.actor_grad == "dynamics":
            actor_obj = target                                       # backprop through dynamics
        elif cfg.actor_grad == "reinforce":
            actor_obj = logprobs * adv
        else:  # both (DreamerV2 mix)
            actor_obj = cfg.actor_mix * logprobs * adv + (1 - cfg.actor_mix) * target
        actor_loss = -((actor_obj + cfg.actor_ent * entropies) * w).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        a_gn = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()

        # ---- critic ----
        value_dist = self.critic.dist(feats[:-1].detach())
        v_target = symlog(target.detach()) if cfg.symlog else target.detach()
        critic_loss = -(value_dist.log_prob(v_target) * w).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        c_gn = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        self.updates += 1
        if self.updates % cfg.target_every == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

        return dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(),
                    imag_return=target.mean().item(), imag_reward=reward.mean().item(),
                    imag_value=value_t.mean().item(), entropy=entropies.mean().item(),
                    actor_gradnorm=float(a_gn), critic_gradnorm=float(c_gn))

    def train_step(self, data):
        post, m1 = self.train_world_model(data)
        m2 = self.train_actor_critic(post, data["weight"])
        return {**m1, **m2}


def lambda_return(reward, value, disc, lam):
    """V^lambda_t = r_{t+1} + g_{t+1} [ (1-lam) v_{t+1} + lam V^lambda_{t+1} ],  V^lambda_H = v_H.

    Inputs are aligned as reward[t]=r_{t+1}, value[t]=v_{t+1}, disc[t]=gamma_{t+1}, shape [H, B].
    Returns V^lambda_0..V^lambda_{H-1}, aligned with imagination states s_0..s_{H-1}.
    """
    H = reward.shape[0]
    last = value[-1]
    outs = []
    for t in reversed(range(H)):
        last = reward[t] + disc[t] * ((1 - lam) * value[t] + lam * last)
        outs.append(last)
    return torch.stack(list(reversed(outs)), 0)


# --------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------
class PixelWrapper(gym.Wrapper):
    """Render to 64x64 CHW float in [-0.5, 0.5] (DreamerV2 image preprocessing)."""

    def __init__(self, env, size=64):
        super().__init__(env)
        from PIL import Image
        self.Image = Image
        self.size = size
        self.observation_space = gym.spaces.Box(-0.5, 0.5, (3, size, size), np.float32)

    def _obs(self):
        frame = self.env.render()
        img = self.Image.fromarray(frame).resize((self.size, self.size), self.Image.BILINEAR)
        return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0 - 0.5

    def reset(self, **kw):
        _, info = self.env.reset(**kw)
        return self._obs(), info

    def step(self, a):
        _, r, term, trunc, info = self.env.step(a)
        return self._obs(), r, term, trunc, info


def make_env(name, pixels, seed):
    env = gym.make(name, render_mode="rgb_array" if pixels else None)
    if pixels:
        env = PixelWrapper(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# --------------------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------------------
def evaluate(agent, env, episodes, act_dim, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        state = agent.rssm.initial(1, device)
        action = torch.zeros(1, act_dim, device=device)
        is_first, done, ret = True, False, 0.0
        while not done:
            state, action = agent.policy(obs, state, action, is_first, mode="eval")
            obs, r, term, trunc, _ = env.step(int(action[0].argmax().item()))
            ret += r
            is_first, done = False, term or trunc
        returns.append(ret)
    return float(np.mean(returns)), float(np.std(returns))


def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    env = make_env(cfg.env, cfg.pixels, cfg.seed)
    eval_env = make_env(cfg.env, cfg.pixels, cfg.seed + 10_000)
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n

    agent = Dreamer(obs_shape, act_dim, cfg, device)
    replay = SequenceReplay(cfg.replay_capacity, obs_shape, act_dim, cfg.seq_len)

    outdir = pathlib.Path(cfg.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(cfg), open(outdir / "config.json", "w"), indent=2)
    csv_f = open(outdir / "metrics.csv", "w", newline="")
    writer = csv.writer(csv_f)
    header_written = False

    def new_ep(obs):
        return dict(obs=[obs.astype(np.float32)], action=[np.zeros(act_dim, np.float32)],
                    reward=[0.0], is_first=[1.0], is_terminal=[0.0])

    obs, _ = env.reset()
    ep = new_ep(obs)
    state = agent.rssm.initial(1, device)
    prev_action = torch.zeros(1, act_dim, device=device)
    is_first = True
    env_steps, ep_return, train_returns = 0, 0.0, []
    t0 = time.time()
    metrics_acc = defaultdict(list)

    while env_steps < cfg.total_steps:
        # ---------------- act ----------------
        if env_steps < cfg.prefill:
            a = env.action_space.sample()
            action = F.one_hot(torch.tensor([a], device=device), act_dim).float()
        else:
            state, action = agent.policy(obs, state, prev_action, is_first, mode="train")
            a = int(action[0].argmax().item())
        obs, r, term, trunc, _ = env.step(a)
        env_steps += 1
        ep_return += r
        prev_action, is_first = action, False
        ep["obs"].append(obs.astype(np.float32))
        ep["action"].append(action[0].detach().cpu().numpy())
        ep["reward"].append(float(r))
        ep["is_first"].append(0.0)
        ep["is_terminal"].append(float(term))  # truncation is NOT a terminal state

        if term or trunc:
            replay.add_episode(ep)
            train_returns.append(ep_return)
            obs, _ = env.reset()
            ep = new_ep(obs)
            state = agent.rssm.initial(1, device)
            prev_action = torch.zeros(1, act_dim, device=device)
            is_first, ep_return = True, 0.0

        # ---------------- learn ----------------
        if env_steps >= cfg.prefill and env_steps % cfg.train_every == 0:
            for _ in range(cfg.train_steps):
                batch = replay.sample(cfg.batch_size, rng)
                data = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                        for k, v in batch.items()}
                m = agent.train_step(data)
                for k, v in m.items():
                    metrics_acc[k].append(v)

        # ---------------- log / eval ----------------
        if env_steps % cfg.eval_every == 0:
            ev_mean, ev_std = evaluate(agent, eval_env, cfg.eval_episodes, act_dim, device)
            recent = train_returns[-10:] if train_returns else [0.0]
            row = dict(env_steps=env_steps, grad_steps=agent.updates,
                       eval_return=ev_mean, eval_std=ev_std,
                       train_return=float(np.mean(recent)),
                       episodes=len(train_returns), wall_time=time.time() - t0,
                       **{k: float(np.mean(v)) for k, v in metrics_acc.items()})
            metrics_acc.clear()
            if not header_written:
                writer.writerow(list(row)), csv_f.flush()
                header_written = True
            writer.writerow([row[k] for k in row])
            csv_f.flush()
            print(f"[{cfg.tag}] steps={env_steps:6d} grad={agent.updates:6d} "
                  f"eval={ev_mean:6.1f}+-{ev_std:5.1f} train={row['train_return']:6.1f} "
                  f"kl={row.get('kl', 0):.2f} imagR={row.get('imag_return', 0):6.2f} "
                  f"ent={row.get('entropy', 0):.3f} t={row['wall_time']:.0f}s", flush=True)

    csv_f.close()
    torch.save(agent.state_dict(), outdir / "agent.pt")
    print(f"[{cfg.tag}] done in {time.time() - t0:.0f}s", flush=True)


def get_config(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--pixels", action="store_true")
    p.add_argument("--logdir", default="runs/debug")
    p.add_argument("--tag", default="run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    # world model
    p.add_argument("--deter", type=int, default=256)
    p.add_argument("--stoch", type=int, default=16)
    p.add_argument("--classes", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--cnn_depth", type=int, default=32)
    p.add_argument("--kl_scale", type=float, default=1.0)
    p.add_argument("--kl_balance", type=float, default=0.8)
    p.add_argument("--kl_free", type=float, default=1.0)
    p.add_argument("--symlog", action="store_true")
    # behaviour
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--actor_grad", default="dynamics", choices=["dynamics", "reinforce", "both"])
    p.add_argument("--actor_mix", type=float, default=1.0)
    p.add_argument("--actor_ent", type=float, default=3e-3)
    p.add_argument("--expl_eps", type=float, default=0.0)
    # optimisation
    p.add_argument("--model_lr", type=float, default=3e-4)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=100.0)
    p.add_argument("--target_every", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--train_every", type=int, default=5)
    p.add_argument("--train_steps", type=int, default=1)
    p.add_argument("--prefill", type=int, default=1000)
    p.add_argument("--replay_capacity", type=int, default=200_000)
    p.add_argument("--total_steps", type=int, default=60_000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_episodes", type=int, default=5)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(get_config())
