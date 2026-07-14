"""CleanRL-style PPO for Isaac RL.

Rewrite following https://github.com/vwxyzjn/cleanrl 's single-file philosophy.
The whole training loop is in this file — no framework abstractions to fight.

Design choices:

  * MLP policy, no LSTM. Simplest thing that could possibly work. Add
    recurrence back if partial observability turns out to be the bottleneck.
  * Factorized MultiDiscrete policy: one Categorical head per action factor
    (move, shoot, use_item, drop_bomb, use_pillcard). Independent factors
    means we can decompose logprob(action) = sum_k logprob(action[k]).
  * Dict obs -> flat vector via spaces.flatten_dict_obs, then a shared MLP
    trunk feeds both policy heads and the value head.
  * Standard PPO-clip loss with GAE-lambda advantages. No tricks.
  * Metrics logged: kill count, death count, episode reward, episode length,
    policy entropy per factor. Enough for the paper's baseline plots.

USAGE

    python train.py --config configs/curriculum.yaml

train.py at repo root is the entry point; it wires the Isaac fleet + this
trainer together.

DEBUGGING

If PPO doesn't learn on Setup A (sealed room, 1 fly), the bug is in this
file, the env, the mod, or the reward. Nothing else. Bisect by:

  1. Try random policy for 10k steps. Confirm kills happen by chance.
  2. Confirm shaper emits r_kill=+1 on those (check the info dict).
  3. Confirm this file's rollout buffer contains those rewards.
  4. Confirm PPO's advantage estimates are non-zero.
  5. If advantages are non-zero and policy doesn't improve — LR too low,
     or PPO clip too tight. Bump both.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from .spaces import ACTION_FACTORS, flatten_dict_obs

log = logging.getLogger(__name__)


# ==========================================================================
# CONFIG
# ==========================================================================


@dataclass
class PPOConfig:
    # ---- Env fleet ----
    n_envs: int = 2
    base_port: int = 9500
    reset_stage: int | None = 1
    max_episode_steps: int = 1800
    total_env_steps: int = 1_000_000

    # ---- Curriculum setup (see mod's ISAAC_RL_STAGE env var) ----
    # 'A' sealed room, 1 fly
    # 'B' sealed room, 3 flies
    # 'C' normal room, 1 fly (unsealed)
    # 'D' normal room, vanilla enemies
    # 'E' full run, no restrictions
    stage: str = "A"

    # ---- PPO hyperparameters ----
    rollout_length: int = 128         # steps per env per update
    n_epochs: int = 4                 # passes over rollout per update
    n_minibatches: int = 4
    lr: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    # ---- Network ----
    hidden_dim: int = 256
    n_hidden_layers: int = 2

    # ---- Runtime ----
    device: str = "cuda"
    seed: int = 42
    run_name: str = "cleanrl_ppo"
    checkpoint_dir: str = "runs"
    checkpoint_every: int = 200_000
    log_every_updates: int = 1


# ==========================================================================
# POLICY NETWORK
# ==========================================================================


def _obs_dim(env) -> int:
    """Get the flat obs dimension by asking the env for a sample obs."""
    obs_list, _ = env.reset()
    flat = flatten_dict_obs(obs_list[0])
    return sum(int(np.prod(v.shape)) for v in flat.values())


def _flat_obs(o: dict[str, Any]) -> np.ndarray:
    """Flatten one dict obs into a 1D float32 array."""
    parts = []
    for k in sorted(flatten_dict_obs(o).keys()):
        v = flatten_dict_obs(o)[k]
        parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


class ActorCritic(nn.Module):
    """Shared trunk -> factorized action heads + value head.

    Action heads: one Linear layer per MultiDiscrete factor. Factor k has
    ACTION_FACTORS[k] logits. Sampling is independent across factors.
    """
    def __init__(self, obs_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        # One logit head per action factor.
        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, int(n_choices)) for n_choices in ACTION_FACTORS]
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        # Orthogonal init (standard for PPO stability).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        # Smaller gain on the action heads = start close to uniform.
        for head in self.action_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> tuple[list[Categorical], torch.Tensor]:
        h = self.trunk(x)
        dists = [Categorical(logits=head(h)) for head in self.action_heads]
        v = self.value_head(h).squeeze(-1)
        return dists, v

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action + return logprob, entropy, value. All shape (B,) or (B, K)."""
        dists, v = self.forward(x)
        actions = torch.stack([d.sample() for d in dists], dim=-1)   # (B, K)
        logp = torch.stack([d.log_prob(actions[:, k]) for k, d in enumerate(dists)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        return actions, logp, ent, v

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute logprob + entropy + value for GIVEN actions (PPO update)."""
        dists, v = self.forward(x)
        logp = torch.stack([d.log_prob(actions[:, k]) for k, d in enumerate(dists)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        return logp, ent, v


# ==========================================================================
# ROLLOUT BUFFER
# ==========================================================================


class Rollout:
    """Pre-allocated buffer for a fixed-length rollout across N envs.

    Shape: (rollout_length, n_envs). Everything on GPU for speed once collected.
    """
    def __init__(self, rollout_len: int, n_envs: int, obs_dim: int, n_factors: int, device: torch.device):
        self.obs = torch.zeros(rollout_len, n_envs, obs_dim, device=device)
        self.actions = torch.zeros(rollout_len, n_envs, n_factors, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(rollout_len, n_envs, device=device)
        self.rewards = torch.zeros(rollout_len, n_envs, device=device)
        self.dones = torch.zeros(rollout_len, n_envs, device=device)
        self.values = torch.zeros(rollout_len, n_envs, device=device)
        self.T = rollout_len

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor,
                    gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (advantages, returns), both shape (T, n_envs)."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros_like(next_value)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                nextnonterm = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterm = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterm - self.values[t]
            last_gae = delta + gamma * gae_lambda * nextnonterm * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns


# ==========================================================================
# TRAIN LOOP
# ==========================================================================


def train(cfg: PPOConfig, env) -> None:
    """Main PPO loop. `env` is a SyncVecEnv (see vec_env.py) that's already
    connected to the Isaac fleet."""

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("device=%s stage=%s n_envs=%d rollout=%d", device, cfg.stage, cfg.n_envs, cfg.rollout_length)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs_dim = _obs_dim(env)
    net = ActorCritic(obs_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    n_factors = len(ACTION_FACTORS)
    rb = Rollout(cfg.rollout_length, cfg.n_envs, obs_dim, n_factors, device)

    # ---- Run dir + TB ----
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.checkpoint_dir) / cfg.run_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    log.info("run dir: %s", run_dir)

    # ---- Initial reset ----
    obs_list, _ = env.reset()
    obs = torch.from_numpy(np.stack([_flat_obs(o) for o in obs_list])).to(device)
    dones = torch.zeros(cfg.n_envs, device=device)

    # ---- Episode trackers ----
    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_lens = np.zeros(cfg.n_envs, dtype=np.int64)
    ep_kills = np.zeros(cfg.n_envs, dtype=np.int64)
    completed_ep_rewards: list[float] = []
    completed_ep_lens: list[int] = []
    completed_ep_kills: list[int] = []

    global_step = 0
    update = 0
    t_start = time.time()

    while global_step < cfg.total_env_steps:
        update += 1

        # Anneal LR linearly to zero.
        if cfg.anneal_lr:
            frac = 1.0 - (global_step / cfg.total_env_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * max(frac, 0.0)

        # -------- ROLLOUT --------
        for t in range(cfg.rollout_length):
            rb.obs[t] = obs
            rb.dones[t] = dones

            with torch.no_grad():
                actions, logp, _, values = net.act(obs)
            rb.actions[t] = actions
            rb.logprobs[t] = logp
            rb.values[t] = values

            action_np = actions.cpu().numpy().astype(np.int64)
            next_obs_list, rewards, terms, truncs, infos = env.step(action_np)
            done_np = np.logical_or(terms, truncs).astype(np.float32)

            rb.rewards[t] = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).to(device)
            obs = torch.from_numpy(np.stack([_flat_obs(o) for o in next_obs_list])).to(device)
            dones = torch.from_numpy(done_np).to(device)

            # Episode-return bookkeeping.
            ep_rewards += np.asarray(rewards, dtype=np.float32)
            ep_lens += 1
            for i, info in enumerate(infos):
                for ev in (info.get("raw") or {}).get("events", []) or []:
                    if ev.get("kind") == "kill":
                        ep_kills[i] += 1

            for i in range(cfg.n_envs):
                if terms[i] or truncs[i]:
                    completed_ep_rewards.append(float(ep_rewards[i]))
                    completed_ep_lens.append(int(ep_lens[i]))
                    completed_ep_kills.append(int(ep_kills[i]))
                    ep_rewards[i] = 0.0
                    ep_lens[i] = 0
                    ep_kills[i] = 0

            global_step += cfg.n_envs

        # -------- COMPUTE GAE --------
        with torch.no_grad():
            _, _, _, next_value = net.act(obs)
        advantages, returns = rb.compute_gae(next_value, dones, cfg.gamma, cfg.gae_lambda)

        # Flatten (T, N, ...) -> (T*N, ...)
        b_obs = rb.obs.reshape(-1, obs_dim)
        b_actions = rb.actions.reshape(-1, n_factors)
        b_logprobs = rb.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rb.values.reshape(-1)

        # -------- PPO UPDATE --------
        batch_size = cfg.rollout_length * cfg.n_envs
        minibatch_size = batch_size // cfg.n_minibatches
        b_inds = np.arange(batch_size)
        pg_losses, v_losses, ent_losses, approx_kls, clipfracs = [], [], [], [], []

        for _epoch in range(cfg.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]

                new_logp, new_ent, new_v = net.evaluate(b_obs[mb], b_actions[mb])
                ratio = torch.exp(new_logp - b_logprobs[mb])
                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss = 0.5 * ((new_v - b_returns[mb]) ** 2).mean()
                ent_loss = new_ent.mean()

                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    log_ratio = new_logp - b_logprobs[mb]
                    approx_kls.append(((torch.exp(log_ratio) - 1) - log_ratio).mean().item())
                    clipfracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())

        # -------- LOG --------
        if update % cfg.log_every_updates == 0:
            elapsed = time.time() - t_start
            sps = global_step / max(elapsed, 1e-6)
            avg_r = np.mean(completed_ep_rewards[-20:]) if completed_ep_rewards else 0.0
            avg_len = np.mean(completed_ep_lens[-20:]) if completed_ep_lens else 0.0
            avg_kills = np.mean(completed_ep_kills[-20:]) if completed_ep_kills else 0.0
            log.info(
                "[step %d/%d %.1f%%] upd=%d sps=%.0f | ep_r=%.2f ep_len=%.0f kills=%.2f | pg=%.4f v=%.4f ent=%.4f",
                global_step, cfg.total_env_steps, 100.0 * global_step / cfg.total_env_steps,
                update, sps, avg_r, avg_len, avg_kills,
                np.mean(pg_losses), np.mean(v_losses), np.mean(ent_losses),
            )
            writer.add_scalar("charts/sps", sps, global_step)
            writer.add_scalar("charts/ep_r_mean", avg_r, global_step)
            writer.add_scalar("charts/ep_len_mean", avg_len, global_step)
            writer.add_scalar("charts/kills_mean", avg_kills, global_step)
            writer.add_scalar("loss/policy", np.mean(pg_losses), global_step)
            writer.add_scalar("loss/value", np.mean(v_losses), global_step)
            writer.add_scalar("loss/entropy", np.mean(ent_losses), global_step)
            writer.add_scalar("loss/approx_kl", np.mean(approx_kls), global_step)
            writer.add_scalar("loss/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], global_step)

        # -------- CHECKPOINT --------
        if global_step - (global_step % cfg.checkpoint_every) >= cfg.checkpoint_every:
            ckpt = {
                "step": global_step,
                "net": net.state_dict(),
                "opt": optimizer.state_dict(),
                "cfg": cfg,
            }
            latest = run_dir / "latest.pt"
            torch.save(ckpt, latest)

    writer.close()
    log.info("training complete: %d steps", global_step)
