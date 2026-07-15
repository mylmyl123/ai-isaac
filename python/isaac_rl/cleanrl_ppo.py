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

import dataclasses
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

from .spaces import (
    ACTION_FACTORS,
    EGO_CHANNELS,
    EGO_GRID,
    SCALAR_DIM,
    flatten_dict_obs,
    split_obs,
)

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
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = False
    lr_floor_frac: float = 0.1       # Never anneal below lr * lr_floor_frac.

    # ---- Action masking (Phase-1 fix) ----
    # Some stages don't use all 5 action factors. Masking forces the unused
    # factors to a fixed value (0) at sample time so their entropy doesn't
    # bleed into the loss. See swarm-outputs/01-red-team-audit.md.
    mask_unused_action_factors: bool = True

    # ---- Reward shaping (PBRS cold-start fix, 2026-07-14) ----
    # Potential-based reward shaping coefficient. 0.0 = pure 3-term reward
    # (the baseline). >0 densifies the sparse signal via Phi = -dist-to-enemy,
    # provably policy-invariant (Ng 1999). Passed into RewardConfig by train.py
    # along with `gamma` (which MUST match for invariance).
    pbrs_coef: float = 0.0
    # Dense per-hit reward (2026-07-14). >0 rewards every tear that connects
    # with an enemy (not just kills), giving the shoot head a direction-
    # correlated gradient. Scaled by damage-fraction so r_kill stays dominant.
    # 0.0 = off. SUPERSEDED by r_idle (caused park-and-spray); keep off.
    r_hit: float = 0.0
    # Idle penalty (2026-07-15). Flat per-step penalty once >idle_grace ticks
    # since the last kill (counter resets on each kill, frozen while no enemy is
    # present). Makes surviving without killing bleed reward -> fixes the
    # park-and-spray exploit. grace=12 tracks the mod's respawn cadence so
    # loitering is penalized but a prompt killer is not. 0.0 = off.
    r_idle: float = -0.005
    idle_grace: int = 12

    # ---- Closer-spawn curriculum (Phase-2 cold-start fix) ----
    # Enemy spawn-distance band (px from player), passed to the mod via env
    # vars. null/None = mod default (200-500). Set closer (e.g. 90-170) to
    # bootstrap the first accidental kills on a stationary Horf, then relaunch
    # wider to anneal out to the full anti-camp task.
    spawn_min: float | None = None
    spawn_max: float | None = None

    # ---- Network ----
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    # Input normalization (2026-07-15): running mean/std over the obs vector,
    # applied inside the network. Fixes the Tanh saturation that made the
    # policy blind to enemy position. On by default; set false for the ablation.
    normalize_obs: bool = True

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

    Optional action masking: `active_factors` (int) restricts the LOSS to
    the first N factors. The remaining K-N factors are still sampled (so the
    env gets a valid action vector) but forced to 0, and their log_prob /
    entropy are excluded from the loss. Prevents the entropy bonus from
    leaking into useless action heads (drop_bomb / use_pillcard on Stage 0).
    """
    def __init__(self, obs_dim: int, hidden_dim: int, n_layers: int,
                 active_factors: int | None = None, normalize_obs: bool = True):
        super().__init__()
        # ---- Input normalization (2026-07-15) ----
        # The obs is a 2606-dim raw vector; unnormalized it saturates the Tanh
        # trunk (measured: 99% of units pegged, a raw frame-counter swamped the
        # enemy-bearing signal ~7600:1, so the policy was blind to enemy
        # position). Standardize with a running mean/std (Welford, updated only
        # during rollout collection) applied at the top of forward(), clipped to
        # [-10, 10]. As registered buffers this saves/loads with the checkpoint
        # and applies IDENTICALLY in act/evaluate/forward and the offline probe.
        self.normalize_obs = bool(normalize_obs)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

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

        # How many action factors participate in the loss.
        n_all = len(ACTION_FACTORS)
        if active_factors is None or active_factors < 1 or active_factors > n_all:
            active_factors = n_all
        self.active_factors = int(active_factors)
        # Orthogonal init (standard for PPO stability).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        # Smaller gain on the action heads = start close to uniform.
        for head in self.action_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @torch.no_grad()
    def update_norm(self, x: torch.Tensor) -> None:
        """Update running obs mean/var from a batch (Welford / parallel merge).
        Call ONLY during rollout collection, never inside the PPO update epochs
        (so normalization stats aren't skewed by repeated passes over the same
        rollout). No-op when normalize_obs is False."""
        if not self.normalize_obs:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])
        delta = batch_mean - self.obs_mean
        tot = self.obs_count + batch_count
        self.obs_mean += delta * (batch_count / tot)
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.obs_count * batch_count / tot)
        self.obs_var.copy_(m2 / tot)
        self.obs_count.copy_(tot)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_obs:
            return x
        return torch.clamp((x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10.0, 10.0)

    def forward(self, x: torch.Tensor) -> tuple[list[Categorical], torch.Tensor]:
        h = self.trunk(self._normalize(x))
        dists = [Categorical(logits=head(h)) for head in self.action_heads]
        v = self.value_head(h).squeeze(-1)
        return dists, v

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action + return logprob, entropy, value. All shape (B,) or (B, K).
        Masked factors are sampled but forced to 0 and NOT included in logprob/entropy."""
        dists, v = self.forward(x)
        actions = torch.stack([d.sample() for d in dists], dim=-1)   # (B, K)
        # Zero out masked factors so env gets a deterministic "idle" action there.
        if self.active_factors < len(dists):
            actions[:, self.active_factors:] = 0
        active = dists[:self.active_factors]
        logp = torch.stack([d.log_prob(actions[:, k]) for k, d in enumerate(active)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in active], dim=-1).sum(-1)
        return actions, logp, ent, v

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute logprob + entropy + value for GIVEN actions (PPO update).
        Only over ACTIVE factors (must match act())."""
        dists, v = self.forward(x)
        active = dists[:self.active_factors]
        logp = torch.stack([d.log_prob(actions[:, k]) for k, d in enumerate(active)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in active], dim=-1).sum(-1)
        return logp, ent, v


# ==========================================================================
# CNN ACTOR-CRITIC (2026-07-15 architecture rebuild)
# ==========================================================================


class CNNActorCritic(nn.Module):
    """Egocentric-grid CNN + scalar-MLP fusion -> factored action heads + value.

    Replaces the flat-MLP ActorCritic that was measurably BLIND to enemy
    position (constant shoot direction regardless of bearing). Inputs are TWO
    tensors from spaces.split_obs():
      * grid   (B, 14, 21, 21) — egocentric spatial image, fed UNNORMALIZED
        (already in [-1,1]; normalizing sparse presence planes destroys the
        zero baseline the CNN relies on).
      * scalar (B, 161)        — low-D vector, Welford-normalized here.

    ReLU everywhere (escapes the Tanh saturation that caused the blindness),
    orthogonal init gain=sqrt(2) on the ReLU path. Keeps the factored
    MultiDiscrete heads + active_factors masking identical to the MLP version.

    NON-RECURRENT (phase 1). forward(grid, scalar) is stateless so the offline
    bearing probe (single frame) works. The GRU is added in phase 2 only after
    this passes the probe.
    """
    def __init__(self, hidden_dim: int = 256, active_factors: int | None = None,
                 normalize_obs: bool = True):
        super().__init__()
        # Welford normalizer over the SCALAR branch only (grid stays raw).
        self.normalize_obs = bool(normalize_obs)
        self.register_buffer("obs_mean", torch.zeros(SCALAR_DIM))
        self.register_buffer("obs_var", torch.ones(SCALAR_DIM))
        self.register_buffer("obs_count", torch.tensor(1e-4))

        # CNN tower over the 14x21x21 grid.
        self.cnn = nn.Sequential(
            nn.Conv2d(EGO_CHANNELS, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # 21x21
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),            # 11x11
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),            # 6x6
        )
        cnn_flat = 64 * 6 * 6  # 2304
        self.cnn_fc = nn.Sequential(nn.Linear(cnn_flat, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

        # Scalar branch.
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        # Fusion trunk.
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )

        # Separate post-trunk actor / critic heads.
        self.actor_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.critic_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, int(n_choices)) for n_choices in ACTION_FACTORS]
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        # Optional aux head: predict unit-vec-to-nearest-enemy (label from
        # spaces.nearest_enemy_bearing). Forces the CNN to encode bearing; used
        # with a small MSE aux loss. Off unless the trainer wires it.
        self.aux_bearing = nn.Linear(hidden_dim, 2)

        n_all = len(ACTION_FACTORS)
        if active_factors is None or active_factors < 1 or active_factors > n_all:
            active_factors = n_all
        self.active_factors = int(active_factors)

        # Init: gain sqrt(2) on the ReLU path, small on action heads, 1 on value.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for head in self.action_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @torch.no_grad()
    def update_norm(self, scalar: torch.Tensor) -> None:
        """Update running mean/var from a batch of SCALAR vectors (rollout only)."""
        if not self.normalize_obs:
            return
        batch_mean = scalar.mean(dim=0)
        batch_var = scalar.var(dim=0, unbiased=False)
        batch_count = float(scalar.shape[0])
        delta = batch_mean - self.obs_mean
        tot = self.obs_count + batch_count
        self.obs_mean += delta * (batch_count / tot)
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.obs_count * batch_count / tot)
        self.obs_var.copy_(m2 / tot)
        self.obs_count.copy_(tot)

    def _norm_scalar(self, scalar: torch.Tensor) -> torch.Tensor:
        if not self.normalize_obs:
            return scalar
        return torch.clamp((scalar - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10.0, 10.0)

    def _trunk(self, grid: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        g = self.cnn(grid)
        g = self.cnn_fc(g.reshape(g.shape[0], -1))
        s = self.scalar_mlp(self._norm_scalar(scalar))
        return self.fuse(torch.cat([g, s], dim=-1))

    def forward(self, grid: torch.Tensor, scalar: torch.Tensor):
        h = self._trunk(grid, scalar)
        ha = self.actor_mlp(h)
        dists = [Categorical(logits=head(ha)) for head in self.action_heads]
        v = self.value_head(self.critic_mlp(h)).squeeze(-1)
        return dists, v

    def act(self, grid: torch.Tensor, scalar: torch.Tensor):
        dists, v = self.forward(grid, scalar)
        actions = torch.stack([d.sample() for d in dists], dim=-1)
        if self.active_factors < len(dists):
            actions[:, self.active_factors:] = 0
        active = dists[:self.active_factors]
        logp = torch.stack([d.log_prob(actions[:, k]) for k, d in enumerate(active)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in active], dim=-1).sum(-1)
        return actions, logp, ent, v

    def evaluate(self, grid: torch.Tensor, scalar: torch.Tensor, actions: torch.Tensor):
        dists, v = self.forward(grid, scalar)
        active = dists[:self.active_factors]
        logp = torch.stack([d.log_prob(actions[:, k]) for k, d in enumerate(active)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in active], dim=-1).sum(-1)
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
    # Determine how many action factors are active for this stage.
    # Stages 0, A, B use only move + shoot. Stages C, D, E use all 5.
    n_all = len(ACTION_FACTORS)
    if cfg.mask_unused_action_factors and str(cfg.stage) in ("0", "A", "B"):
        active_factors = 2
    else:
        active_factors = n_all
    log.info("action factors: %d active out of %d (stage=%s mask=%s)",
             active_factors, n_all, cfg.stage, cfg.mask_unused_action_factors)

    net = CNNActorCritic(hidden_dim=cfg.hidden_dim, active_factors=active_factors,
                         normalize_obs=cfg.normalize_obs).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    n_factors = len(ACTION_FACTORS)
    # CNN rollout storage: grid (T,N,14,21,21) + scalar (T,N,161) instead of a
    # single flat obs tensor. Reuse the Rollout class for actions/logprobs/
    # rewards/dones/values/GAE (those are obs-agnostic).
    T, N = cfg.rollout_length, cfg.n_envs
    rb = Rollout(T, N, 1, n_factors, device)   # obs_dim=1 placeholder; we don't use rb.obs
    rb_grid = torch.zeros(T, N, EGO_CHANNELS, EGO_GRID, EGO_GRID, device=device)
    rb_scalar = torch.zeros(T, N, SCALAR_DIM, device=device)

    def _split_batch(obs_list):
        """obs dict list -> (grid[N,14,21,21], scalar[N,161]) torch tensors."""
        grids, scalars = [], []
        for o in obs_list:
            g, s = split_obs(o)
            grids.append(g); scalars.append(s)
        return (torch.from_numpy(np.stack(grids)).to(device),
                torch.from_numpy(np.stack(scalars)).to(device))

    # ---- Run dir + TB ----
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.checkpoint_dir) / cfg.run_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    log.info("run dir: %s", run_dir)

    # Save the full config next to the checkpoint. This is what push_data.ps1
    # will find and what makes the run reproducible. Also write the config as
    # a TB 'text' summary so it's visible in the TB UI.
    try:
        import yaml as _yaml
        cfg_dict = dataclasses.asdict(cfg)
        (run_dir / "config.yaml").write_text(_yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
        writer.add_text("config", "```yaml\n" + _yaml.safe_dump(cfg_dict, sort_keys=False) + "\n```", 0)
    except Exception as e:
        log.warning("could not persist config.yaml: %s", e)

    # Log the git commit hash so we always know exactly which code ran.
    try:
        import subprocess as _sp
        sha = _sp.check_output(["git", "rev-parse", "HEAD"], stderr=_sp.DEVNULL).decode().strip()
        (run_dir / "git_sha.txt").write_text(sha + "\n", encoding="utf-8")
        writer.add_text("git_sha", sha, 0)
        log.info("git sha: %s", sha)
    except Exception:
        pass

    # ---- Initial reset ----
    obs_list, _ = env.reset()
    grid, scalar = _split_batch(obs_list)
    dones = torch.zeros(cfg.n_envs, device=device)

    # ---- Episode trackers ----
    # r_kill used to convert episode kill-reward back into a kill COUNT for the
    # kills_mean metric (Phase 2 kill-counting fix). The env is constructed with
    # RewardConfig() defaults in train.py, so read that default here.
    from .reward import RewardConfig as _RewardConfig
    cfg_r_kill = float(_RewardConfig().r_kill)
    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_lens = np.zeros(cfg.n_envs, dtype=np.int64)
    ep_kills = np.zeros(cfg.n_envs, dtype=np.int64)
    ep_deaths = np.zeros(cfg.n_envs, dtype=np.int64)
    completed_ep_rewards: list[float] = []
    completed_ep_lens: list[int] = []
    completed_ep_kills: list[int] = []
    # Phase 2b: running per-episode reward-breakdown accumulator (kill / death /
    # step / pbrs). Lets us LOG each reward component to TB so we can see
    # whether PBRS (or any term) is actually contributing — previously the
    # breakdown was computed but never surfaced, so a too-weak PBRS looked
    # identical to PBRS-off. Keyed by breakdown name -> list of per-episode totals.
    from collections import deque as _deque
    completed_ep_breakdown: dict[str, _deque] = {}

    # Per-episode CSV log — opened once, appended every episode. This gives
    # us variance across episodes, not just moving averages. Small ~500KB
    # for a full 200k-step run.
    episodes_csv = open(run_dir / "episodes.csv", "w", encoding="utf-8", buffering=1)
    episodes_csv.write("step,env_idx,ep_r,ep_len,ep_kills,terminated,truncated\n")

    # Running action histogram: counts per action-factor. Useful for spotting
    # 'policy always picks action 0' failure modes.
    action_hist = [np.zeros(int(n), dtype=np.int64) for n in ACTION_FACTORS]

    global_step = 0
    update = 0
    t_start = time.time()

    while global_step < cfg.total_env_steps:
        update += 1

        # LR anneal with a floor. Setting anneal_lr=False disables entirely.
        # If enabled, LR is annealed linearly from `lr` to `lr * lr_floor_frac`
        # instead of to zero. Prior Stage A run froze the policy at LR=4.8e-7,
        # preventing recovery from any local optimum. Floor at 10% (default)
        # keeps the policy learning throughout the whole budget.
        if cfg.anneal_lr:
            frac = 1.0 - (global_step / cfg.total_env_steps)
            frac = max(frac, cfg.lr_floor_frac)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * frac

        # -------- ROLLOUT --------
        for t in range(cfg.rollout_length):
            rb_grid[t] = grid
            rb_scalar[t] = scalar
            rb.dones[t] = dones

            # Update scalar-branch normalization stats from freshly-collected
            # obs ONLY (not during the PPO update epochs, which reuse this data).
            net.update_norm(scalar)
            with torch.no_grad():
                actions, logp, _, values = net.act(grid, scalar)
            rb.actions[t] = actions
            rb.logprobs[t] = logp
            rb.values[t] = values

            action_np = actions.cpu().numpy().astype(np.int64)
            # Update per-factor action histogram.
            for k in range(action_np.shape[1]):
                counts = np.bincount(action_np[:, k], minlength=int(ACTION_FACTORS[k]))
                action_hist[k] += counts
            next_obs_list, rewards, terms, truncs, infos = env.step(action_np)
            done_np = np.logical_or(terms, truncs).astype(np.float32)

            rb.rewards[t] = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).to(device)
            grid, scalar = _split_batch(next_obs_list)
            dones = torch.from_numpy(done_np).to(device)

            # Episode-return bookkeeping.
            ep_rewards += np.asarray(rewards, dtype=np.float32)
            ep_lens += 1

            for i in range(cfg.n_envs):
                if terms[i] or truncs[i]:
                    # Phase 2 (2026-07-14): count kills from the reward
                    # breakdown, NOT from info["raw"]["events"]. On the death /
                    # mod_restart terminal step, info["raw"] is the NEXT
                    # episode's first frame (env.py returns the reconnected obs),
                    # so per-step event-counting miscounts kills on the terminal
                    # tick. vec_env.py splices reward_breakdown_episode from the
                    # terminal info; kill reward / r_kill is the ground-truth
                    # kill count the agent actually optimized. This makes
                    # kills_mean consistent with the reward signal by
                    # construction, and is immune to the terminal-obs frame swap.
                    bd_ep = infos[i].get("reward_breakdown_episode") or {}
                    r_kill = float(cfg_r_kill)
                    ep_kills[i] = int(round(float(bd_ep.get("kill", 0.0)) / r_kill)) if r_kill else 0
                    # Record each reward-breakdown component (last 20 eps) for TB.
                    for _k, _v in bd_ep.items():
                        dq = completed_ep_breakdown.get(_k)
                        if dq is None:
                            dq = _deque(maxlen=20)
                            completed_ep_breakdown[_k] = dq
                        dq.append(float(_v))
                    completed_ep_rewards.append(float(ep_rewards[i]))
                    completed_ep_lens.append(int(ep_lens[i]))
                    completed_ep_kills.append(int(ep_kills[i]))
                    episodes_csv.write(
                        f"{global_step},{i},{ep_rewards[i]:.4f},{ep_lens[i]},{ep_kills[i]},"
                        f"{int(terms[i])},{int(truncs[i])}\n"
                    )
                    ep_rewards[i] = 0.0
                    ep_lens[i] = 0
                    ep_kills[i] = 0

            global_step += cfg.n_envs

        # -------- COMPUTE GAE --------
        with torch.no_grad():
            _, _, _, next_value = net.act(grid, scalar)
        advantages, returns = rb.compute_gae(next_value, dones, cfg.gamma, cfg.gae_lambda)

        # Flatten (T, N, ...) -> (T*N, ...)
        b_grid = rb_grid.reshape(-1, EGO_CHANNELS, EGO_GRID, EGO_GRID)
        b_scalar = rb_scalar.reshape(-1, SCALAR_DIM)
        b_actions = rb.actions.reshape(-1, n_factors)
        b_logprobs = rb.logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rb.values.reshape(-1)

        # Phase 2 (2026-07-14): normalize advantages ONCE over the full batch,
        # not per-minibatch. Kill events are rare (~1 per ~150 ticks), so most
        # 64-sample minibatches contain 0-1 high-advantage sample; per-minibatch
        # renorm then rescales that lone sample against near-zero neighbors,
        # amplifying noise into artificial advantages. Batch-level norm keeps
        # the advantage scale consistent across all minibatches.
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # -------- PPO UPDATE --------
        batch_size = cfg.rollout_length * cfg.n_envs
        minibatch_size = batch_size // cfg.n_minibatches
        b_inds = np.arange(batch_size)
        pg_losses, v_losses, ent_losses, approx_kls, clipfracs = [], [], [], [], []

        for _epoch in range(cfg.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]

                new_logp, new_ent, new_v = net.evaluate(b_grid[mb], b_scalar[mb], b_actions[mb])
                ratio = torch.exp(new_logp - b_logprobs[mb])
                # Advantages already normalized batch-level above (Phase 2).
                mb_adv = b_advantages[mb]

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
            writer.add_scalar("charts/ep_r_std", float(np.std(completed_ep_rewards[-20:])) if completed_ep_rewards else 0.0, global_step)
            writer.add_scalar("charts/n_completed_episodes", len(completed_ep_rewards), global_step)
            writer.add_scalar("loss/policy", np.mean(pg_losses), global_step)
            writer.add_scalar("loss/value", np.mean(v_losses), global_step)
            writer.add_scalar("loss/entropy", np.mean(ent_losses), global_step)
            writer.add_scalar("loss/approx_kl", np.mean(approx_kls), global_step)
            writer.add_scalar("loss/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], global_step)

            # Per-episode mean of each reward component (kill/death/step/pbrs).
            # Watch reward/pbrs to confirm PBRS is actually contributing, and
            # reward/kill to see the true kill signal magnitude.
            for _k, _dq in completed_ep_breakdown.items():
                if _dq:
                    writer.add_scalar(f"reward/{_k}", float(np.mean(_dq)), global_step)

            # ---- Per-action-factor entropy: which head is collapsing? ----
            # Recompute distribution entropy per factor from the last minibatch's
            # obs so we get a fresh reading (not just the mean over updates).
            with torch.no_grad():
                dists, _ = net.forward(b_grid[:min(256, batch_size)], b_scalar[:min(256, batch_size)])
            factor_names = ["move", "shoot", "use_item", "drop_bomb", "use_pillcard"]
            for k, d in enumerate(dists):
                name = factor_names[k] if k < len(factor_names) else f"factor_{k}"
                writer.add_scalar(f"entropy_per_factor/{name}", float(d.entropy().mean().item()), global_step)

            # ---- Action histogram: is the policy stuck on one action? ----
            for k, hist in enumerate(action_hist):
                total = hist.sum()
                if total > 0:
                    # Most-used action fraction. 1.0 = policy always picks the
                    # same action for this factor (collapsed). Near uniform =
                    # ~1/n_choices (0.11 for 9-way move factor).
                    top_frac = float(hist.max() / total)
                    name = factor_names[k] if k < len(factor_names) else f"factor_{k}"
                    writer.add_scalar(f"action_top_frac/{name}", top_frac, global_step)

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
    episodes_csv.close()
    log.info("training complete: %d steps", global_step)
