"""Single-file recurrent PPO trainer for Isaac RL.

CleanRL-style. Reads a Hydra-flavored config dict, spawns N SocketIsaacEnv workers
via vec_env.py, collects rollouts, computes GAE returns, does K epochs of clipped
policy updates + value regression + RND predictor training.

Run:
    PYTHONPATH=python python -m isaac_rl.ppo --config python/isaac_rl/configs/stage1_single_room.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import yaml
except ImportError:
    yaml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .bc import _load_demos_to_tensors, bc_pretrain
from .curriculum import CurriculumScheduler
from .model import IsaacPolicy, PolicyConfig
from .rnd import RND
from .spaces import ACTION_FACTORS, MAX_ENEMIES, MAX_PROJECTILES, flatten_dict_obs
from .torch_utils import batch_obs_to_tensors, stack_time_batch
from .vec_env import build_vec_env


log = logging.getLogger("ppo")


@dataclass
class PPOConfig:
    # Rollout
    n_envs: int = 4
    rollout_steps: int = 256
    total_env_steps: int = 5_000_000

    # PPO
    n_epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-4
    lr_decay: bool = True
    clip: float = 0.2
    vf_coef: float = 0.5
    # Value-function clipping: analogous to policy clipping. Prevents the
    # value net from taking large single-update steps. Standard PPO trick.
    # Set to 0 to disable (unclipped MSE loss).
    vf_clip: float = 0.2
    # Reward normalization: divide rewards by the running std of returns
    # before feeding into GAE. Stabilises gradients when rewards span
    # multiple orders of magnitude (dense +0.005 vs terminal +50). Standard
    # trick from "What Matters in On-Policy RL" (Andrychowicz 2020).
    normalize_rewards: bool = True
    # Symlog reward transformation (DreamerV3, Hafner 2023).
    # symlog(x) = sign(x) * log(1 + |x|)
    # Compresses wide-magnitude rewards into a narrower range without
    # needing running stats: -50 -> -3.93, -3 -> -1.39, -0.01 -> -0.00995,
    # +2 -> +1.10, +50 -> +3.93. Complements or replaces normalize_rewards
    # (they can compose; symlog first, then RMS on the transformed signal).
    # Set to True to enable. Cheap and stateless.
    symlog_rewards: bool = False
    # Auxiliary supervised losses (UNREAL-style representation learning).
    # For each obs the model predicts (nearest_enemy_dist, enemy_count,
    # nearest_proj_dist) all normalised. MSE loss added to PPO's total loss
    # with coefficient aux_coef. Set to 0 to disable.
    aux_coef: float = 0.1
    # B3: Predict-future-rewards aux task coefficient. Model predicts next
    # N tick rewards from current hidden state. Forces PREDICTIVE
    # representations (not just descriptive). Set to 0 to disable.
    reward_pred_coef: float = 0.05

    # Early-stop the PPO update loop if approx_kl exceeds this threshold.
    # Prevents catastrophic single-update drift. Standard value is 0.015-0.03
    # (Schulman 2017); we use 0.03 as a soft ceiling. Set to 0 to disable.
    target_kl: float = 0.03

    # Learning rate warmup: linearly ramp LR from 0 to cfg.lr over the first
    # N updates. Prevents catastrophic early-update instability, especially
    # important for BC-warm-started runs where the initial policy is highly
    # competent and we don't want to destroy it with huge early gradients.
    # Set to 0 to disable warmup entirely.
    lr_warmup_updates: int = 100

    # B4: Latent variable conditioning (Gaussian per-episode z). z_dim>0
    # enables. Auto-syncs with PolicyConfig.z_dim.

    # B2: Curriculum learning stages. Each stage is a dict with
    # 'until_step' and 'overrides' keys. Applied by CurriculumScheduler.
    # See python/isaac_rl/curriculum.py for supported overrides.
    # Empty list disables curriculum entirely.
    curriculum: list = field(default_factory=list)

    # Weight decay (AdamW instead of Adam). Small L2 regularisation.
    # Prevents overfitting to demo distribution, encourages generalisation.
    # Standard trick from language model / vision transformer training,
    # increasingly common in RL (Cobbe 2020, DreamerV3).
    weight_decay: float = 1e-5
    # Kickstarting (Schmitt et al. 2018): KL divergence penalty toward a
    # frozen BC-pretrained teacher. Prevents catastrophic forgetting of
    # BC behavior during early PPO. Coefficient decays linearly to zero
    # over kickstart_decay_updates PPO updates. Set kickstart_coef=0 to
    # disable.
    kickstart_coef: float = 0.5           # initial KL weight (high enough to matter)
    kickstart_decay_updates: int = 500    # updates over which coef decays to 0
    # Path to a frozen teacher checkpoint (typically bc_pretrained.pt from a
    # previous run). Auto-populated from run_dir if BC ran this session.
    kickstart_teacher_path: str | None = None
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.999
    gae_lambda: float = 0.95

    # RND
    use_rnd: bool = True
    rnd_coef: float = 0.1
    rnd_lr: float = 1e-4

    # Env
    base_port: int = 9500
    reset_stage: int | None = None
    max_episode_steps: int = 27000
    isaac_binary: str | None = None
    launch_isaac: bool = True
    accept_timeout_s: float = 300.0

    # Runtime
    device: str = "cuda"
    seed: int = 42
    run_name: str = "ppo-isaac"
    checkpoint_dir: str = "runs"
    checkpoint_every: int = 500_000
    resume_from: str | None = None   # path to a .pt checkpoint to resume from

    # Heuristic + BC bootstrap (see heuristic.py / bc.py):
    collect_demos_n: int | None = None      # if set, collect this many heuristic demo steps before PPO
    bc_pretrain_file: str | None = None     # if set, load this .npz and BC-pretrain the policy first
    bc_epochs: int = 10                     # BC epochs
    bc_batch_size: int = 256                # BC minibatch size
    bc_lr: float = 3.0e-4                   # BC learning rate

    # Value-function warmup: freeze policy for first N PPO updates while the
    # value function catches up. Critical when using BC pretraining — the
    # pretrained policy is competent but the value network starts random, and
    # updates from an untrained value net produce noisy advantages that
    # destroy the pretrained policy. Set to 0 for random-init runs.
    vf_warmup_updates: int = 0
    log_every: int = 1   # log a progress line every N PPO updates. Default 1 = once per rollout (~17s at n_envs=4).

    # Policy net
    policy: dict = field(default_factory=dict)
    # Reward config (see reward.py RewardConfig). Any field name in RewardConfig can
    # be overridden here to tune reward shaping without editing code.
    reward: dict = field(default_factory=dict)


def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml not installed. `pip install pyyaml`")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _cfg_from_yaml(path: str | None) -> PPOConfig:
    if not path:
        return PPOConfig()
    d = _load_yaml(path)
    # Split policy sub-dict, everything else at top level.
    policy = d.pop("policy", {}) or {}
    return PPOConfig(**d, policy=policy)


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """Vectorized GAE. Shapes: rewards [T,B], values [T,B], dones [T,B], next_value [B]."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            next_v = next_value
        else:
            next_v = values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transformation (DreamerV3 / Hafner 2023).

    symlog(x) = sign(x) * log(1 + |x|)

    Compresses wide-magnitude signals into a narrow, symmetric range while
    preserving sign and monotonicity. Unlike running-std normalisation, it's
    stateless — no dependence on policy-dependent statistics that can drift.

    Reference magnitudes for Isaac rewards:
        -50 -> -3.93   (terminal mom-kill loss)
         -3 -> -1.39   (death)
         -0.1 -> -0.0953 (idle penalty per tick)
         -0.003 -> -0.003 (step penalty; almost identity in [-1, 1])
         +0.001 -> +0.001 (full HP tick)
         +0.5 -> +0.405 (kill)
         +2.0 -> +1.10 (room clear)
         +50 -> +3.93 (mom kill)

    All rewards end up in ~[-4, +4], gradient magnitudes are comparable.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: symexp(x) = sign(x) * (exp(|x|) - 1).

    Not currently used (values are learned in symlog space), but useful for
    diagnostic logging of "true-scale" predicted returns.
    """
    return torch.sign(x) * torch.expm1(torch.abs(x))


def aux_labels_from_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute auxiliary regression labels from a flat obs dict.

    Returns a [B, 3] tensor of (nearest_enemy_norm_dist, enemy_count_frac,
    nearest_proj_norm_dist). All values are in a small numerical range so the
    MSE targets are stable.

    Labels are deterministic functions of the obs — not future information.
    The point is not to "guess unknowns" but to force the trunk to encode
    these summary statistics as first-class features (representation-learning
    regularizer). Cheap and standard technique (Jaderberg et al. 2017).
    """
    # Feature layout (see spaces.py / obs.lua):
    #   feats[..., 2] = dx / 480 (already normalised)
    #   feats[..., 3] = dy / 270
    e_feats = obs["enemies_feats"]        # [B, MAX_ENEMIES, ENEMY_FEATS]
    e_mask = obs["enemies_mask"]          # [B, MAX_ENEMIES], 0/1 float
    p_feats = obs["projectiles_feats"]    # [B, MAX_PROJ, PROJ_FEATS]
    p_mask = obs["projectiles_mask"]      # [B, MAX_PROJ]

    # Nearest normalised distance to any masked entity. For invisible entries,
    # substitute a large value (2.0) so they never win the min. Then clamp
    # the final min to a stable range.
    def _nearest_dist(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dx = feats[..., 2]                # already normalised (dx / 480)
        dy = feats[..., 3]                # (dy / 270)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        # For unmasked (invisible) entries, force a large distance so min
        # ignores them. Use 2.0 (well outside the [0,√2] achievable range).
        big = torch.full_like(dist, 2.0)
        masked = torch.where(mask.bool(), dist, big)
        return masked.min(dim=-1)[0].clamp(min=0.0, max=2.0)

    nearest_enemy = _nearest_dist(e_feats, e_mask)
    nearest_proj = _nearest_dist(p_feats, p_mask)
    enemy_count = e_mask.sum(dim=-1) / float(MAX_ENEMIES)   # in [0, 1]

    return torch.stack([nearest_enemy, enemy_count, nearest_proj], dim=-1)


class RunningMeanStd:
    """Numerically-stable running mean/variance tracker (Welford / parallel algorithm).

    Used for reward normalization: track the running std of returns and divide
    incoming rewards by it before feeding into GAE. Stabilises gradients when
    reward magnitudes span multiple orders (e.g. our dense +0.005 shaping vs
    terminal +50 mom-kill).

    Standard trick from "What Matters in On-Policy RL" (Andrychowicz 2020).
    """
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor | np.ndarray) -> None:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x).reshape(-1)
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = int(x.size)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))


def train(cfg: PPOConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- reward config from YAML overrides -----------------------------
    from .reward import RewardConfig
    reward_cfg = RewardConfig()
    for k, v in (cfg.reward or {}).items():
        if hasattr(reward_cfg, k):
            setattr(reward_cfg, k, v)
            log.info("reward override: %s = %s", k, v)
        else:
            log.warning("unknown reward field in config: %s", k)

    # --- vec env --------------------------------------------------------
    env = build_vec_env(
        n_envs=cfg.n_envs,
        base_port=cfg.base_port,
        reset_stage=cfg.reset_stage,
        max_episode_steps=cfg.max_episode_steps,
        isaac_binary=cfg.isaac_binary,
        launch_isaac=cfg.launch_isaac,
        accept_timeout_s=cfg.accept_timeout_s,
        reward_config=reward_cfg,
    )
    log.info("vec env ready with %d workers", cfg.n_envs)

    # --- policy ---------------------------------------------------------
    policy_cfg = PolicyConfig(**cfg.policy)
    policy = IsaacPolicy(policy_cfg).to(device)
    rnd = RND(feat_dim=policy_cfg.trunk_dim).to(device) if cfg.use_rnd else None
    # Optimizer: AdamW with PPO-tuned eps=1e-5 (Andrychowicz 2020, Engstrom 2020).
    # PyTorch's default eps=1e-8 is too small for RL. AdamW adds decoupled
    # weight decay for mild regularisation (standard in modern RL training).
    optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay)
    rnd_optim = torch.optim.AdamW(rnd.predictor.parameters(), lr=cfg.rnd_lr, eps=1e-5, weight_decay=cfg.weight_decay) if rnd is not None else None

    # --- logging --------------------------------------------------------
    run_dir = Path(cfg.checkpoint_dir) / cfg.run_name / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir) if SummaryWriter is not None else None
    log.info("run dir: %s", run_dir)

    # --- reset ----------------------------------------------------------
    obs_np, infos = env.reset()
    obs_t = batch_obs_to_tensors(obs_np, device)
    hidden = policy.initial_hidden(cfg.n_envs, device)
    dones_t = torch.zeros(cfg.n_envs, device=device)

    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float64)
    ep_lens = np.zeros(cfg.n_envs, dtype=np.int64)
    completed_rewards: list[float] = []
    completed_lens: list[int] = []
    completed_extras: dict[str, list[float]] = {}

    global_step = 0
    updates = 0
    t_start = time.time()

    # Reward normalization (see cfg.normalize_rewards). Reset per-run since
    # the running stats are policy-dependent.
    reward_rms = RunningMeanStd()

    # --- resume from checkpoint if requested ----------------------------
    if getattr(cfg, "resume_from", None):
        ckpt_path = Path(cfg.resume_from).expanduser()
        if not ckpt_path.exists():
            log.warning("resume: checkpoint file %s does not exist, starting fresh", ckpt_path)
        else:
            log.info("resume: loading checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            policy.load_state_dict(ckpt["policy"])
            optim.load_state_dict(ckpt["optim"])
            if rnd is not None and ckpt.get("rnd_predictor") is not None:
                rnd.predictor.load_state_dict(ckpt["rnd_predictor"])
            if rnd is not None and ckpt.get("rnd_target") is not None:
                rnd.target.load_state_dict(ckpt["rnd_target"])
            global_step = int(ckpt.get("global_step", 0))
            log.info("resume: continuing from step %d", global_step)

    # ---- Demo collection (heuristic) ----------------------------------
    # If --collect-demos N was passed, run the heuristic policy for N steps
    # across all envs and save the trajectories to runs/demos/<timestamp>.npz.
    # If --bc-pretrain-file was NOT also passed, we short-circuit and use this
    # freshly-written file for BC. Either way, if only --collect-demos was set
    # (no BC, no PPO), we exit after saving.
    demos_path: Path | None = None
    if getattr(cfg, "collect_demos_n", None):
        from .heuristic import HeuristicPolicy
        from .bc import collect_demos
        ts = time.strftime("%Y%m%d-%H%M%S")
        demos_path = Path(cfg.checkpoint_dir) / "demos" / f"heuristic_{ts}.npz"
        heur = HeuristicPolicy()
        collect_demos(env, heur, int(cfg.collect_demos_n), demos_path)
        # If user didn't also ask for BC pretraining, we're done — exit before PPO.
        if not getattr(cfg, "bc_pretrain_file", None) and cfg.bc_epochs <= 0:
            log.info("demo collection complete; --bc-* not requested. Exiting.")
            env.close()
            return
        # Otherwise chain: use the freshly-collected demos as the BC input.
        if not getattr(cfg, "bc_pretrain_file", None):
            cfg.bc_pretrain_file = str(demos_path)
            log.info("chaining: BC pretrain will use just-collected demos at %s", demos_path)

    # ---- BC pretraining ------------------------------------------------
    if getattr(cfg, "bc_pretrain_file", None):
        from .bc import bc_pretrain
        bc_file = Path(cfg.bc_pretrain_file).expanduser()
        if not bc_file.exists():
            log.warning("BC pretrain file %s does not exist — skipping BC step", bc_file)
        elif cfg.bc_epochs <= 0:
            log.info("bc_epochs=%d — skipping BC step", cfg.bc_epochs)
        else:
            log.info("BC pretraining from %s for %d epochs", bc_file, cfg.bc_epochs)
            bc_pretrain(
                policy, bc_file,
                epochs=cfg.bc_epochs,
                batch_size=cfg.bc_batch_size,
                lr=cfg.bc_lr,
                device=device,
            )
            # Save a post-BC checkpoint so users can resume from a pretrained state
            # without having to re-run BC every time.
            bc_ckpt = run_dir / "bc_pretrained.pt"
            torch.save({
                "policy": policy.state_dict(),
                "cfg": asdict(cfg),
                "global_step": global_step,
            }, bc_ckpt)
            log.info("saved BC-pretrained checkpoint: %s", bc_ckpt)
            # Auto-populate kickstart teacher path from freshly-saved BC weights.
            # User can override via cfg.kickstart_teacher_path in YAML.
            if cfg.kickstart_coef > 0 and cfg.kickstart_teacher_path is None:
                cfg.kickstart_teacher_path = str(bc_ckpt)
                log.info("kickstart teacher auto-set to %s", bc_ckpt)

    # ---- Kickstarting teacher (frozen BC policy) --------------------------
    # Load a frozen copy of the BC policy that PPO will be penalised for
    # diverging from. KL(current || teacher) is added to the loss with
    # coefficient kickstart_coef which decays linearly to 0 over
    # kickstart_decay_updates PPO updates. Prevents catastrophic forgetting
    # of BC behaviour during early PPO. Set kickstart_coef=0 to disable.
    teacher_policy = None
    if cfg.kickstart_coef > 0 and cfg.kickstart_teacher_path:
        teacher_ckpt_path = Path(cfg.kickstart_teacher_path)
        if teacher_ckpt_path.exists():
            teacher_policy = IsaacPolicy(PolicyConfig(**cfg.policy)).to(device)
            teacher_ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
            teacher_policy.load_state_dict(teacher_ckpt["policy"])
            teacher_policy.eval()
            for p in teacher_policy.parameters():
                p.requires_grad_(False)
            log.info("kickstart teacher loaded from %s (coef=%.3f, decay over %d updates)",
                     teacher_ckpt_path, cfg.kickstart_coef, cfg.kickstart_decay_updates)
        else:
            log.warning("kickstart_teacher_path %s does not exist — kickstarting disabled",
                        teacher_ckpt_path)
            cfg.kickstart_coef = 0.0

    # ---- B2: Curriculum scheduler ----------------------------------------
    # Applies stage-based hyperparameter overrides (entropy coef, LR,
    # reward-shaping weights) based on global_step. Empty list = disabled.
    curriculum = CurriculumScheduler(cfg.curriculum)
    if curriculum.stages:
        log.info("curriculum enabled: %d stages", len(curriculum.stages))

    # Helper: save a checkpoint at the current state. Called every
    # cfg.checkpoint_every steps AND on any exit (Ctrl+C, exception, normal
    # completion) via the finally-block below. Also copies to latest.pt for
    # easy --resume usage.
    def _save_ckpt(tag: str) -> None:
        ckpt_path = run_dir / "ckpts" / f"step_{global_step}.pt"
        try:
            torch.save({
                "policy": policy.state_dict(),
                "rnd_predictor": rnd.predictor.state_dict() if rnd is not None else None,
                "rnd_target": rnd.target.state_dict() if rnd is not None else None,
                "optim": optim.state_dict(),
                "cfg": asdict(cfg),
                "global_step": global_step,
            }, ckpt_path)
            # Overwrite latest.pt in the run dir. Trainer users can pass this
            # to --resume without knowing the specific step number.
            latest = run_dir / "latest.pt"
            import shutil as _shutil
            _shutil.copyfile(ckpt_path, latest)
            log.info("[%s] saved checkpoint: %s (also latest.pt)", tag, ckpt_path)
        except Exception as e:
            log.exception("[%s] failed to save checkpoint: %s", tag, e)

    # Signal-based clean shutdown: set a flag on Ctrl+C so the training loop
    # can finish the current rollout gracefully, save a final checkpoint, and
    # exit. Without this, Ctrl+C throws KeyboardInterrupt mid-loop and any
    # progress since the last scheduled checkpoint is lost.
    _shutdown_requested = {"flag": False}
    def _on_sigint(signum, frame):
        if not _shutdown_requested["flag"]:
            log.warning("Ctrl+C received — finishing current rollout then saving. Press Ctrl+C again to force-exit.")
            _shutdown_requested["flag"] = True
        else:
            log.warning("Second Ctrl+C — aborting immediately (progress since last save WILL be lost)")
            raise KeyboardInterrupt()
    try:
        _prev_sigint = signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, AttributeError):
        # Not on main thread or platform doesn't support — fall back to default.
        _prev_sigint = None

    _last_heartbeat = time.time()
    while global_step < cfg.total_env_steps:
        # Heartbeat: print a short activity line if the last log line was more
        # than 30s ago. Long rollouts can otherwise appear frozen — the trainer
        # is silently collecting steps but nothing new prints until the update.
        if time.time() - _last_heartbeat > 30.0:
            sps = global_step / max(1e-6, time.time() - t_start)
            log.info("... collecting rollout (step=%s, sps=%.0f)", f"{global_step:,}", sps)
            _last_heartbeat = time.time()
        # --- collect rollout -------------------------------------------
        rollout_obs: list[dict[str, torch.Tensor]] = []
        rollout_actions: list[torch.Tensor] = []
        rollout_logprobs: list[torch.Tensor] = []
        rollout_values: list[torch.Tensor] = []
        rollout_rewards: list[torch.Tensor] = []
        rollout_dones: list[torch.Tensor] = []
        rollout_int_rewards: list[torch.Tensor] = []
        init_hidden = hidden.detach().clone()

        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                logits, value, hidden = policy.step(obs_t, hidden, done_mask=dones_t)
                action = policy.sample_from_logits(logits)
                logp = policy.log_prob_from_logits(logits, action)

                # RND intrinsic reward on the just-encoded state.
                if rnd is not None:
                    feats = policy.encode(obs_t).detach()
                    int_rew = rnd.intrinsic_reward(feats) * cfg.rnd_coef
                else:
                    int_rew = torch.zeros(cfg.n_envs, device=device)

            rollout_obs.append(obs_t)
            rollout_actions.append(action)
            rollout_logprobs.append(logp)
            rollout_values.append(value)
            rollout_int_rewards.append(int_rew)

            action_np = action.cpu().numpy()
            next_obs_np, rewards_np, terms, truncs, infos = env.step(action_np)
            # IMPORTANT: for GAE we need to distinguish TRUE terminals from
            # rollout-boundary truncations. True terminals (death) get no
            # bootstrap; truncations get bootstrapped from the value network.
            # We track both separately: dones_np = only true terminals.
            # trunc_np = only truncations (episode continues in the game world,
            # we just chose to stop rolling).
            #
            # Standard PPO GAE handling (Andrychowicz 2020, Sutton & Barto):
            #   at truncation: delta = r + gamma * V(s_next) - V(s)  (bootstrap)
            #   at termination: delta = r - V(s)                     (no bootstrap)
            dones_np = np.asarray(terms)               # true terminals only
            trunc_np = np.asarray(truncs)               # truncations only
            reset_np = np.logical_or(dones_np, trunc_np)   # for hidden-state reset

            rewards_t = torch.as_tensor(rewards_np, dtype=torch.float32, device=device) + int_rew
            dones_next = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
            resets_next = torch.as_tensor(reset_np, dtype=torch.float32, device=device)
            rollout_rewards.append(rewards_t)
            rollout_dones.append(dones_next)

            ep_rewards += rewards_np
            ep_lens += 1
            # Handle episode completions for both TRUE terminations (death) and
            # truncations (rollout boundary hit game reset). Both cause the
            # game world to reset, so both count as "episode ended" for
            # logging. Only true terminations are counted as dones for GAE.
            episode_ended = np.logical_or(dones_np, trunc_np)
            for i in range(cfg.n_envs):
                if episode_ended[i]:
                    completed_rewards.append(float(ep_rewards[i]))
                    completed_lens.append(int(ep_lens[i]))
                    ep_rewards[i] = 0.0
                    ep_lens[i] = 0
                    # Log reward breakdown if present.
                    info = infos[i] if i < len(infos) else {}
                    for k, v in (info.get("reward_breakdown") or {}).items():
                        completed_extras.setdefault(k, []).append(float(v))

            obs_t = batch_obs_to_tensors(next_obs_np, device)
            # dones_t is used to reset the GRU hidden state on the NEXT step.
            # It must fire on ANY episode-end (termination OR truncation)
            # because both cause the game world to reset.
            dones_t = resets_next
            global_step += cfg.n_envs

        # --- bootstrap value --------------------------------------------
        with torch.no_grad():
            _, next_value, _ = policy.step(obs_t, hidden, done_mask=dones_t)

        rewards_seq = torch.stack(rollout_rewards, dim=0)              # [T, B]
        values_seq = torch.stack(rollout_values, dim=0)                # [T, B]
        dones_seq = torch.stack(rollout_dones, dim=0)                  # [T, B]

        # Symlog transformation (DreamerV3): applied BEFORE running-std
        # normalization so both can compose. Stateless compression of
        # wide-magnitude rewards; symlog on top of RMS gives a two-stage
        # normalization: (1) compress spikes, (2) rescale to unit variance.
        if cfg.symlog_rewards:
            rewards_seq = symlog(rewards_seq)

        # Reward normalization: track running std of returns (not raw rewards
        # — return-based normalization is the version validated in Andrychowicz
        # 2020) and divide rewards by it before GAE. We compute pseudo-returns
        # by discounting rewards backward, then update the running stats and
        # rescale rewards_seq. Skips the first update where running stats are
        # still at their init.
        if cfg.normalize_rewards:
            # Discounted return per rollout tick (backward pass).
            with torch.no_grad():
                discounted = torch.zeros_like(rewards_seq)
                running = torch.zeros(cfg.n_envs, device=device)
                for t in reversed(range(rewards_seq.shape[0])):
                    running = rewards_seq[t] + cfg.gamma * running * (1.0 - dones_seq[t])
                    discounted[t] = running
                reward_rms.update(discounted)
            rewards_seq = rewards_seq / (reward_rms.std + 1e-8)

        advantages, returns = compute_gae(
            rewards_seq, values_seq, dones_seq, next_value,
            cfg.gamma, cfg.gae_lambda,
        )
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        ret_flat = returns.reshape(-1)
        # Store old values for value-function clipping (see cfg.vf_clip).
        old_values_flat = values_seq.reshape(-1).detach()

        old_logp_flat = torch.stack(rollout_logprobs, dim=0).reshape(-1).detach()
        actions_flat = torch.stack(rollout_actions, dim=0).reshape(-1, len(ACTION_FACTORS))

        # Reassemble sequenced obs by key.
        seq_obs = stack_time_batch(rollout_obs)  # each value [T, B, ...]

        # B2: apply curriculum overrides for the current global_step. This
        # runs BEFORE the LR schedule so curriculum-adjusted cfg.lr is
        # respected by warmup / decay computations.
        if curriculum.stages:
            # Note: reward_overrides are ignored here since we don't have
            # per-env access to reward_shapers through vec_env. Only
            # cfg.ent_coef and cfg.lr are applied. See docs/FUTURE_WORK.md
            # for full environment-curriculum implementation plan.
            stage_changed = curriculum.apply(cfg, reward_shaper=None, global_step=global_step)
            if stage_changed:
                stage = curriculum.current_stage(global_step)
                if stage is not None:
                    log.info("curriculum -> stage until_step=%d overrides=%s",
                             stage.until_step, stage.overrides)
                else:
                    log.info("curriculum -> past all stages, using cfg defaults")

        # LR schedule: warmup + decay. Warmup ramps LR from 0 to cfg.lr
        # linearly over the first lr_warmup_updates PPO updates. This
        # protects BC-pretrained weights from being blown out by the very
        # first PPO update while advantages are still noisy. After warmup,
        # standard linear decay over remaining training kicks in.
        if cfg.lr_warmup_updates > 0 and updates < cfg.lr_warmup_updates:
            warmup_frac = (updates + 1) / cfg.lr_warmup_updates   # 0 -> 1
            effective_lr = cfg.lr * warmup_frac
        elif cfg.lr_decay:
            decay_frac = 1.0 - min(1.0, global_step / cfg.total_env_steps)
            effective_lr = cfg.lr * decay_frac
        else:
            effective_lr = cfg.lr
        for g in optim.param_groups:
            g["lr"] = effective_lr

        # --- PPO epochs -------------------------------------------------
        T = cfg.rollout_steps
        B = cfg.n_envs
        n_samples = T * B
        losses = {"policy": [], "value": [], "entropy": [], "rnd": [], "aux": [], "kickstart": [], "approx_kl": [], "clip_frac": [], "reward_pred": []}

        early_stop = False
        for epoch_i in range(cfg.n_epochs):
            if early_stop:
                break
            # Recompute sequence forward each epoch — recurrent PPO can't shuffle timesteps
            # within an env, but we CAN shuffle across envs and split minibatches env-wise.
            # For a single-GPU workhorse we just do full-batch minibatching env-major.
            env_perm = torch.randperm(B, device=device)
            for start in range(0, B, max(1, cfg.minibatch_size // T)):
                mb_envs = env_perm[start:start + max(1, cfg.minibatch_size // T)]
                if mb_envs.numel() == 0:
                    continue
                mb_seq_obs = {k: v[:, mb_envs] for k, v in seq_obs.items()}
                mb_dones = dones_seq[:, mb_envs]
                mb_rewards = rewards_seq[:, mb_envs]   # for B3 future-reward prediction
                mb_init = init_hidden[mb_envs]

                logits_list, values_new, aux_pred, reward_pred, value_logits = policy.sequence_forward(mb_seq_obs, mb_dones, mb_init)

                # Flatten targets for this minibatch (T, |mb|).
                idx_flat = ((torch.arange(T, device=device)[:, None] * B) + mb_envs[None, :]).reshape(-1)
                mb_old_logp = old_logp_flat[idx_flat]
                mb_adv = adv_flat[idx_flat]
                mb_ret = ret_flat[idx_flat]
                mb_old_values = old_values_flat[idx_flat]
                mb_actions = actions_flat[idx_flat]

                new_logp = policy.log_prob_from_logits(logits_list, mb_actions)
                entropy = policy.entropy_from_logits(logits_list)

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # Diagnostic: approximate KL divergence between old and new
                # policy (Schulman 2020 form: k3 = (r-1) - log(r), unbiased,
                # low-variance estimator). Log this to monitor for policy
                # drift; if approx_kl grows past ~0.02, PPO is likely
                # over-updating (consider lowering LR or clip).
                # Also compute clip fraction (how often the clip actually
                # engaged). If clip fraction is very high (>0.5), the policy
                # is trying to make big jumps and clip is holding it back.
                with torch.no_grad():
                    log_ratio = new_logp - mb_old_logp
                    approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
                # Value function loss. B1: if value_atoms>1, use distributional
                # cross-entropy loss (DreamerV3-style twohot target). Else
                # fall back to scalar MSE (backward-compat).
                if policy.cfg.value_atoms > 1:
                    with torch.no_grad():
                        # Twohot-encode the target returns into a categorical
                        # distribution over value_support atoms.
                        target_dist = policy.value_twohot_target(mb_ret)
                    log_probs = F.log_softmax(value_logits, dim=-1)
                    # Cross-entropy: -sum(target * log_prob).
                    value_loss = -(target_dist * log_probs).sum(-1).mean()
                elif cfg.vf_clip > 0:
                    values_clipped = mb_old_values + torch.clamp(
                        values_new - mb_old_values, -cfg.vf_clip, cfg.vf_clip
                    )
                    v_loss_unclipped = (values_new - mb_ret) ** 2
                    v_loss_clipped = (values_clipped - mb_ret) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(values_new, mb_ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss
                # Auxiliary supervised loss (representation learning).
                aux_loss_val = 0.0
                if cfg.aux_coef > 0:
                    # Compute labels from the flat minibatch obs. mb_seq_obs is
                    # {[T,B,...]}; we need to compute labels on the same flat
                    # ordering the model saw. Reshape into T*B rows.
                    flat_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in mb_seq_obs.items()}
                    with torch.no_grad():
                        aux_targets = aux_labels_from_obs(flat_obs)
                    aux_loss = F.mse_loss(aux_pred, aux_targets)
                    loss = loss + cfg.aux_coef * aux_loss
                    aux_loss_val = float(aux_loss.item())
                # B3: Predict-future-rewards aux task. For each (t, env) in the
                # minibatch, predict the next N tick rewards. Target is built
                # from the rollout's reward sequence with zero-padding at
                # episode boundaries (masked out of the loss).
                reward_pred_loss_val = 0.0
                if cfg.reward_pred_coef > 0:
                    N = policy.cfg.reward_pred_horizon
                    # mb_rewards has shape [T, B_mb]. We need future rewards
                    # for each (t, env) with N-tick lookahead. Also mask
                    # out targets that cross episode boundaries.
                    T_mb, B_mb = mb_rewards.shape
                    fut_targets = torch.zeros(T_mb, B_mb, N, device=device)
                    fut_mask = torch.zeros(T_mb, B_mb, N, device=device)
                    for k in range(1, N + 1):
                        # Shift rewards forward by k; positions past T_mb-1 stay zero.
                        if k < T_mb:
                            fut_targets[:T_mb - k, :, k - 1] = mb_rewards[k:, :]
                            fut_mask[:T_mb - k, :, k - 1] = 1.0
                        # Zero the mask across episode-end (dones) boundaries.
                    # Also mask past dones: any k where an episode ended between
                    # t and t+k should have mask 0. Compute done-cumsum along T.
                    done_cum = torch.zeros(T_mb, B_mb, device=device)
                    for k in range(1, N + 1):
                        if k < T_mb:
                            # Whether any done occurred in (t, t+k].
                            any_done = (mb_dones[1:k+1].sum(0) if k > 0 else torch.zeros_like(mb_dones[0]))
                            # simpler: compute cumulative dones
                    # simpler mask: only positions where the next k steps have no dones
                    cum_done_after = torch.cumsum(mb_dones, dim=0)
                    for k in range(1, N + 1):
                        if k < T_mb:
                            # dones between t+1 and t+k inclusive
                            future_done_count = cum_done_after[k:, :] - cum_done_after[:-k, :]
                            first_done = (future_done_count > 0).float()
                            # If any done in (t, t+k], mask this slot.
                            fut_mask[:T_mb - k, :, k - 1] *= (1.0 - first_done[:T_mb - k, :])
                    fut_targets_flat = fut_targets.reshape(-1, N)
                    fut_mask_flat = fut_mask.reshape(-1, N)
                    # Masked MSE. Denominator uses mask.sum() to normalize
                    # only over valid slots. Add small epsilon.
                    sq_err = (reward_pred - fut_targets_flat) ** 2 * fut_mask_flat
                    reward_pred_loss = sq_err.sum() / (fut_mask_flat.sum() + 1e-8)
                    loss = loss + cfg.reward_pred_coef * reward_pred_loss
                    reward_pred_loss_val = float(reward_pred_loss.item())
                # Kickstarting: KL(current || teacher) added to loss with
                # linearly-decaying coefficient. Prevents BC forgetting.
                kickstart_loss_val = 0.0
                if teacher_policy is not None:
                    with torch.no_grad():
                        t_logits, _, _, _, _ = teacher_policy.sequence_forward(mb_seq_obs, mb_dones, mb_init)
                    # Compute per-head KL and sum across factored heads.
                    kl_total = torch.zeros(logits_list[0].shape[0], device=logits_list[0].device)
                    for i, (student_logits, teacher_logits) in enumerate(zip(logits_list, t_logits)):
                        # KL(student || teacher) = sum_a p_student(a) * (log p_student - log p_teacher)
                        # Standard for policy distillation. We use student-first
                        # (forward KL) because we want the student to stay near
                        # the teacher's action distribution.
                        student_lp = F.log_softmax(student_logits, dim=-1)
                        teacher_lp = F.log_softmax(teacher_logits, dim=-1)
                        student_p = student_lp.exp()
                        kl = (student_p * (student_lp - teacher_lp)).sum(-1)
                        kl_total = kl_total + kl
                    kl_mean = kl_total.mean()
                    # Linear decay of the coefficient from kickstart_coef to 0
                    # over kickstart_decay_updates PPO updates.
                    decay = max(0.0, 1.0 - updates / max(1, cfg.kickstart_decay_updates))
                    effective_coef = cfg.kickstart_coef * decay
                    loss = loss + effective_coef * kl_mean
                    kickstart_loss_val = float(kl_mean.item())
                # Value-function warmup: for the first cfg.vf_warmup_updates PPO
                # updates, ONLY train the value function — zero the policy loss
                # so the pretrained policy weights don't get overwritten by
                # noisy advantages from an untrained value net. Crucial when
                # using BC pretraining, where the policy is competent but the
                # value function starts random.
                if updates < cfg.vf_warmup_updates:
                    loss = cfg.vf_coef * value_loss
                    if cfg.aux_coef > 0 and aux_loss_val > 0:
                        # Keep training the aux head during warmup too
                        # — it helps the trunk features stay useful even
                        # while the policy is frozen.
                        loss = loss + cfg.aux_coef * F.mse_loss(aux_pred, aux_targets)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optim.step()

                losses["policy"].append(float(policy_loss.item()))
                losses["value"].append(float(value_loss.item()))
                if aux_loss_val > 0:
                    losses["aux"].append(aux_loss_val)
                if reward_pred_loss_val > 0:
                    losses["reward_pred"].append(reward_pred_loss_val)
                if kickstart_loss_val > 0:
                    losses["kickstart"].append(kickstart_loss_val)
                losses["entropy"].append(float(-entropy_loss.item()))
                losses["approx_kl"].append(float(approx_kl.item()))
                losses["clip_frac"].append(float(clip_frac.item()))

                # Early-stop the epoch loop if approx_kl exceeds target_kl.
                # Standard PPO safety mechanism to prevent catastrophic drift
                # on a single rollout. Break at the minibatch level so we
                # don't waste compute after the policy has already shifted
                # too far.
                if cfg.target_kl > 0 and float(approx_kl.item()) > cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        # --- RND predictor update ---------------------------------------
        if rnd is not None:
            # Train the predictor on the same minibatch of encoded states from this rollout.
            with torch.no_grad():
                flat_seq_obs = {k: v.reshape(T * B, *v.shape[2:]) for k, v in seq_obs.items()}
            # Do a few gradient steps.
            for _ in range(2):
                idx = torch.randint(0, T * B, (min(1024, T * B),), device=device)
                mb_obs = {k: v[idx] for k, v in flat_seq_obs.items()}
                with torch.no_grad():
                    feats = policy.encode(mb_obs)
                rnd_optim.zero_grad(set_to_none=True)
                rnd_loss = rnd.loss(feats)
                rnd_loss.backward()
                rnd_optim.step()
                losses["rnd"].append(float(rnd_loss.item()))

        updates += 1

        # --- logging ----------------------------------------------------
        if updates % cfg.log_every == 0:
            _last_heartbeat = time.time()   # reset heartbeat since we're logging a full line now
            sps = global_step / max(1e-6, time.time() - t_start)
            recent = completed_rewards[-32:] or [0.0]
            recent_lens = completed_lens[-32:] or [0]
            recent_r = float(np.mean(recent))
            recent_len = float(np.mean(recent_lens))
            # Progress percentage and ETA.
            pct = 100.0 * global_step / max(1, cfg.total_env_steps)
            steps_left = max(0, cfg.total_env_steps - global_step)
            eta_s = steps_left / max(1e-6, sps)
            eta_h, rem = divmod(int(eta_s), 3600)
            eta_m, eta_sec = divmod(rem, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_sec:02d}s"
            # Best reward so far (across the whole run).
            best_r = max(completed_rewards) if completed_rewards else 0.0
            n_eps = len(completed_rewards)
            log.info(
                "[step %s/%s %.1f%%] sps=%.0f ep=%d ep_r=%+.2f (best %+.2f) ep_len=%.0f | pol=%.3f val=%.3f ent=%.3f | ETA %s",
                f"{global_step:,}", f"{cfg.total_env_steps:,}", pct, sps,
                n_eps, recent_r, best_r, recent_len,
                float(np.mean(losses["policy"] or [0])),
                float(np.mean(losses["value"] or [0])),
                float(np.mean(losses["entropy"] or [0])),
                eta_str,
            )
            if writer is not None:
                writer.add_scalar("perf/sps", sps, global_step)
                writer.add_scalar("perf/env_step", global_step, updates)
                writer.add_scalar("rollout/ep_reward", recent_r, global_step)
                writer.add_scalar("rollout/ep_reward_best", best_r, global_step)
                writer.add_scalar("rollout/ep_length", recent_len, global_step)
                writer.add_scalar("rollout/n_episodes", n_eps, global_step)
                for k, vs in losses.items():
                    if vs:
                        writer.add_scalar(f"loss/{k}", float(np.mean(vs)), global_step)
                for k, vs in completed_extras.items():
                    if vs:
                        writer.add_scalar(f"reward/{k}", float(np.mean(vs[-64:])), global_step)

        # --- checkpoint -------------------------------------------------
        if global_step and (global_step // max(1, cfg.checkpoint_every) > (global_step - cfg.n_envs * cfg.rollout_steps) // max(1, cfg.checkpoint_every)):
            _save_ckpt("scheduled")

        # --- clean-shutdown check ----------------------------------------
        if _shutdown_requested["flag"]:
            log.info("clean shutdown requested — saving final checkpoint and exiting training loop")
            _save_ckpt("interrupted")
            break

    # Normal completion (hit total_env_steps) OR clean shutdown OR exhausted loop.
    if not _shutdown_requested["flag"] and global_step >= cfg.total_env_steps:
        _save_ckpt("complete")

    # Restore original SIGINT handler on the way out.
    if _prev_sigint is not None:
        try:
            signal.signal(signal.SIGINT, _prev_sigint)
        except (ValueError, AttributeError):
            pass

    log.info("training complete")
    env.close()
    if writer is not None:
        writer.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()

    cfg = _cfg_from_yaml(args.config)
    for kv in args.override:
        k, _, v = kv.partition("=")
        # Coerce common types.
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        setattr(cfg, k, v)
    log.info("config: %s", cfg)
    train(cfg)


if __name__ == "__main__":
    main()
