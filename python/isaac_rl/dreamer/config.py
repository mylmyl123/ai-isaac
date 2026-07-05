"""Dreamer training config.

Mirrors PPOConfig where fields are shared (env, curriculum, reward). The
world-model / imagination fields track DreamerV3 paper defaults as
reproduced in NM512/dreamerv3-torch.

Loaded from YAML by ``_cfg_from_yaml`` at the bottom of this file (same
pattern as ``ppo.py::_cfg_from_yaml``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DreamerConfig:
    # ---- Env (mirrors PPOConfig; must be filled from YAML) ------------
    n_envs: int = 4
    base_port: int = 9500
    reset_stage: int | None = None
    max_episode_steps: int = 27000
    isaac_binary: str | None = None
    launch_isaac: bool = False   # top-level train.py owns the fleet
    accept_timeout_s: float = 300.0

    # ---- Runtime ------------------------------------------------------
    device: str = "cuda"
    seed: int = 42
    run_name: str = "dreamer-isaac"
    checkpoint_dir: str = "runs"
    checkpoint_every: int = 500_000
    total_env_steps: int = 5_000_000
    resume_from: str | None = None
    log_every: int = 100        # updates between TB writes

    # ---- Encoder / decoder --------------------------------------------
    encoder_embed_dim: int = 1024
    encoder_trunk_dim: int = 768
    decoder_hidden: int = 512
    decoder_layers: int = 2

    # ---- RSSM (DreamerV3 paper defaults; NM512 confirmed) --------------
    rssm_stoch: int = 32          # 32 categoricals ...
    rssm_discrete: int = 32       # ... of 32 classes each  (=> stoch dim = 1024)
    rssm_deter: int = 512
    rssm_hidden: int = 512
    rssm_rec_depth: int = 1
    rssm_act: str = "SiLU"
    rssm_norm: bool = True
    rssm_mean_act: str = "none"
    rssm_std_act: str = "softplus"
    rssm_min_std: float = 0.1
    rssm_unimix_ratio: float = 0.01
    rssm_initial: str = "learned"

    # ---- World-model training ------------------------------------------
    world_model_lr: float = 1e-4
    world_model_eps: float = 1e-8
    world_model_grad_clip: float = 1000.0
    weight_decay: float = 0.0
    batch_size: int = 16
    seq_len: int = 64
    kl_free_bits: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1
    # Loss scales per head (paper defaults).
    reward_loss_scale: float = 1.0
    cont_loss_scale: float = 1.0

    # ---- Behavior (actor + critic in imagination) ----------------------
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    actor_eps: float = 1e-5
    critic_eps: float = 1e-5
    actor_grad_clip: float = 100.0
    critic_grad_clip: float = 100.0
    actor_entropy: float = 3e-4
    actor_hidden: int = 512
    actor_layers: int = 2
    critic_hidden: int = 512
    critic_layers: int = 2
    imag_horizon: int = 15
    gamma: float = 0.997
    gae_lambda: float = 0.95
    unimix_ratio: float = 0.01

    # ---- Value + reward heads -----------------------------------------
    value_atoms: int = 255        # DiscDist bins (symlog space)
    value_v_min: float = -20.0
    value_v_max: float = 20.0
    reward_head_layers: int = 2
    cont_head_layers: int = 2

    # ---- Slow critic target (paper: EMA every N updates) ---------------
    slow_target: bool = True
    slow_target_update: int = 1
    slow_target_fraction: float = 0.02
    reward_ema: bool = True

    # ---- Training loop ------------------------------------------------
    prefill_steps: int = 2500      # random-policy warmup before first WM update
    train_ratio: int = 16          # WM gradient steps per env step, per env
    replay_capacity: int = 1_000_000

    # ---- Curriculum + reward (same as PPOConfig) -----------------------
    curriculum: list = field(default_factory=list)
    reward: dict = field(default_factory=dict)


def cfg_from_yaml(path: str | None) -> DreamerConfig:
    """Load a DreamerConfig from a YAML file. Unknown keys raise (fail-loud)."""
    if not path:
        return DreamerConfig()
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Split known nested subdicts so DreamerConfig(**data) accepts them.
    known_nested = {"curriculum": [], "reward": {}}
    for k, default in known_nested.items():
        data.setdefault(k, default)
    return DreamerConfig(**data)
