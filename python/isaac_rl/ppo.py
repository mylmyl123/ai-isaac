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

from .model import IsaacPolicy, PolicyConfig
from .rnd import RND
from .spaces import ACTION_FACTORS, flatten_dict_obs
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
    log_every: int = 1   # log a progress line every N PPO updates. Default 1 = once per rollout (~17s at n_envs=4).

    # Policy net
    policy: dict = field(default_factory=dict)


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


def train(cfg: PPOConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- vec env --------------------------------------------------------
    env = build_vec_env(
        n_envs=cfg.n_envs,
        base_port=cfg.base_port,
        reset_stage=cfg.reset_stage,
        max_episode_steps=cfg.max_episode_steps,
        isaac_binary=cfg.isaac_binary,
        launch_isaac=cfg.launch_isaac,
        accept_timeout_s=cfg.accept_timeout_s,
    )
    log.info("vec env ready with %d workers", cfg.n_envs)

    # --- policy ---------------------------------------------------------
    policy_cfg = PolicyConfig(**cfg.policy)
    policy = IsaacPolicy(policy_cfg).to(device)
    rnd = RND(feat_dim=policy_cfg.trunk_dim).to(device) if cfg.use_rnd else None
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    rnd_optim = torch.optim.Adam(rnd.predictor.parameters(), lr=cfg.rnd_lr) if rnd is not None else None

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
            dones_np = np.logical_or(terms, truncs)

            rewards_t = torch.as_tensor(rewards_np, dtype=torch.float32, device=device) + int_rew
            dones_next = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
            rollout_rewards.append(rewards_t)
            rollout_dones.append(dones_next)

            ep_rewards += rewards_np
            ep_lens += 1
            for i in range(cfg.n_envs):
                if dones_np[i]:
                    completed_rewards.append(float(ep_rewards[i]))
                    completed_lens.append(int(ep_lens[i]))
                    ep_rewards[i] = 0.0
                    ep_lens[i] = 0
                    # Log reward breakdown if present.
                    info = infos[i] if i < len(infos) else {}
                    for k, v in (info.get("reward_breakdown") or {}).items():
                        completed_extras.setdefault(k, []).append(float(v))

            obs_t = batch_obs_to_tensors(next_obs_np, device)
            dones_t = dones_next
            global_step += cfg.n_envs

        # --- bootstrap value --------------------------------------------
        with torch.no_grad():
            _, next_value, _ = policy.step(obs_t, hidden, done_mask=dones_t)

        rewards_seq = torch.stack(rollout_rewards, dim=0)              # [T, B]
        values_seq = torch.stack(rollout_values, dim=0)                # [T, B]
        dones_seq = torch.stack(rollout_dones, dim=0)                  # [T, B]
        advantages, returns = compute_gae(
            rewards_seq, values_seq, dones_seq, next_value,
            cfg.gamma, cfg.gae_lambda,
        )
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        ret_flat = returns.reshape(-1)

        old_logp_flat = torch.stack(rollout_logprobs, dim=0).reshape(-1).detach()
        actions_flat = torch.stack(rollout_actions, dim=0).reshape(-1, len(ACTION_FACTORS))

        # Reassemble sequenced obs by key.
        seq_obs = stack_time_batch(rollout_obs)  # each value [T, B, ...]

        # LR decay.
        if cfg.lr_decay:
            frac = 1.0 - min(1.0, global_step / cfg.total_env_steps)
            for g in optim.param_groups:
                g["lr"] = cfg.lr * frac

        # --- PPO epochs -------------------------------------------------
        T = cfg.rollout_steps
        B = cfg.n_envs
        n_samples = T * B
        losses = {"policy": [], "value": [], "entropy": [], "rnd": []}

        for _ in range(cfg.n_epochs):
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
                mb_init = init_hidden[mb_envs]

                logits_list, values_new = policy.sequence_forward(mb_seq_obs, mb_dones, mb_init)

                # Flatten targets for this minibatch (T, |mb|).
                idx_flat = ((torch.arange(T, device=device)[:, None] * B) + mb_envs[None, :]).reshape(-1)
                mb_old_logp = old_logp_flat[idx_flat]
                mb_adv = adv_flat[idx_flat]
                mb_ret = ret_flat[idx_flat]
                mb_actions = actions_flat[idx_flat]

                new_logp = policy.log_prob_from_logits(logits_list, mb_actions)
                entropy = policy.entropy_from_logits(logits_list)

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_new, mb_ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optim.step()

                losses["policy"].append(float(policy_loss.item()))
                losses["value"].append(float(value_loss.item()))
                losses["entropy"].append(float(-entropy_loss.item()))

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
