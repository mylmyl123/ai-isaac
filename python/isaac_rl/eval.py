"""Deterministic-policy evaluation harness (PPO or Dreamer).

Load a checkpoint, run N episodes on a fixed seed set with greedy actions,
report mean reward + Mom-kill rate + floors reached.

    PYTHONPATH=python python -m isaac_rl.eval --checkpoint runs/.../step_1000000.pt \
        --config python/isaac_rl/configs/eval_stage4.yaml --algo ppo
    PYTHONPATH=python python -m isaac_rl.eval --checkpoint runs/dreamer_stage1_*/latest.pt \
        --config python/isaac_rl/dreamer/configs/stage1_single_room.yaml --algo dreamer
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .spaces import ACTION_FACTORS
from .torch_utils import batch_obs_to_tensors
from .vec_env import build_vec_env


log = logging.getLogger("eval")


def _episode_accounting(
    completed_rewards: list[float],
    all_max_stages: list[int],
    beat_mom: int,
    ep_rewards: np.ndarray,
    max_stage_seen: np.ndarray,
    r: np.ndarray,
    terms: np.ndarray,
    truncs: np.ndarray,
    infos: list,
    n_envs: int,
) -> int:
    """Update accounting arrays; return new beat_mom count."""
    ep_rewards += r
    for i in range(n_envs):
        stage = 0
        if isinstance(infos[i], dict):
            raw = infos[i].get("raw", {})
            if isinstance(raw, dict):
                g = raw.get("global", {})
                if isinstance(g, dict):
                    stage = int(g.get("stage", 0) or 0)
        if stage > max_stage_seen[i]:
            max_stage_seen[i] = stage
        if terms[i] or truncs[i]:
            completed_rewards.append(float(ep_rewards[i]))
            bd = infos[i].get("reward_breakdown", {}) if isinstance(infos[i], dict) else {}
            if bd.get("beat_mom"):
                beat_mom += 1
            all_max_stages.append(int(max_stage_seen[i]))
            ep_rewards[i] = 0.0
            max_stage_seen[i] = 0
    return beat_mom


def _evaluate_ppo(cfg, checkpoint: str, n_episodes: int) -> dict:
    """PPO eval — original code path."""
    from .model import IsaacPolicy, PolicyConfig

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = IsaacPolicy(PolicyConfig(**cfg.policy)).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    env = build_vec_env(
        n_envs=cfg.n_envs, base_port=cfg.base_port, reset_stage=cfg.reset_stage,
        max_episode_steps=cfg.max_episode_steps, isaac_binary=cfg.isaac_binary,
        launch_isaac=cfg.launch_isaac, accept_timeout_s=cfg.accept_timeout_s,
    )

    obs_np, _ = env.reset()
    obs_t = batch_obs_to_tensors(obs_np, device)
    hidden = policy.initial_hidden(cfg.n_envs, device)
    dones_t = torch.zeros(cfg.n_envs, device=device)

    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float64)
    completed_rewards: list[float] = []
    beat_mom = 0
    max_stage_seen = np.zeros(cfg.n_envs, dtype=np.int64)
    all_max_stages: list[int] = []

    while len(completed_rewards) < n_episodes:
        with torch.no_grad():
            logits, _, hidden = policy.step(obs_t, hidden, done_mask=dones_t)
            action = policy.sample_from_logits(logits, greedy=True)
        obs_np, r, terms, truncs, infos = env.step(action.cpu().numpy())
        beat_mom = _episode_accounting(
            completed_rewards, all_max_stages, beat_mom, ep_rewards, max_stage_seen,
            r, terms, truncs, infos, cfg.n_envs,
        )
        obs_t = batch_obs_to_tensors(obs_np, device)
        dones_t = torch.as_tensor(np.logical_or(terms, truncs), dtype=torch.float32, device=device)

    env.close()
    return _summarize(completed_rewards, all_max_stages, beat_mom)


def _evaluate_dreamer(cfg, checkpoint: str, n_episodes: int) -> dict:
    """Dreamer eval — actor takes RSSM latent."""
    from .dreamer.action import onehot_to_indices
    from .dreamer.isaac_models import IsaacImagBehavior, IsaacWorldModel

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    wm = IsaacWorldModel(cfg)
    behavior = IsaacImagBehavior(cfg, wm)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"])
    behavior.actor.load_state_dict(ckpt["actor"])
    behavior.critic.load_state_dict(ckpt["critic"])
    wm.eval()
    behavior.actor.eval()

    env = build_vec_env(
        n_envs=cfg.n_envs, base_port=cfg.base_port, reset_stage=cfg.reset_stage,
        max_episode_steps=cfg.max_episode_steps, isaac_binary=cfg.isaac_binary,
        launch_isaac=cfg.launch_isaac, accept_timeout_s=cfg.accept_timeout_s,
    )

    action_factors = tuple(int(x) for x in ACTION_FACTORS.tolist())
    onehot_dim = int(sum(action_factors))

    obs_list, _ = env.reset()
    prev_action_onehot = torch.zeros(cfg.n_envs, onehot_dim, device=device)
    rssm_state = wm.initial_state(cfg.n_envs)
    is_first = torch.ones(cfg.n_envs, device=device)

    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float64)
    completed_rewards: list[float] = []
    beat_mom = 0
    max_stage_seen = np.zeros(cfg.n_envs, dtype=np.int64)
    all_max_stages: list[int] = []

    while len(completed_rewards) < n_episodes:
        obs_t = batch_obs_to_tensors(obs_list, device)
        with torch.no_grad():
            embed = wm.encode_obs(obs_t)
            post, _ = wm.obs_step(rssm_state, prev_action_onehot, embed, is_first)
            feat = wm.dynamics.get_feat(post)
            action_dist = behavior.actor(feat)
            # Greedy: use mode() (argmax one-hot).
            action_onehot = action_dist.mode()
        env_action = onehot_to_indices(action_onehot.cpu(), action_factors).numpy().astype(np.int64)
        obs_list, r, terms, truncs, infos = env.step(env_action)
        beat_mom = _episode_accounting(
            completed_rewards, all_max_stages, beat_mom, ep_rewards, max_stage_seen,
            r, terms, truncs, infos, cfg.n_envs,
        )
        rssm_state = post
        # Reset RSSM row + prev action on done.
        for i in range(cfg.n_envs):
            if terms[i] or truncs[i]:
                for k in rssm_state:
                    rssm_state[k][i] = wm.initial_state(1)[k][0]
                action_onehot[i] = 0.0
        prev_action_onehot = action_onehot
        is_first = torch.as_tensor(np.logical_or(terms, truncs).astype(np.float32), device=device)

    env.close()
    return _summarize(completed_rewards, all_max_stages, beat_mom)


def _summarize(completed_rewards: list[float], all_max_stages: list[int], beat_mom: int) -> dict:
    return {
        "n_episodes": len(completed_rewards),
        "mean_reward": float(np.mean(completed_rewards)) if completed_rewards else 0.0,
        "median_reward": float(np.median(completed_rewards)) if completed_rewards else 0.0,
        "mom_kills": beat_mom,
        "mom_kill_rate": beat_mom / max(1, len(completed_rewards)),
        "mean_max_stage": float(np.mean(all_max_stages)) if all_max_stages else 0.0,
    }


def evaluate(cfg, checkpoint: str, n_episodes: int = 32, algo: str = "ppo") -> dict:
    """Dispatch to the right eval path based on ``algo``."""
    if algo == "dreamer":
        return _evaluate_dreamer(cfg, checkpoint, n_episodes)
    return _evaluate_ppo(cfg, checkpoint, n_episodes)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--algo", choices=("ppo", "dreamer"), default="ppo")
    ap.add_argument("--episodes", type=int, default=32)
    args = ap.parse_args()

    if args.algo == "dreamer":
        from .dreamer.config import cfg_from_yaml
        cfg = cfg_from_yaml(args.config)
    else:
        from .ppo import _cfg_from_yaml
        cfg = _cfg_from_yaml(args.config)
    metrics = evaluate(cfg, args.checkpoint, n_episodes=args.episodes, algo=args.algo)
    log.info("eval metrics (algo=%s):", args.algo)
    for k, v in metrics.items():
        log.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
