"""Deterministic-policy evaluation harness.

Load a checkpoint, run N episodes on a fixed seed set with greedy actions,
report mean reward + Mom-kill rate + floors reached.

    PYTHONPATH=python python -m isaac_rl.eval --checkpoint runs/.../step_1000000.pt \
        --config python/isaac_rl/configs/eval_stage4.yaml
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .model import IsaacPolicy, PolicyConfig
from .ppo import _cfg_from_yaml
from .torch_utils import batch_obs_to_tensors
from .vec_env import build_vec_env


log = logging.getLogger("eval")


def evaluate(cfg, checkpoint: str, n_episodes: int = 32) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    policy = IsaacPolicy(PolicyConfig(**cfg.policy)).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    env = build_vec_env(
        n_envs=cfg.n_envs,
        base_port=cfg.base_port,
        reset_stage=cfg.reset_stage,
        max_episode_steps=cfg.max_episode_steps,
        isaac_binary=cfg.isaac_binary,
        launch_isaac=cfg.launch_isaac,
        accept_timeout_s=cfg.accept_timeout_s,
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
        ep_rewards += r
        for i in range(cfg.n_envs):
            stage = int(infos[i].get("raw", {}).get("global", {}).get("stage", 0)) if isinstance(infos[i], dict) else 0
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
        obs_t = batch_obs_to_tensors(obs_np, device)
        dones_t = torch.as_tensor(np.logical_or(terms, truncs), dtype=torch.float32, device=device)

    env.close()

    metrics = {
        "n_episodes": len(completed_rewards),
        "mean_reward": float(np.mean(completed_rewards)),
        "median_reward": float(np.median(completed_rewards)),
        "mom_kills": beat_mom,
        "mom_kill_rate": beat_mom / max(1, len(completed_rewards)),
        "mean_max_stage": float(np.mean(all_max_stages)) if all_max_stages else 0.0,
    }
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=32)
    args = ap.parse_args()

    cfg = _cfg_from_yaml(args.config)
    metrics = evaluate(cfg, args.checkpoint, n_episodes=args.episodes)
    log.info("eval metrics:")
    for k, v in metrics.items():
        log.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
