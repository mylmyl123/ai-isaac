"""Evaluation harness for the CleanRL PPO trainer (post 2026-07-13 reset).

Load a checkpoint saved by isaac_rl.cleanrl_ppo, run N episodes with
greedy actions, report per-episode reward + kill count.

Usage:
    python -m isaac_rl.eval --checkpoint runs\\<name>\\<ts>\\latest.pt \\
        --config configs\\curriculum.yaml --n-episodes 30

Runs the same env stack as training (via train.py or a manual Isaac fleet).
Passes 'greedy=True' to the actor (argmax of logits per factor).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac_rl.cleanrl_ppo import ActorCritic, PPOConfig, _flat_obs, _obs_dim   # noqa: E402
from isaac_rl.spaces import ACTION_FACTORS                                     # noqa: E402
from isaac_rl.vec_env import build_vec_env                                     # noqa: E402
from isaac_rl.reward import RewardConfig                                       # noqa: E402


log = logging.getLogger("eval")


def _greedy_actions(net: ActorCritic, x: torch.Tensor) -> torch.Tensor:
    """Argmax per action factor. Returns (B, K) int64 tensor."""
    with torch.no_grad():
        dists, _ = net.forward(x)
        return torch.stack([d.logits.argmax(dim=-1) for d in dists], dim=-1)


def evaluate(
    checkpoint: str,
    cfg: PPOConfig,
    n_episodes: int = 30,
) -> dict:
    """Return a metric dict summarizing greedy-policy performance."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    env = build_vec_env(
        n_envs=cfg.n_envs, base_port=cfg.base_port, reset_stage=cfg.reset_stage,
        max_episode_steps=cfg.max_episode_steps,
        launch_isaac=False, reward_config=RewardConfig(),
    )
    obs_dim = _obs_dim(env)
    net = ActorCritic(obs_dim, cfg.hidden_dim, cfg.n_hidden_layers).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()

    obs_list, _ = env.reset()
    obs = torch.from_numpy(np.stack([_flat_obs(o) for o in obs_list])).to(device)

    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_lens = np.zeros(cfg.n_envs, dtype=np.int64)
    ep_kills = np.zeros(cfg.n_envs, dtype=np.int64)
    completed_r, completed_len, completed_kill = [], [], []

    while len(completed_r) < n_episodes:
        actions = _greedy_actions(net, obs).cpu().numpy().astype(np.int64)
        obs_list, rewards, terms, truncs, infos = env.step(actions)
        obs = torch.from_numpy(np.stack([_flat_obs(o) for o in obs_list])).to(device)

        ep_rewards += np.asarray(rewards, dtype=np.float32)
        ep_lens += 1
        for i, info in enumerate(infos):
            for ev in (info.get("raw") or {}).get("events", []) or []:
                if ev.get("kind") == "kill":
                    ep_kills[i] += 1

        for i in range(cfg.n_envs):
            if terms[i] or truncs[i]:
                completed_r.append(float(ep_rewards[i]))
                completed_len.append(int(ep_lens[i]))
                completed_kill.append(int(ep_kills[i]))
                ep_rewards[i] = 0.0
                ep_lens[i] = 0
                ep_kills[i] = 0
                log.info("episode %d/%d: r=%.2f len=%d kills=%d",
                         len(completed_r), n_episodes, completed_r[-1], completed_len[-1], completed_kill[-1])

    env.close()

    return {
        "n_episodes": len(completed_r),
        "mean_reward": float(np.mean(completed_r)),
        "std_reward": float(np.std(completed_r)),
        "mean_length": float(np.mean(completed_len)),
        "mean_kills": float(np.mean(completed_kill)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to a latest.pt from a training run.")
    ap.add_argument("--config", required=True, help="YAML config path (same one used for training).")
    ap.add_argument("--n-episodes", type=int, default=30)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = PPOConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    results = evaluate(args.checkpoint, cfg, args.n_episodes)
    log.info("EVAL RESULTS:")
    for k, v in results.items():
        log.info("  %s = %s", k, v)


if __name__ == "__main__":
    main()
