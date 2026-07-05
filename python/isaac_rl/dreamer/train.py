"""DreamerV3 training loop for Isaac RL.

Structure:
  1. Build vec env (existing build_vec_env; shared with PPO)
  2. Instantiate IsaacWorldModel + IsaacImagBehavior
  3. Replay buffer, empty
  4. Prefill: random-policy rollout for cfg.prefill_steps env-steps
  5. Main loop, until cfg.total_env_steps env-steps:
     a. Env rollout N steps with current actor
        (actor takes RSSM latent produced online during rollout)
     b. For each env step this round: cfg.train_ratio WM gradient updates
     c. After WM update: imagination rollouts + actor/critic updates
     d. Log to TB every cfg.log_every updates
     e. Checkpoint every cfg.checkpoint_every env-steps
     f. Save + exit clean on SIGINT

Copies the shape of ppo.py:531-991 for signal handling, checkpointing, TB.
"""
from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..spaces import ACTION_FACTORS, flatten_dict_obs
from ..torch_utils import batch_obs_to_tensors
from ..vec_env import build_vec_env
from .action import indices_to_onehot, onehot_to_indices
from .config import DreamerConfig, cfg_from_yaml
from .isaac_models import IsaacImagBehavior, IsaacWorldModel
from .replay import SequenceReplay, encode_and_add


log = logging.getLogger("dreamer")

ACTION_FACTORS_TUPLE = tuple(int(x) for x in ACTION_FACTORS.tolist())
ONEHOT_DIM = int(sum(ACTION_FACTORS_TUPLE))


def _obs_batch_to_torch(obs_list: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack a list of env obs (nested dicts) into a batched flat-dict tensor set."""
    return batch_obs_to_tensors(obs_list, device)


def _sample_random_action(n_envs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (env_action[n_envs, 2] int64, onehot[n_envs, ONEHOT_DIM] float32)."""
    move = rng.integers(0, ACTION_FACTORS_TUPLE[0], size=n_envs)
    shoot = rng.integers(0, ACTION_FACTORS_TUPLE[1], size=n_envs)
    env_action = np.stack([move, shoot], axis=1).astype(np.int64)
    onehot = np.zeros((n_envs, ONEHOT_DIM), dtype=np.float32)
    for i in range(n_envs):
        onehot[i, move[i]] = 1.0
        onehot[i, ACTION_FACTORS_TUPLE[0] + shoot[i]] = 1.0
    return env_action, onehot


def train(cfg: DreamerConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ---- reward config -------------------------------------------------
    from ..reward import RewardConfig
    reward_cfg = RewardConfig()
    for k, v in (cfg.reward or {}).items():
        if hasattr(reward_cfg, k):
            setattr(reward_cfg, k, v)
            log.info("reward override: %s = %s", k, v)

    # ---- vec env -------------------------------------------------------
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

    # ---- models --------------------------------------------------------
    world_model = IsaacWorldModel(cfg)
    behavior = IsaacImagBehavior(cfg, world_model)
    log.info(
        "params: WM=%.2fM  actor=%.2fM  critic=%.2fM",
        sum(p.numel() for p in world_model.parameters()) / 1e6,
        sum(p.numel() for p in behavior.actor.parameters()) / 1e6,
        sum(p.numel() for p in behavior.critic.parameters()) / 1e6,
    )

    # ---- replay --------------------------------------------------------
    replay = SequenceReplay(cfg.replay_capacity, onehot_dim=ONEHOT_DIM)

    # ---- logging -------------------------------------------------------
    run_dir = Path(cfg.checkpoint_dir) / cfg.run_name / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir) if SummaryWriter is not None else None
    log.info("run dir: %s", run_dir)

    # ---- reset ----------------------------------------------------------
    obs_list, _ = env.reset()
    prev_action_onehot = np.zeros((cfg.n_envs, ONEHOT_DIM), dtype=np.float32)
    # RSSM state per env (batched, size [n_envs, ...])
    rssm_state = world_model.initial_state(cfg.n_envs)
    # is_first for the first observation of each episode.
    is_first_flags = np.ones(cfg.n_envs, dtype=bool)

    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float64)
    ep_lens = np.zeros(cfg.n_envs, dtype=np.int64)
    completed_rewards: list[float] = []
    completed_lens: list[int] = []
    completed_extras: dict[str, list[float]] = {}

    global_step = 0
    update = 0
    t_start = time.time()

    # ---- resume --------------------------------------------------------
    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from).expanduser()
        if ckpt_path.exists():
            log.info("resume: loading %s", ckpt_path)
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            world_model.load_state_dict(state["world_model"])
            behavior.actor.load_state_dict(state["actor"])
            behavior.critic.load_state_dict(state["critic"])
            global_step = int(state.get("global_step", 0))
            log.info("resume: continuing from step %d", global_step)
        else:
            log.warning("resume: %s does not exist, starting fresh", ckpt_path)

    # ---- checkpoint helper --------------------------------------------
    def _save_ckpt(tag: str) -> None:
        ckpt_path = run_dir / "ckpts" / f"step_{global_step}.pt"
        try:
            torch.save({
                "world_model": world_model.state_dict(),
                "actor": behavior.actor.state_dict(),
                "critic": behavior.critic.state_dict(),
                "cfg": asdict(cfg),
                "global_step": global_step,
            }, ckpt_path)
            import shutil
            shutil.copyfile(ckpt_path, run_dir / "latest.pt")
            log.info("[%s] saved checkpoint: %s (also latest.pt)", tag, ckpt_path)
        except Exception as e:
            log.exception("[%s] failed to save checkpoint: %s", tag, e)

    # ---- signal handling ----------------------------------------------
    shutdown = {"flag": False}
    def _on_sigint(signum, frame):
        if not shutdown["flag"]:
            log.warning("Ctrl+C received; will save and exit after this iter. Ctrl+C again to force.")
            shutdown["flag"] = True
        else:
            log.warning("Second Ctrl+C — aborting immediately")
            raise KeyboardInterrupt()
    try:
        prev_sigint = signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, AttributeError):
        prev_sigint = None

    # ==================================================================
    # PREFILL: random policy fills replay so we can start WM training.
    # ==================================================================
    log.info("prefill: %d env-steps with random policy", cfg.prefill_steps)
    prefill_done = 0
    while prefill_done < cfg.prefill_steps and not shutdown["flag"]:
        env_action, onehot = _sample_random_action(cfg.n_envs, rng)
        next_obs_list, rewards_np, terms, truncs, infos = env.step(env_action)
        for i in range(cfg.n_envs):
            replay.add(
                flatten_dict_obs(obs_list[i]),
                onehot[i],
                float(rewards_np[i]),
                is_first=bool(is_first_flags[i]),
                is_terminal=bool(terms[i]),
                is_last=bool(terms[i] or truncs[i]),
            )
        prefill_done += cfg.n_envs
        global_step += cfg.n_envs
        # Track ep rewards during prefill so the first TB entries aren't blank.
        ep_rewards += rewards_np
        ep_lens += 1
        for i in range(cfg.n_envs):
            if terms[i] or truncs[i]:
                completed_rewards.append(float(ep_rewards[i]))
                completed_lens.append(int(ep_lens[i]))
                ep_rewards[i] = 0.0
                ep_lens[i] = 0
                info = infos[i] if i < len(infos) else {}
                for k, v in (info.get("reward_breakdown") or {}).items():
                    completed_extras.setdefault(k, []).append(float(v))
        # is_first on next step is True if this step ended an episode.
        is_first_flags = np.logical_or(terms, truncs)
        obs_list = next_obs_list
    log.info("prefill complete: replay has %d transitions", len(replay))

    # ==================================================================
    # MAIN LOOP: env rollout -> WM + behavior updates.
    # ==================================================================
    heartbeat_t = time.time()
    while global_step < cfg.total_env_steps and not shutdown["flag"]:
        # ---- env rollout: N steps -------------------------------------
        # We interleave env stepping with WM/behavior updates. Each iteration:
        # step envs once, then run train_ratio WM+behavior updates.
        # This mirrors NM512's per-env-step update schedule.

        # (a) One env step using the current policy.
        obs_t = _obs_batch_to_torch(obs_list, device)
        # Reset RSSM state on newly-first steps.
        is_first_t = torch.as_tensor(is_first_flags.astype(np.float32), device=device)
        with torch.no_grad():
            embed = world_model.encode_obs(obs_t)
            prev_action_t = torch.as_tensor(prev_action_onehot, device=device)
            post, _ = world_model.obs_step(rssm_state, prev_action_t, embed, is_first_t)
            feat = world_model.dynamics.get_feat(post)
            action_dist = behavior.actor(feat)
            action_onehot_t = action_dist.sample()             # [n_envs, ONEHOT_DIM]
            action_onehot = action_onehot_t.cpu().numpy()
        # RSSM state carries forward.
        rssm_state = post

        # Convert one-hot -> env-facing int actions.
        env_action = onehot_to_indices(
            torch.as_tensor(action_onehot), ACTION_FACTORS_TUPLE,
        ).numpy().astype(np.int64)

        next_obs_list, rewards_np, terms, truncs, infos = env.step(env_action)

        # Push transitions to replay. is_first is *current* obs's is_first,
        # meaning "the RSSM should reset at this step". The current obs was
        # observed BEFORE the action; if this env just reset, is_first_flags
        # reflects that.
        for i in range(cfg.n_envs):
            replay.add(
                flatten_dict_obs(obs_list[i]),
                action_onehot[i],
                float(rewards_np[i]),
                is_first=bool(is_first_flags[i]),
                is_terminal=bool(terms[i]),
                is_last=bool(terms[i] or truncs[i]),
            )
        global_step += cfg.n_envs
        ep_rewards += rewards_np
        ep_lens += 1
        for i in range(cfg.n_envs):
            if terms[i] or truncs[i]:
                completed_rewards.append(float(ep_rewards[i]))
                completed_lens.append(int(ep_lens[i]))
                ep_rewards[i] = 0.0
                ep_lens[i] = 0
                info = infos[i] if i < len(infos) else {}
                for k, v in (info.get("reward_breakdown") or {}).items():
                    completed_extras.setdefault(k, []).append(float(v))
        # Reset RSSM state row + one-hot action for env rows that just terminated.
        for i in range(cfg.n_envs):
            if terms[i] or truncs[i]:
                for k in rssm_state:
                    rssm_state[k][i] = world_model.initial_state(1)[k][0]
                action_onehot[i] = 0.0
        is_first_flags = np.logical_or(terms, truncs)
        prev_action_onehot = action_onehot
        obs_list = next_obs_list

        # (b) WM + behavior updates. train_ratio grad-steps per env-step per env.
        n_updates = max(1, cfg.train_ratio // cfg.n_envs)
        wm_metrics: dict[str, float] = {}
        beh_metrics: dict[str, float] = {}
        for _ in range(n_updates):
            if len(replay) < cfg.batch_size * cfg.seq_len:
                break
            batch = replay.sample(cfg.batch_size, cfg.seq_len, rng=rng)
            post_batch, ctx, wmm = world_model.train_step(batch)
            wm_metrics = wmm
            bmm = behavior.train_step(post_batch)
            beh_metrics = bmm
            update += 1

        # ---- log ------------------------------------------------------
        if update > 0 and (update % cfg.log_every == 0 or shutdown["flag"]):
            sps = global_step / max(1e-6, time.time() - t_start)
            recent = completed_rewards[-32:] or [0.0]
            recent_lens = completed_lens[-32:] or [0]
            recent_r = float(np.mean(recent))
            recent_len = float(np.mean(recent_lens))
            pct = 100.0 * global_step / max(1, cfg.total_env_steps)
            best_r = max(completed_rewards) if completed_rewards else 0.0
            n_eps = len(completed_rewards)
            log.info(
                "[step %s/%s %.1f%%] upd=%d sps=%.0f ep=%d ep_r=%+.2f (best %+.2f) ep_len=%.0f | wm=%.2f actor=%+.4f critic=%.3f",
                f"{global_step:,}", f"{cfg.total_env_steps:,}", pct, update, sps,
                n_eps, recent_r, best_r, recent_len,
                wm_metrics.get("loss/total", float("nan")),
                beh_metrics.get("loss/actor", float("nan")),
                beh_metrics.get("loss/critic", float("nan")),
            )
            if writer is not None:
                writer.add_scalar("perf/sps", sps, global_step)
                writer.add_scalar("perf/updates", update, global_step)
                writer.add_scalar("rollout/ep_reward", recent_r, global_step)
                writer.add_scalar("rollout/ep_reward_best", best_r, global_step)
                writer.add_scalar("rollout/ep_length", recent_len, global_step)
                writer.add_scalar("rollout/n_episodes", n_eps, global_step)
                for k, v in wm_metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(k, v, global_step)
                for k, v in beh_metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(k, v, global_step)
                for k, vs in completed_extras.items():
                    if vs:
                        writer.add_scalar(f"reward/{k}", float(np.mean(vs[-64:])), global_step)
            heartbeat_t = time.time()

        # ---- checkpoint -----------------------------------------------
        boundary = cfg.checkpoint_every
        if boundary and (global_step // boundary) > ((global_step - cfg.n_envs) // boundary):
            _save_ckpt("scheduled")

        # ---- heartbeat -----------------------------------------------
        if time.time() - heartbeat_t > 30.0:
            sps = global_step / max(1e-6, time.time() - t_start)
            log.info("... running (step=%s sps=%.0f replay=%d)", f"{global_step:,}", sps, len(replay))
            heartbeat_t = time.time()

    # ---- final save + shutdown ---------------------------------------
    if global_step >= cfg.total_env_steps and not shutdown["flag"]:
        _save_ckpt("complete")
    elif shutdown["flag"]:
        _save_ckpt("interrupted")

    if prev_sigint is not None:
        try:
            signal.signal(signal.SIGINT, prev_sigint)
        except (ValueError, AttributeError):
            pass

    log.info("dreamer training complete")
    env.close()
    if writer is not None:
        writer.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = cfg_from_yaml(args.config)
    for kv in args.override:
        k, _, v = kv.partition("=")
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
