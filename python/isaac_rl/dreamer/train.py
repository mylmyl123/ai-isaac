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
from collections import deque
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
from ..reward import REWARD_BREAKDOWN_KEYS
from ..torch_utils import batch_obs_to_tensors
from ..vec_env import build_vec_env
from .action import indices_to_onehot, onehot_to_indices
from .config import DreamerConfig, cfg_from_yaml
from .isaac_models import IsaacImagBehavior, IsaacWorldModel
from .replay import SequenceReplay, encode_and_add


log = logging.getLogger("dreamer")

ACTION_FACTORS_TUPLE = tuple(int(x) for x in ACTION_FACTORS.tolist())
ONEHOT_DIM = int(sum(ACTION_FACTORS_TUPLE))


class _Prof:
    """Lightweight per-section wall-clock profiler with CUDA sync.

    Uses time.perf_counter with torch.cuda.synchronize() bracketing so GPU
    ops actually finish before the timer stops. Accumulates per-section
    totals across the interval; on dump we log the mean-per-call time to
    TB as ``time/<section>_ms`` and share-of-total as ``time_pct/<section>``.

    Usage:
        prof = _Prof(device)
        with prof("env_step"):
            ...
        prof.dump_to_writer(writer, global_step)
        prof.reset()
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self._device = device
        self._is_cuda = device.type == "cuda"
        self._enabled = enabled
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._active: str | None = None
        self._t0: float = 0.0

    def __call__(self, section: str):
        return _ProfContext(self, section)

    def _start(self, section: str):
        if not self._enabled:
            return
        if self._is_cuda:
            torch.cuda.synchronize()
        self._active = section
        self._t0 = time.perf_counter()

    def _stop(self):
        if not self._enabled or self._active is None:
            return
        if self._is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self._sums[self._active] = self._sums.get(self._active, 0.0) + dt
        self._counts[self._active] = self._counts.get(self._active, 0) + 1
        self._active = None

    def dump_to_writer(self, writer, step: int, log_lines: bool = False):
        if not self._enabled or writer is None:
            return
        for k, total_s in self._sums.items():
            n = self._counts.get(k, 1)
            ms = 1000.0 * total_s / max(1, n)
            writer.add_scalar(f"time/{k}_ms", ms, step)
        total = sum(self._sums.values())
        if total > 0:
            for k, s in self._sums.items():
                writer.add_scalar(f"time_pct/{k}", 100.0 * s / total, step)
        if log_lines:
            parts = sorted(self._sums.keys(), key=lambda k: -self._sums[k])
            summary = ", ".join(
                f"{k}={1000.0 * self._sums[k] / max(1, self._counts.get(k, 1)):.1f}ms"
                for k in parts
            )
            log.info("prof: %s", summary)

    def reset(self):
        self._sums.clear()
        self._counts.clear()


class _ProfContext:
    def __init__(self, prof: _Prof, section: str):
        self._prof = prof
        self._section = section

    def __enter__(self):
        self._prof._start(self._section)
        return self

    def __exit__(self, *args):
        self._prof._stop()


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

    # ---- Speed knobs (2026-07-05) --------------------------------------
    # TF32 matmul on Ampere+ (only matters when NOT using bf16/fp16 autocast).
    if getattr(cfg, "tf32", True) and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # cuDNN autotuner: 30s warmup cost, ~5% steady-state speedup.
    if getattr(cfg, "cudnn_benchmark", True) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_str = getattr(cfg, "amp_dtype", "off")
    if amp_str in ("bf16", "fp16") and device.type == "cuda":
        log.info("AMP enabled: dtype=%s (Ampere+ recommended: bf16)", amp_str)
    elif device.type == "cuda":
        log.info("AMP disabled (running in fp32 + TF32=%s)", getattr(cfg, "tf32", True))

    # Per-section profiler. Times are dumped to TB every log_every updates as
    # time/<section>_ms (mean-per-call) and time_pct/<section> (share of total
    # interval). Reveals where wall-clock is going: env-step, encoder, RSSM
    # observe, decoder, WM backward, imagination, actor/critic, replay sample,
    # obs marshaling. Disabled on CPU (no meaningful CUDA sync).
    prof = _Prof(device, enabled=True)

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
        "params: WM=%.2fM  actor=%.2fM  critic=%.2fM  rssm_compiled=%s",
        sum(p.numel() for p in world_model.parameters()) / 1e6,
        sum(p.numel() for p in behavior.actor.parameters()) / 1e6,
        sum(p.numel() for p in behavior.critic.parameters()) / 1e6,
        getattr(world_model, "_rssm_compiled", False),
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
    # Pre-populate with every known reward key so "never fired" components
    # still show up as flat-zero traces in TB. Bounded deque (last 512 eps)
    # prevents unbounded memory growth on long runs. See REWARD_BREAKDOWN_KEYS
    # in reward.py for the source of truth.
    _EXTRAS_WINDOW = 512
    completed_extras: dict[str, deque[float]] = {
        k: deque(maxlen=_EXTRAS_WINDOW) for k in REWARD_BREAKDOWN_KEYS
    }
    # Track episode-end reason. Categories (post 2026-07-09 crash-split):
    #   - shaper_terminated: normal termination via shaper (death, floor clear, etc.)
    #   - truncated: hit max_episode_steps
    #   - mod_restart: mod cycled its socket cleanly (in-process restart on death).
    #     NOT a penalty case — the HP-based death detection in the shaper
    #     already handled the actual death.
    #   - isaac_crash: Isaac process actually died (real crash). Applies -1 penalty.
    #   - mod_socket_error: LEGACY (pre-2026-07-09) combined category. Kept for
    #     backward-compatible tag names when reading old runs.
    #   - unknown: catch-all fallback.
    _ENDS = ("shaper_terminated", "truncated", "mod_restart", "isaac_crash", "mod_socket_error", "unknown")
    completed_end_reasons: dict[str, deque[int]] = {
        r: deque(maxlen=_EXTRAS_WINDOW) for r in _ENDS
    }
    _crash_warned = False

    # 2026-07-09: Phase C behavior metrics. Aggregated in the shaper, emitted
    # via info["behavior_metrics"] on each terminal step. Logged as
    # `behavior/{metric}` scalars — pure telemetry, NOT rewards.
    completed_behavior: dict[str, deque[float]] = {}

    def _record_ep_extras(bd: dict) -> None:
        """Append per-episode reward-breakdown to completed_extras.

        Zero-fills known keys that didn't appear this episode so every reward
        component gets a datapoint per episode (needed for meaningful
        frac_nonzero and mean stats). Unknown keys (e.g. from a newer env) are
        added on the fly with a bounded deque.
        """
        seen: set[str] = set()
        for k, v in (bd or {}).items():
            if k not in completed_extras:
                completed_extras[k] = deque(maxlen=_EXTRAS_WINDOW)
            completed_extras[k].append(float(v))
            seen.add(k)
        for k in REWARD_BREAKDOWN_KEYS:
            if k not in seen:
                completed_extras[k].append(0.0)

    def _record_ep_end_reason(info: dict) -> None:
        """One-hot append the ep_end_reason from `info` to each reason's deque."""
        reason = info.get("ep_end_reason", "unknown") if isinstance(info, dict) else "unknown"
        if reason not in completed_end_reasons:
            reason = "unknown"
        for r in _ENDS:
            completed_end_reasons[r].append(1 if r == reason else 0)

    def _record_ep_behavior(info: dict) -> None:
        """Record per-episode behavior metrics for TB logging. 2026-07-09 Phase C.

        These are PURE TELEMETRY — not rewards. They answer 'is the agent
        starting to do hierarchical Isaac gameplay' (visit shops, use items,
        reach later floors, kill bosses) independent of what we're
        explicitly shaping. See RewardShaper.episode_behavior_metrics() for
        the metric set.
        """
        if not isinstance(info, dict):
            return
        beh = info.get("behavior_metrics")
        if not isinstance(beh, dict):
            return
        for k, v in beh.items():
            if k not in completed_behavior:
                completed_behavior[k] = deque(maxlen=_EXTRAS_WINDOW)
            completed_behavior[k].append(float(v))

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
                # Prefer the per-episode sum breakdown over the terminal-tick
                # one. The terminal-tick breakdown is what shipped through
                # 2026-07-07; the episode-total breakdown was added 2026-07-08
                # after we discovered the terminal-only view hid all
                # non-terminal reward events (kill, damage_dealt, new_room,
                # room_clear, pickup_*) from TB.
                bd = info.get("reward_breakdown_episode") or info.get("reward_breakdown") or {}
                _record_ep_extras(bd)
                _record_ep_end_reason(info)
                _record_ep_behavior(info)
        # is_first on next step is True if this step ended an episode.
        is_first_flags = np.logical_or(terms, truncs)
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
        with prof("act_marshal_obs"):
            obs_t = _obs_batch_to_torch(obs_list, device)
            # Reset RSSM state on newly-first steps.
            is_first_t = torch.as_tensor(is_first_flags.astype(np.float32), device=device)
        with prof("act_forward"):
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

        with prof("env_step"):
            next_obs_list, rewards_np, terms, truncs, infos = env.step(env_action)

        # Push transitions to replay. is_first is *current* obs's is_first,
        # meaning "the RSSM should reset at this step". The current obs was
        # observed BEFORE the action; if this env just reset, is_first_flags
        # reflects that.
        with prof("replay_add"):
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
                bd = info.get("reward_breakdown_episode") or info.get("reward_breakdown") or {}
                _record_ep_extras(bd)
                _record_ep_end_reason(info)
                _record_ep_behavior(info)
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
            with prof("replay_sample"):
                batch = replay.sample(cfg.batch_size, cfg.seq_len, rng=rng)
            with prof("wm_train_step"):
                post_batch, ctx, wmm = world_model.train_step(batch)
            wm_metrics = wmm
            # Propagate WM sub-timers up to the outer profiler.
            for _name, _ms in getattr(world_model, "last_step_times", {}).items():
                prof._sums[_name] = prof._sums.get(_name, 0.0) + (_ms / 1000.0)
                prof._counts[_name] = prof._counts.get(_name, 0) + 1
            with prof("beh_train_step"):
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
                    if not vs:
                        continue
                    # Last 64 episodes: mean of the per-episode sum of this
                    # component + fraction of those episodes where it fired.
                    tail = list(vs)[-64:]
                    arr = np.asarray(tail, dtype=np.float32)
                    writer.add_scalar(f"reward/{k}", float(arr.mean()), global_step)
                    writer.add_scalar(
                        f"reward/{k}_frac_nonzero",
                        float((arr != 0.0).mean()),
                        global_step,
                    )
                # Phase C behavior metrics (2026-07-09). Same last-64-episode
                # window as reward/{k}. Prefixed `behavior/` to keep them
                # separate in TB. Purely telemetry — no gradient impact.
                for k, vs in completed_behavior.items():
                    if not vs:
                        continue
                    tail = list(vs)[-64:]
                    arr = np.asarray(tail, dtype=np.float32)
                    writer.add_scalar(f"behavior/{k}", float(arr.mean()), global_step)
                    writer.add_scalar(f"behavior/{k}_max", float(arr.max()), global_step)
                # Episode-end reason distribution. `mod_socket_error` dominating
                # (>50%) means the mod's terminal-obs send is failing every
                # episode — usually because the Isaac window is backgrounded
                # and Windows throttles it. Warn once when this happens.
                for r in _ENDS:
                    vs2 = completed_end_reasons[r]
                    if not vs2:
                        continue
                    tail = list(vs2)[-64:]
                    frac = float(np.mean(tail)) if tail else 0.0
                    writer.add_scalar(f"rollout/ep_end_{r}_frac", frac, global_step)
                # 2026-07-09: after crash-split, watch isaac_crash (real
                # crash, applies -1 penalty) separately from mod_restart
                # (clean mod cycle, no penalty). High isaac_crash rate =
                # actual instability. High mod_restart rate = high death
                # rate (agent dying a lot) which is normal for early
                # training and NOT a problem — don't warn on it.
                mse_tail = (
                    list(completed_end_reasons["isaac_crash"])[-64:]
                    or list(completed_end_reasons["mod_socket_error"])[-64:]
                )
                if len(mse_tail) >= 32 and not _crash_warned:
                    mse_frac = float(np.mean(mse_tail))
                    if mse_frac > 0.5:
                        log.warning(
                            "HIGH SOCKET-ERROR RATE: %.0f%% of the last %d episodes ended via "
                            "mod socket error, not the shaper. Check the Isaac window is "
                            "focused + visible (backgrounded windows are throttled and drop "
                            "the terminal obs, so the agent sees ONLY -1 crash penalty per "
                            "episode with no HP or in-game reward signal).",
                            100 * mse_frac, len(mse_tail),
                        )
                        _crash_warned = True
                # Per-section timing dump. Log a summary line every 10 dumps so
                # the terminal shows what's dominating without spamming.
                prof.dump_to_writer(writer, global_step, log_lines=(update % (cfg.log_every * 10) == 0))
                prof.reset()
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
