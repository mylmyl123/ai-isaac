"""Vectorized Isaac environment.

We can't use gym.AsyncVectorEnv naively because each env needs to bind a distinct
port and be paired with its own Isaac process. Simplest correct thing on one
machine: run each env in-thread with its own socket, and step them sequentially.

That sounds slow but Isaac's game clock is the bottleneck (30 Hz per instance).
When the trainer sends actions to env i, env j's Isaac is already running its
next frame in parallel. The gains from real async are modest; keeping this simple
avoids a class of pickling / process-boundary bugs.

If you want true parallelism later, swap this for gym.AsyncVectorEnv with
per-worker port assignment. See launch_env() below — it's already picklable.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any

import numpy as np

from .env import SocketIsaacEnv
from .reward import RewardConfig


log = logging.getLogger(__name__)


class SyncVecEnv:
    """N SocketIsaacEnv workers stepped sequentially in a single thread."""

    def __init__(self, envs: list[SocketIsaacEnv]):
        self.envs = envs
        self.n = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self._last_obs: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None):
        obs = []
        infos = []
        for i, env in enumerate(self.envs):
            s = None if seed is None else seed + i
            o, info = env.reset(seed=s)
            obs.append(o)
            infos.append(info)
        self._last_obs = obs
        return obs, infos

    def step(self, actions: np.ndarray):
        obs = []
        rewards = np.zeros(self.n, dtype=np.float32)
        terms = np.zeros(self.n, dtype=bool)
        truncs = np.zeros(self.n, dtype=bool)
        infos = []
        # DreamerV3 needs the terminal obs *before* auto-reset so it can train
        # the continue-flag / reward decoder on the actual last-of-episode state.
        # PPO ignores this field — it only uses `dones` masking. Fully
        # backward-compatible: existing callers keep unpacking the 5-tuple.
        terminal_obs: list[dict[str, Any] | None] = []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            rewards[i] = r
            terms[i] = term
            truncs[i] = trunc
            if term or trunc:
                # Preserve pre-reset obs AND the terminal step's info dict
                # (which carries reward_breakdown from the RewardShaper). Both
                # PPO and Dreamer log reward_breakdown from completed episodes;
                # if we let env.reset() overwrite info, they see empty breakdowns
                # every time — silent bug that hid room_clear/kill/damage
                # events from TensorBoard for the entire history of the project.
                terminal_obs.append(o)
                terminal_info = info                                  # preserve
                o, reset_info = env.reset()
                info = reset_info
                # Splice reward_breakdown (and any other reward-side keys) back
                # in from the terminal step so completed_extras logging works.
                if isinstance(terminal_info, dict) and "reward_breakdown" in terminal_info:
                    info["reward_breakdown"] = terminal_info["reward_breakdown"]
                # Same for the episode-total breakdown (2026-07-08). This is
                # what trainers should PREFER for reward/{k} logging — the
                # terminal-step breakdown alone hid all non-terminal reward
                # events (kill, damage_dealt, new_room, room_clear, ...).
                if isinstance(terminal_info, dict) and "reward_breakdown_episode" in terminal_info:
                    info["reward_breakdown_episode"] = terminal_info["reward_breakdown_episode"]
                # Same for ep_end_reason (added 2026-07-07 to distinguish
                # real crashes from proper shaper-terminated episodes).
                if isinstance(terminal_info, dict) and "ep_end_reason" in terminal_info:
                    info["ep_end_reason"] = terminal_info["ep_end_reason"]
            else:
                terminal_obs.append(None)
            obs.append(o)
            infos.append(info)
        self._last_obs = obs
        # Attach terminal_obs on infos too, for callers that only unpack the
        # 5-tuple (i.e. existing PPO code path). Zero risk to PPO — it never
        # reads info["terminal_obs"].
        for i, tobs in enumerate(terminal_obs):
            if tobs is not None:
                infos[i]["terminal_obs"] = tobs
        return obs, rewards, terms, truncs, infos

    def close(self):
        for env in self.envs:
            env.close()


def _launch_isaac_process(port: int, isaac_binary: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["ISAAC_RL_PORT"] = str(port)
    cmd = [isaac_binary, "--luadebug"]
    log.info("launching isaac: %s (port=%d)", " ".join(cmd), port)
    return subprocess.Popen(cmd, env=env)


def build_vec_env(
    n_envs: int,
    base_port: int = 9500,
    reset_stage: int | None = None,
    max_episode_steps: int = 27000,
    isaac_binary: str | None = None,
    launch_isaac: bool = True,
    reward_config: RewardConfig | None = None,
    accept_timeout_s: float = 300.0,
) -> SyncVecEnv:
    """Bind N ports, optionally spawn N Isaac processes, wait for them to connect."""
    envs: list[SocketIsaacEnv] = []
    for i in range(n_envs):
        port = base_port + i
        env = SocketIsaacEnv(
            port=port,
            accept_timeout_s=accept_timeout_s,
            max_steps=max_episode_steps,
            reward_config=reward_config,
            reset_stage=reset_stage,
            env_idx=i,
        )
        envs.append(env)

    if launch_isaac:
        if not isaac_binary:
            raise ValueError(
                "launch_isaac=True but isaac_binary not set. "
                "Set ppo.isaac_binary in your config or pass launch_isaac=false and start Isaac manually."
            )
        for i in range(n_envs):
            _launch_isaac_process(base_port + i, isaac_binary)
            # Small stagger so the first frames don't fight for CPU during load.
            time.sleep(1.0)

    return SyncVecEnv(envs)
