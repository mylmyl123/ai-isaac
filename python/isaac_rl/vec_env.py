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
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            rewards[i] = r
            terms[i] = term
            truncs[i] = trunc
            if term or trunc:
                # Auto-reset for on-policy training convenience.
                o, info = env.reset()
            obs.append(o)
            infos.append(info)
        self._last_obs = obs
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
