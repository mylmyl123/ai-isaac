"""Gymnasium environment that talks to one Isaac process over a TCP socket.

Design notes:
- The trainer owns the server socket. Isaac connects into us via LuaSocket.
- One env == one Isaac process. Multi-env parallelism is in vec_env.py.
- `reset()` sends a `reset` command down the wire; Lua runs `restart 0` on the
  next tick and reconnects on MC_POST_GAME_STARTED.
- `step()` is synchronous: send action → wait for next obs frame → shape reward.
"""
from __future__ import annotations

import logging
import socket
import time
from typing import Any

import gymnasium as gym
import numpy as np

from .protocol import recv_frame, send_frame
from .reward import RewardConfig, RewardShaper
from .spaces import (
    ACTION_FACTORS,
    action_space,
    encode_action,
    encode_obs,
    observation_space,
)


log = logging.getLogger(__name__)


class SocketIsaacEnv(gym.Env):
    """One Isaac instance behind a socket. Step-locked with the mod at the control rate."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 9500,
        host: str = "127.0.0.1",
        accept_timeout_s: float = 300.0,
        max_steps: int = 27000,          # ~30 min at 15 Hz
        reward_config: RewardConfig | None = None,
        reset_stage: int | None = None,   # curriculum: force `stage N` on reset
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.accept_timeout_s = accept_timeout_s
        self.max_steps = max_steps
        self.reset_stage = reset_stage

        self.observation_space = observation_space()
        self.action_space = action_space()

        self._server: socket.socket | None = None
        self._client: socket.socket | None = None
        self._last_action = np.zeros(len(ACTION_FACTORS), dtype=np.int64)
        self._last_seed: int | None = None
        self._steps = 0

        self.reward_shaper = RewardShaper(reward_config)

        self._open_server()

    # -- lifecycle --------------------------------------------------------

    def _open_server(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(1)
        self._server = s
        log.info("listening for Isaac on %s:%d", self.host, self.port)

    def _accept(self) -> None:
        assert self._server is not None
        self._server.settimeout(self.accept_timeout_s)
        client, addr = self._server.accept()
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("Isaac connected from %s", addr)
        self._client = client
        hello = recv_frame(client)
        self._last_seed = hello.get("seed")
        log.info("handshake: %s", hello)

    def close(self) -> None:
        for s in (self._client, self._server):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._client = None
        self._server = None

    # -- gym api ----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.reward_shaper.reset()

        if self._client is None:
            self._accept()
            raw = recv_frame(self._client)
        else:
            payload: dict[str, Any] = {"reset": True}
            if seed is not None:
                payload["seed"] = int(seed)
            if self.reset_stage is not None:
                payload["stage"] = int(self.reset_stage)
            send_frame(self._client, payload)
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None
            self._accept()
            raw = recv_frame(self._client)

        self._steps = 0
        self._last_action[:] = 0
        obs = encode_obs(raw, last_action=self._last_action)
        info: dict[str, Any] = {"seed": self._last_seed, "raw": raw}
        return obs, info

    def step(self, action):
        assert self._client is not None, "reset() must be called before step()"
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        send_frame(self._client, encode_action(a))
        raw = recv_frame(self._client)
        self._last_action = a
        self._steps += 1
        obs = encode_obs(raw, last_action=self._last_action)

        reward, terminated, breakdown = self.reward_shaper(raw)
        truncated = self._steps >= self.max_steps
        info: dict[str, Any] = {
            "raw": raw,
            "steps": self._steps,
            "reward_breakdown": breakdown,
        }
        return obs, reward, terminated, truncated, info


def wait_for_isaac(port: int = 9500, **kwargs) -> SocketIsaacEnv:
    return SocketIsaacEnv(port=port, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    env = wait_for_isaac(port=args.port)
    obs, info = env.reset()
    log.info("initial obs keys: %s", sorted(obs.keys()))
    log.info("seed: %s", info.get("seed"))

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    ep_reward = 0.0
    for i in range(args.steps):
        a = rng.integers(low=0, high=ACTION_FACTORS)
        obs, r, term, trunc, info = env.step(a)
        ep_reward += r
        if i % 100 == 0:
            hz = (i + 1) / max(time.perf_counter() - t0, 1e-6)
            log.info("step %d @ %.1f Hz — hp_red=%.0f  ep_reward=%.2f", i, hz, obs["player"][4], ep_reward)
        if term or trunc:
            log.info("episode ended (term=%s trunc=%s) reward=%.2f", term, trunc, ep_reward)
            obs, info = env.reset()
            ep_reward = 0.0
    env.close()
