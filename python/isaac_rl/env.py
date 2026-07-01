"""Gymnasium environment that talks to one Isaac process over a TCP socket.

Design notes:
- The trainer owns the server socket. Isaac connects into us via LuaSocket.
- One env == one Isaac process. Multi-env parallelism is in vec_env.py.
- `reset()` sends a `reset` command down the wire; Lua runs `restart 0` on the
  next tick and reconnects on MC_POST_GAME_STARTED.
- `step()` is synchronous: send action → wait for next obs frame → shape reward.
- If Isaac dies mid-step (socket closed), we invoke the crash callback
  registered for our port (usually IsaacFleet.respawn), wait for the new
  Isaac to reconnect on the same server socket, and surface the crash to the
  trainer as terminated=True with a small penalty so PPO gracefully starts a
  fresh episode. Training then continues without human intervention.
"""
from __future__ import annotations

import logging
import socket
import time
from typing import Any, Callable

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


# Per-port crash callback registry. train.py fills this with the IsaacFleet's
# respawn method for every port it manages, BEFORE build_vec_env constructs the
# SocketIsaacEnvs. When an env detects its Isaac has died it looks up the port
# here and invokes the callback. Decoupling this from the env constructor lets
# build_vec_env stay callback-agnostic.
_ON_CRASH: dict[int, Callable[[int], None]] = {}


def register_on_crash(port: int, cb: Callable[[int], None]) -> None:
    """Register a callable(port) -> None that respawns Isaac for the given port.

    The callback must be idempotent w.r.t. leftover Isaac processes on that
    port (i.e. it should kill any zombie child before launching a new one).
    """
    _ON_CRASH[port] = cb


def _crash_penalty_obs(port: int) -> dict[str, Any]:
    """Build a minimal raw-obs dict for a crash-induced terminal step.

    We surface a synthetic 'crash' event so RewardShaper can penalize + terminate
    without needing the mod to have sent us anything. The obs itself is
    zero-filled downstream by encode_obs when fields are missing.
    """
    return {
        "schema": 1,
        "tick": 0,
        "player": {"is_dead": True, "hp_red": 0},
        "events": [{"kind": "crash", "port": port}],
    }


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
        # Short timeout so KeyboardInterrupt is observable while we wait for Isaac to connect.
        # Total wait budget is self.accept_timeout_s.
        import time as _time
        deadline = _time.time() + self.accept_timeout_s
        self._server.settimeout(1.0)
        while True:
            try:
                client, addr = self._server.accept()
                break
            except socket.timeout:
                if _time.time() > deadline:
                    raise TimeoutError(f"Isaac did not connect on port {self.port} within {self.accept_timeout_s}s")
                continue
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Bump the OS recv buffer to 1 MB. During training the trainer stops
        # reading from the socket for a few seconds while it runs a PPO update
        # (no env.step() calls). With the default ~64KB recv buffer, Isaacs
        # mod fills it in ~2 seconds at 15 obs/sec, TCP window closes, mods
        # sock:send() times out with a PARTIAL write, and the framing on the
        # wire is now corrupted for good. Symptom: mysterious 'Isaac died
        # mid-step [WinError 10054]' warnings. 1MB buys ~30s of PPO-update
        # headroom, well beyond any reasonable update duration.
        try:
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError as _e:
            log.warning("port %d: failed to set SO_RCVBUF=1MB: %s (falling back to OS default)", self.port, _e)
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
            try:
                raw = recv_frame(self._client)
            except (ConnectionError, OSError) as e:
                log.warning("port %d: crash during initial obs (%s) — respawning", self.port, e)
                raw = self._handle_crash_and_reaccept()
        else:
            payload: dict[str, Any] = {"reset": True}
            if seed is not None:
                payload["seed"] = int(seed)
            if self.reset_stage is not None:
                payload["stage"] = int(self.reset_stage)
            try:
                send_frame(self._client, payload)
            except (ConnectionError, OSError) as e:
                log.warning("port %d: crash while sending reset (%s) — respawning", self.port, e)
                raw = self._handle_crash_and_reaccept()
            else:
                try:
                    self._client.close()
                except OSError:
                    pass
                self._client = None
                self._accept()
                try:
                    raw = recv_frame(self._client)
                except (ConnectionError, OSError) as e:
                    log.warning("port %d: crash during post-reset obs (%s) — respawning", self.port, e)
                    raw = self._handle_crash_and_reaccept()

        self._steps = 0
        self._last_action[:] = 0
        obs = encode_obs(raw, last_action=self._last_action)
        info: dict[str, Any] = {"seed": self._last_seed, "raw": raw}
        return obs, info

    def step(self, action):
        assert self._client is not None, "reset() must be called before step()"
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        try:
            send_frame(self._client, encode_action(a))
            raw = recv_frame(self._client)
        except (ConnectionError, OSError) as e:
            log.warning("port %d: Isaac died mid-step (%s) — respawning", self.port, e)
            # Kick off respawn and wait for the new Isaac to reconnect so the
            # NEXT reset() (which the trainer will call because terminated=True)
            # can just read the handshake+obs immediately.
            self._handle_crash_and_reaccept(read_first_obs=False)
            # Terminal step with penalty. Rewards from the shaper are optional
            # here; we hardcode a fixed penalty so training sees a clear signal
            # 'don't do whatever led to this'. It's small enough not to dominate.
            self._last_action = a
            self._steps += 1
            obs = encode_obs(_crash_penalty_obs(self.port), last_action=self._last_action)
            info: dict[str, Any] = {
                "raw": _crash_penalty_obs(self.port),
                "steps": self._steps,
                "reward_breakdown": {"crash_penalty": -1.0},
                "crashed": True,
            }
            return obs, -1.0, True, False, info

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

    def _handle_crash_and_reaccept(self, read_first_obs: bool = True) -> dict[str, Any]:
        """Respawn Isaac and re-accept its connection on our server socket.

        Called when a recv/send on the client socket raises ConnectionError /
        OSError. Returns the raw obs dict from the new Isaac's first exchange,
        or a synthetic crash-terminal dict if read_first_obs=False (or if the
        respawn itself failed after all retries).

        This method must NEVER propagate an exception — doing so would kill the
        whole training loop the moment a single Isaac has a bad boot. On total
        failure we return a crash-obs and leave self._client = None so the
        trainer's next reset() attempts another respawn.
        """
        # Close the (already-dead) client socket.
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None

        MAX_RESPAWN_ATTEMPTS = 3
        for attempt in range(1, MAX_RESPAWN_ATTEMPTS + 1):
            cb = _ON_CRASH.get(self.port)
            if cb is None:
                log.warning(
                    "port %d: no on_crash callback registered; waiting for manual Isaac restart",
                    self.port,
                )
            else:
                try:
                    cb(self.port)
                except Exception as e:
                    log.exception("port %d: respawn callback failed: %s", self.port, e)

            # Try to accept the new Isaac's connection + handshake. _accept()
            # blocks on the server socket up to self.accept_timeout_s (default
            # 300s), then reads the handshake frame.
            try:
                self._accept()
            except (ConnectionError, OSError, TimeoutError) as e:
                # Includes ConnectionResetError (WinError 10054) when Isaac
                # accepts the TCP handshake and then dies before sending the
                # bridge handshake frame. Try again.
                log.warning(
                    "port %d: respawn attempt %d/%d failed at accept/handshake: %s",
                    self.port, attempt, MAX_RESPAWN_ATTEMPTS, e,
                )
                if self._client is not None:
                    try:
                        self._client.close()
                    except OSError:
                        pass
                    self._client = None
                # Small backoff before firing the callback again so Isaac's
                # per-process cleanup can settle.
                time.sleep(2.0)
                continue

            # Accept + handshake succeeded. Optionally read the first obs.
            if not read_first_obs:
                return _crash_penalty_obs(self.port)
            try:
                return recv_frame(self._client)
            except (ConnectionError, OSError) as e:
                log.warning(
                    "port %d: respawned Isaac died during first obs (%s); attempt %d/%d",
                    self.port, e, attempt, MAX_RESPAWN_ATTEMPTS,
                )
                if self._client is not None:
                    try:
                        self._client.close()
                    except OSError:
                        pass
                    self._client = None
                time.sleep(2.0)
                continue

        # All retries exhausted. Leave client=None so the trainer's next reset()
        # will trigger another respawn cycle. Surface a crash-terminal so the
        # current step/reset returns a valid obs shape.
        log.error(
            "port %d: %d respawn attempts all failed; returning crash-terminal, will retry on next reset",
            self.port, MAX_RESPAWN_ATTEMPTS,
        )
        return _crash_penalty_obs(self.port)


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
