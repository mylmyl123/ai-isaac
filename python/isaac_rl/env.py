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
    HISTORY_FEATS,
    HISTORY_FRAMES,
    PLAYER_HISTORY_DIM,
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
        env_idx: int = 0,                 # position in the vec-env (0-indexed)
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.env_idx = env_idx
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
        # Per-episode reward-breakdown accumulator. Reset in reset(); summed
        # each step from the shaper's per-step breakdown. Passed back to the
        # trainer on the terminal step as `reward_breakdown_episode` so we
        # can log the TRUE episode-total contribution of each reward
        # component. Prior to 2026-07-08 we logged only the terminal-tick
        # breakdown, which hid every non-terminal reward event (kill,
        # damage_dealt, new_room, room_clear, pickup_*, etc.) from TB.
        self._episode_breakdown: dict[str, float] = {}
        # Frame stacking: rolling buffer of past player states for the
        # "player_history" obs field. Shape [HISTORY_FRAMES, HISTORY_FEATS]:
        # each row is [nx, ny, vx, vy] normalised. Newest frame at index -1.
        # Zeroed on reset. Updated each step from the incoming obs's player
        # position + velocity.
        self._player_history = np.zeros((HISTORY_FRAMES, HISTORY_FEATS), dtype=np.float32)
        # B4: Per-episode latent variable z ~ N(0, I). Sampled at reset,
        # constant for the whole episode. Encourages strategic diversity.
        # z_dim=0 disables (self._z remains a zero vector).
        self._z_dim = 16
        self._z = np.zeros(self._z_dim, dtype=np.float32)

        self.reward_shaper = RewardShaper(reward_config)

        self._open_server()

    # -- observation helpers ---------------------------------------------

    def _update_player_history(self, raw: dict, is_reset: bool = False) -> None:
        """Push a new player state into the rolling history buffer.

        Frame stacking (added 2026-07-02). Provides short-term motion context
        beyond what the GRU's internal state already captures — explicitly
        redundant, but helps BC learn dynamics from small demo sets.

        On reset, buffer is zeroed BEFORE recording the initial frame so that
        all 4 slots hold the initial state (avoids startup transients).
        """
        player = raw.get("player") or {}
        bounds = raw.get("room_bounds") or {}
        tl_x = float(bounds.get("tl_x", 0) or 0)
        tl_y = float(bounds.get("tl_y", 0) or 0)
        br_x = float(bounds.get("br_x", 1) or 1)
        br_y = float(bounds.get("br_y", 1) or 1)
        width = max(1.0, br_x - tl_x)
        height = max(1.0, br_y - tl_y)
        px = float(player.get("x", 0) or 0)
        py = float(player.get("y", 0) or 0)
        nx = 2.0 * (px - tl_x) / width - 1.0
        ny = 2.0 * (py - tl_y) / height - 1.0
        vx = float(player.get("vx", 0) or 0) / 10.0
        vy = float(player.get("vy", 0) or 0) / 10.0
        new_frame = np.array([nx, ny, vx, vy], dtype=np.float32)

        if is_reset:
            # Broadcast: fill the whole history with the initial state.
            self._player_history[:] = new_frame[None, :]
        else:
            # Shift oldest out (index 0), append newest at index -1.
            self._player_history = np.roll(self._player_history, -1, axis=0)
            self._player_history[-1] = new_frame

    def _build_obs(self, raw: dict) -> dict[str, Any]:
        """encode_obs + inject frame-stacked player_history and latent z."""
        obs = encode_obs(raw, last_action=self._last_action)
        obs["player_history"] = self._player_history.reshape(-1).copy()
        obs["z"] = self._z.copy()   # B4: episode-level latent
        return obs

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
        self._episode_breakdown = {}
        self._last_action[:] = 0
        # B4: Sample new latent z for this episode.
        if self._z_dim > 0:
            self._z = np.random.randn(self._z_dim).astype(np.float32)
        # Reset frame-stacking buffer to the initial player state
        # (broadcasts across all 4 slots so no startup transient).
        self._update_player_history(raw, is_reset=True)
        obs = self._build_obs(raw)
        info: dict[str, Any] = {"seed": self._last_seed, "raw": raw}
        return obs, info

    def step(self, action):
        assert self._client is not None, "reset() must be called before step()"
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        # Apply human override if enabled AND this env is the current target.
        try:
            from isaac_rl.human_override import apply_override
            a = apply_override(a, env_idx=self.env_idx)
        except ImportError:
            pass
        try:
            send_frame(self._client, encode_action(a))
            raw = recv_frame(self._client)
        except (ConnectionError, OSError) as e:
            # 2026-07-09 REDESIGN: split "mod cycled socket cleanly" from
            # "Isaac process actually crashed". Prior code applied a fixed
            # -1 crash_penalty on BOTH cases, which produced a ~-0.9/ep
            # baseline drag regardless of policy quality (measured on the
            # 2026-07-08 15h run: 90% of episodes ended via this path with
            # crash_penalty firing). Now:
            #   - mod cycled socket → the mod deliberately closed to restart
            #     mid-run on player death. Isaac is alive and will reconnect
            #     in <500 ms. HP-based death detection in the shaper (commit
            #     8d72114) has already terminated the episode with r_death.
            #     Applying an extra -1 on top is double-counting the death.
            #     Return terminated=True with NO extra penalty.
            #   - Isaac actually crashed → respawn path, apply -1 crash_penalty
            #     as before. This is the rare true-crash case (SIGSEGV,
            #     out-of-memory, etc.).
            log.warning(
                "port %d: connection interrupted mid-step (%s: %s) — checking if mod restart or real crash",
                self.port, type(e).__name__, e,
            )
            reconnected_raw = self._try_accept_after_close(wait_s=3.0)
            self._last_action = a
            self._steps += 1
            if reconnected_raw is not None:
                # Mod cycled its socket cleanly — in-process restart on death.
                # Terminate the episode with 0 penalty; the shaper's HP-based
                # death detection has already applied r_death on the last
                # obs where HP transitioned to 0.
                log.info("port %d: mod cycled socket (expected on death) — terminating cleanly", self.port)
                obs = self._build_obs(reconnected_raw)
                info: dict[str, Any] = {
                    "raw": reconnected_raw,
                    "steps": self._steps,
                    "reward_breakdown": {},
                    "reward_breakdown_episode": dict(self._episode_breakdown),
                    "crashed": False,
                    "ep_end_reason": "mod_restart",
                    "behavior_metrics": self.reward_shaper.episode_behavior_metrics(),
                }
                # NOTE: reconnected_raw came from a FRESH mod (post-restart),
                # so it's a valid initial obs for the next episode. Store it
                # so env.reset() doesn't re-accept and lose the frame.
                # (env.reset() checks self._client is not None; we've already
                # set it via _try_accept_after_close.)
                return obs, 0.0, True, False, info
            # Real crash: no reconnection. Fall through to respawn.
            log.warning(
                "port %d: no reconnection within 3s — assuming real crash, respawning",
                self.port,
            )
            self._handle_crash_and_reaccept(read_first_obs=False)
            crash_raw = _crash_penalty_obs(self.port)
            # Advance frame-stack even on crash to avoid stale history.
            self._update_player_history(crash_raw)
            obs = self._build_obs(crash_raw)
            info: dict[str, Any] = {
                "raw": _crash_penalty_obs(self.port),
                "steps": self._steps,
                "reward_breakdown": {"crash_penalty": -1.0},
                "crashed": True,
                "ep_end_reason": "isaac_crash",
                "behavior_metrics": self.reward_shaper.episode_behavior_metrics(),
            }
            # Fold crash_penalty into the per-episode running sum, then emit
            # the whole episode-total breakdown.
            self._episode_breakdown["crash_penalty"] = (
                self._episode_breakdown.get("crash_penalty", 0.0) - 1.0
            )
            info["reward_breakdown_episode"] = dict(self._episode_breakdown)
            return obs, -1.0, True, False, info

        self._last_action = a
        self._steps += 1
        self._update_player_history(raw)
        obs = self._build_obs(raw)

        reward, terminated, breakdown = self.reward_shaper(raw, action=a)
        # Accumulate the per-step breakdown into a per-episode running sum.
        # This is the source of truth for TB reward/{k} logging — the shaper
        # emits breakdown PER TICK, so summing over the episode gives the
        # actual total contribution of each key.
        for k, v in breakdown.items():
            self._episode_breakdown[k] = self._episode_breakdown.get(k, 0.0) + float(v)
        truncated = self._steps >= self.max_steps
        info: dict[str, Any] = {
            "raw": raw,
            "steps": self._steps,
            "reward_breakdown": breakdown,
        }
        # Tag the episode-end reason so trainers can log crash-vs-death rates.
        # This is what tells us at a glance whether the socket layer is
        # working: high `mod_socket_error` frac = window backgrounded /
        # throttled / mod crashed. High `shaper_terminated` frac = normal.
        if terminated:
            info["ep_end_reason"] = "shaper_terminated"
            info["reward_breakdown_episode"] = dict(self._episode_breakdown)
            # Behavior metrics (Phase C, 2026-07-09): pure telemetry, not
            # rewards. Trainer logs these under behavior/* so we can see
            # whether the agent is starting to demonstrate emergent
            # hierarchical play (visit shops, use items, reach later
            # floors) even where we haven't explicitly rewarded it.
            info["behavior_metrics"] = self.reward_shaper.episode_behavior_metrics()
        elif truncated:
            info["ep_end_reason"] = "truncated"
            info["reward_breakdown_episode"] = dict(self._episode_breakdown)
            info["behavior_metrics"] = self.reward_shaper.episode_behavior_metrics()
        return obs, reward, terminated, truncated, info

    def _try_accept_after_close(self, wait_s: float = 3.0) -> dict[str, Any] | None:
        """Try to accept a NEW client connection after the current one closed.

        Isaacs mod closes its socket during in-process restarts (character
        died -> mod runs `restart` -> MC_POST_GAME_STARTED closes+reconnects
        the socket). The mod is STILL ALIVE and will reconnect within a few
        game frames — typically 100-500ms. If we treat that socket close as
        a crash and call fleet.respawn(), we KILL the alive Isaac process.

        This helper tries to accept a new client on our server socket with a
        short timeout. If Isaac reconnects within wait_s, it means the mod
        did an in-process restart and were golden — just read the handshake
        + first obs and return. Return None if no reconnection happens in
        time (real crash, caller falls back to fleet.respawn).
        """
        assert self._server is not None
        import time as _time
        deadline = _time.time() + wait_s
        self._server.settimeout(0.5)
        while _time.time() < deadline:
            try:
                client, addr = self._server.accept()
            except socket.timeout:
                continue
            except (ConnectionError, OSError):
                return None
            # New connection arrived — configure it, read handshake + obs.
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            except OSError:
                pass
            try:
                hello = recv_frame(client)
                self._last_seed = hello.get("seed")
                log.info("port %d: mod reconnected (in-process restart), handshake: %s", self.port, hello)
                raw = recv_frame(client)
            except (ConnectionError, OSError) as e:
                log.warning("port %d: reconnected client failed handshake: %s", self.port, e)
                try: client.close()
                except OSError: pass
                return None
            self._client = client
            return raw
        return None

    def _handle_crash_and_reaccept(self, read_first_obs: bool = True) -> dict[str, Any]:
        """Respawn Isaac and re-accept its connection on our server socket.

        Called when a recv/send on the client socket raises ConnectionError /
        OSError. First tries to detect an in-process restart (mod is alive,
        just cycled its socket) via _try_accept_after_close. Only if that
        fails does it actually kill Isaac and launch a fresh one.

        Returns the raw obs dict from the new Isaac's first exchange, or a
        synthetic crash-terminal dict if read_first_obs=False (or if the
        respawn itself failed after all retries).

        This method must NEVER propagate an exception — doing so would kill
        the whole training loop the moment a single Isaac has a bad boot. On
        total failure we return a crash-obs and leave self._client = None so
        the trainer's next reset() attempts another respawn.
        """
        # Close the (already-dead-from-our-perspective) client socket.
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None

        # First: is Isaac actually dead, or did the mod just cycle its socket
        # during an in-process restart? Wait a moment for a reconnection
        # BEFORE killing the process. If Isaac is fine and just restarted,
        # this saves us from murdering a healthy child process.
        raw = self._try_accept_after_close(wait_s=3.0)
        if raw is not None:
            log.info("port %d: skipping respawn — Isaac was alive and reconnected after mod restart", self.port)
            return raw

        log.warning("port %d: no reconnection within 3s — assuming real crash, respawning", self.port)

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
