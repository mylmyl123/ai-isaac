"""Diagnostic recorder for the heuristic policy.

Logs every tick's decision to a JSONL file. Each line is one tick with:
  - timestamp
  - env_idx
  - room_index
  - player position (x, y) and velocity (vx, vy)
  - enemies visible (count + nearest position)
  - doors state (which slots are open/locked/etc)
  - chosen action (move, shoot)
  - decision branch taken (combat vs door-seek vs random)
  - locked door target (if any)
  - all other relevant state variables

Usage:
    from isaac_rl.debug_recorder import DebugRecorder
    rec = DebugRecorder(save_path="runs/debug_trace.jsonl", enabled=True)

Then in the heuristic's act():
    rec.log_tick(env_idx, raw_obs, chosen_action, branch, extra_state)

To activate during training, set env var:
    ISAAC_HEURISTIC_DEBUG=1                     # Windows: $env:ISAAC_HEURISTIC_DEBUG="1"

The recorder auto-saves to runs/<run_name>/heuristic_debug.jsonl.

After the run, upload that file and I'll analyze the trace.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DebugRecorder:
    """Thread-safe JSONL recorder for heuristic debug traces.

    Buffers ticks in memory and flushes to disk periodically to keep I/O
    off the hot path. Rotates the buffer size to bound memory.
    """

    _instance: "DebugRecorder | None" = None

    def __init__(
        self,
        save_path: str | Path | None = None,
        enabled: bool = False,
        flush_every: int = 500,
        max_buffer: int = 10_000,
    ):
        self.save_path = Path(save_path) if save_path else None
        self.enabled = enabled and (self.save_path is not None)
        self.flush_every = flush_every
        self.max_buffer = max_buffer
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._tick_count = 0
        self._start_wall = time.time()
        if self.enabled and self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate any existing file so a new run starts fresh.
            with open(self.save_path, "w") as f:
                f.write("")   # empty file
            log.info("[debug_recorder] recording heuristic trace -> %s", self.save_path)

    @classmethod
    def get_instance(cls) -> "DebugRecorder | None":
        return cls._instance

    @classmethod
    def set_instance(cls, inst: "DebugRecorder") -> None:
        cls._instance = inst

    def log_tick(
        self,
        env_idx: int,
        raw_obs: dict[str, Any],
        action: Any,
        branch: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record one tick. Non-blocking; flushes when buffer fills."""
        if not self.enabled:
            return
        # Extract compact snapshot from raw_obs.
        player = raw_obs.get("player") or {}
        gg = raw_obs.get("global") or {}
        enemies = raw_obs.get("enemies") or {}
        doors = raw_obs.get("doors") or []
        room_bounds = raw_obs.get("room_bounds") or {}

        mask = enemies.get("mask") or []
        feats = enemies.get("feats") or []
        n_enemies = sum(1 for m in mask if m)
        nearest_enemy: dict[str, float] | None = None
        best_d = float("inf")
        for i, f in enumerate(feats):
            if i >= len(mask) or not mask[i] or not f or len(f) < 4:
                continue
            dx = float(f[2]) * 480.0
            dy = float(f[3]) * 270.0
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best_d = d
                nearest_enemy = {"dx": round(dx, 1), "dy": round(dy, 1), "dist": round(d, 1)}

        # Compact door state: 4 slots, each as [exists, open, locked, boss, treasure, secret].
        door_summary: list[str] = []
        for slot in range(min(4, len(doors))):
            d = doors[slot] or []
            if len(d) < 6:
                door_summary.append("--")
                continue
            flags = []
            if d[0]: flags.append("E")
            if d[1]: flags.append("O")
            if d[2]: flags.append("L")
            if d[3]: flags.append("B")
            if d[4]: flags.append("T")
            if d[5]: flags.append("S")
            door_summary.append("".join(flags) if flags else "-")
        slot_names = ["L", "U", "R", "D"]
        doors_str = " ".join(f"{slot_names[i]}={door_summary[i]}" for i in range(len(door_summary)))

        tick_data = {
            "t": round(time.time() - self._start_wall, 3),
            "env": env_idx,
            "room": gg.get("room_index"),
            "clear": bool(gg.get("is_clear")),
            "px": round(float(player.get("x", 0) or 0), 1),
            "py": round(float(player.get("y", 0) or 0), 1),
            "vx": round(float(player.get("vx", 0) or 0), 2),
            "vy": round(float(player.get("vy", 0) or 0), 2),
            "hp": player.get("hp_red"),
            "n_enemies": n_enemies,
            "nearest_enemy": nearest_enemy,
            "doors": doors_str,
            "bounds": {
                "tl": (round(float(room_bounds.get("tl_x", 0) or 0), 0),
                       round(float(room_bounds.get("tl_y", 0) or 0), 0)),
                "br": (round(float(room_bounds.get("br_x", 0) or 0), 0),
                       round(float(room_bounds.get("br_y", 0) or 0), 0)),
            },
            "move": int(action[0]) if hasattr(action, "__getitem__") else None,
            "shoot": int(action[1]) if hasattr(action, "__getitem__") and len(action) > 1 else None,
            "branch": branch,
        }
        if extra:
            tick_data["extra"] = extra

        with self._lock:
            self._buffer.append(tick_data)
            self._tick_count += 1
            need_flush = len(self._buffer) >= self.flush_every
            if len(self._buffer) > self.max_buffer:
                # Emergency drop of the oldest half to bound memory.
                self._buffer = self._buffer[-(self.max_buffer // 2):]
        if need_flush:
            self.flush()

    def flush(self) -> None:
        """Write buffered ticks to disk as JSONL."""
        if not self.enabled or not self.save_path:
            return
        with self._lock:
            to_write = self._buffer
            self._buffer = []
        if not to_write:
            return
        try:
            with open(self.save_path, "a") as f:
                for tick in to_write:
                    f.write(json.dumps(tick) + "\n")
        except OSError as e:
            log.warning("[debug_recorder] write failed: %s", e)

    def close(self) -> None:
        self.flush()
        log.info("[debug_recorder] wrote %d ticks to %s", self._tick_count, self.save_path)


def get_or_create(save_path: str | Path | None = None) -> DebugRecorder | None:
    """Get the singleton recorder, creating it if the env var enables it."""
    inst = DebugRecorder.get_instance()
    if inst is not None:
        return inst
    enabled = bool(os.environ.get("ISAAC_HEURISTIC_DEBUG", "").strip())
    if not enabled:
        return None
    inst = DebugRecorder(save_path=save_path, enabled=True)
    DebugRecorder.set_instance(inst)
    return inst
