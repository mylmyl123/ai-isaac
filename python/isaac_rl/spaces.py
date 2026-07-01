"""Observation / action spaces for the Isaac RL bridge.

Keep this in sync with `mods/isaac-rl-bridge/obs.lua`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


SCHEMA_VERSION = 1

# MultiDiscrete factors — mirrored in mods/isaac-rl-bridge/main.lua apply_action().
ACTION_FACTORS = np.array([9, 5, 2, 2, 2], dtype=np.int64)
ACTION_KEYS = ("move", "shoot", "use_active", "drop_bomb", "pill_card")


def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(ACTION_FACTORS)


ROOM_H, ROOM_W = 9, 15
MAX_ENEMIES = 24
MAX_PROJECTILES = 48
MAX_PICKUPS = 16

PLAYER_DIM = 40
GLOBAL_DIM = 20
ENEMY_FEATS = 16
PROJ_FEATS = 10
PICKUP_FEATS = 8

PASSIVES_K = 256


def observation_space() -> spaces.Dict:
    return spaces.Dict({
        "player":     spaces.Box(-np.inf, np.inf, shape=(PLAYER_DIM,), dtype=np.float32),
        "passives":   spaces.MultiBinary(PASSIVES_K),
        "room_grid":  spaces.Box(0.0, 1.0, shape=(4, ROOM_H, ROOM_W), dtype=np.float32),
        "doors":      spaces.Box(0.0, 1.0, shape=(4, 6), dtype=np.float32),
        "enemies": spaces.Dict({
            "feats": spaces.Box(-np.inf, np.inf, shape=(MAX_ENEMIES, ENEMY_FEATS), dtype=np.float32),
            "mask":  spaces.MultiBinary(MAX_ENEMIES),
        }),
        "projectiles": spaces.Dict({
            "feats": spaces.Box(-np.inf, np.inf, shape=(MAX_PROJECTILES, PROJ_FEATS), dtype=np.float32),
            "mask":  spaces.MultiBinary(MAX_PROJECTILES),
        }),
        "pickups": spaces.Dict({
            "feats": spaces.Box(-np.inf, np.inf, shape=(MAX_PICKUPS, PICKUP_FEATS), dtype=np.float32),
            "mask":  spaces.MultiBinary(MAX_PICKUPS),
        }),
        "global":      spaces.Box(-np.inf, np.inf, shape=(GLOBAL_DIM,), dtype=np.float32),
        "last_action": spaces.Box(0.0, 1.0, shape=(len(ACTION_FACTORS),), dtype=np.float32),
    })


def zero_obs() -> dict[str, Any]:
    space = observation_space()

    def make(sp):
        if isinstance(sp, spaces.Dict):
            return {k: make(v) for k, v in sp.spaces.items()}
        if isinstance(sp, spaces.MultiBinary):
            return np.zeros(sp.shape, dtype=np.int8)
        return np.zeros(sp.shape, dtype=sp.dtype)

    return make(space)


# --- Decode helpers -------------------------------------------------------

_PLAYER_FIELDS = (
    "x", "y", "vx", "vy",
    "hp_red", "hp_soul", "hp_black", "hp_max",
    "keys", "bombs", "coins",
    "damage", "fire_delay", "move_speed", "tear_range", "shot_speed", "luck",
    "can_shoot", "frame_count", "is_dead",
)

_GLOBAL_FIELDS = (
    "stage", "stage_type", "room_index", "safe_grid_index",
    "room_type", "is_clear", "curses",
    "frames_since_room", "frames_since_hit", "visited_rooms",
)


def _copy_entity_feats(raw_group: dict | None, max_n: int, feat_dim: int):
    """Common decoder for enemies/projectiles/pickups."""
    feats = np.zeros((max_n, feat_dim), dtype=np.float32)
    mask = np.zeros(max_n, dtype=np.int8)
    if not raw_group:
        return feats, mask
    rows = raw_group.get("feats") or []
    m = raw_group.get("mask") or []
    for i, row in enumerate(rows[:max_n]):
        if not row:
            continue
        vals = [float(v or 0) for v in row[:feat_dim]]
        feats[i, :len(vals)] = vals
        mask[i] = int(m[i]) if i < len(m) else 1
    return feats, mask


def _decode_room_grid(raw: dict | None) -> np.ndarray:
    grid = np.zeros((4, ROOM_H, ROOM_W), dtype=np.float32)
    if not raw:
        return grid
    for ch, key in enumerate(("walls", "rocks", "spikes", "poop")):
        arr = raw.get(key) or []
        # Lua sends a flat row-major array of length H*W.
        n = min(len(arr), ROOM_H * ROOM_W)
        if n:
            grid[ch].reshape(-1)[:n] = np.asarray(arr[:n], dtype=np.float32)
    return grid


def _decode_doors(raw: list | None) -> np.ndarray:
    out = np.zeros((4, 6), dtype=np.float32)
    if not raw:
        return out
    for i in range(min(4, len(raw))):
        row = raw[i] or [0, 0, 0, 0, 0, 0]
        for j in range(min(6, len(row))):
            out[i, j] = float(row[j] or 0)
    return out


def _decode_passives(raw: list | None) -> np.ndarray:
    out = np.zeros(PASSIVES_K, dtype=np.int8)
    if not raw:
        return out
    for idx in raw:
        # Lua sends 1-based indices; convert to 0-based and clip.
        try:
            i = int(idx) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < PASSIVES_K:
            out[i] = 1
    return out


def encode_obs(raw: dict[str, Any], last_action: np.ndarray | None = None) -> dict[str, Any]:
    """Convert a JSON obs dict from Lua into a gym Dict observation."""
    obs = zero_obs()

    p = raw.get("player") or {}
    for i, name in enumerate(_PLAYER_FIELDS):
        if i >= PLAYER_DIM:
            break
        v = p.get(name, 0)
        obs["player"][i] = float(bool(v)) if isinstance(v, bool) else float(v or 0)

    g = raw.get("global") or {}
    for i, name in enumerate(_GLOBAL_FIELDS):
        if i >= GLOBAL_DIM:
            break
        v = g.get(name, 0)
        obs["global"][i] = float(bool(v)) if isinstance(v, bool) else float(v or 0)

    obs["passives"] = _decode_passives(raw.get("passives"))
    obs["room_grid"] = _decode_room_grid(raw.get("room_grid"))
    obs["doors"] = _decode_doors(raw.get("doors"))

    for key, dim, feat_dim in [
        ("enemies", MAX_ENEMIES, ENEMY_FEATS),
        ("projectiles", MAX_PROJECTILES, PROJ_FEATS),
        ("pickups", MAX_PICKUPS, PICKUP_FEATS),
    ]:
        feats, mask = _copy_entity_feats(raw.get(key), dim, feat_dim)
        obs[key] = {"feats": feats, "mask": mask}

    if last_action is not None:
        denom = np.maximum(ACTION_FACTORS - 1, 1).astype(np.float32)
        obs["last_action"][:] = np.asarray(last_action, dtype=np.float32) / denom

    return obs


def encode_action(action: np.ndarray | list[int]) -> dict[str, int]:
    a = np.asarray(action, dtype=np.int64).reshape(-1)
    return {ACTION_KEYS[i]: int(a[i]) for i in range(len(ACTION_KEYS))}


def flatten_dict_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return the same dict with nested obs['enemies']['feats'] etc. exposed as flat keys.

    Convenient for batching into torch tensors — the trainer just concatenates by key.
    """
    out: dict[str, np.ndarray] = {
        "player": obs["player"],
        "passives": obs["passives"].astype(np.float32),
        "room_grid": obs["room_grid"],
        "doors": obs["doors"],
        "global": obs["global"],
        "last_action": obs["last_action"],
    }
    for key in ("enemies", "projectiles", "pickups"):
        out[f"{key}_feats"] = obs[key]["feats"]
        out[f"{key}_mask"] = obs[key]["mask"].astype(np.float32)
    return out
