"""Observation / action spaces for the Isaac RL bridge.

Split off from env.py so the schema is importable from tests and the trainer without
starting a socket server. Keep this file in sync with `mods/isaac-rl-bridge/obs.lua`.

M1 covers only the scalar player/global fields. Entity/projectile/grid tensors will
be added as the Lua obs builder grows past its placeholder tables.
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


# Room interior is 9x15 tiles; grid stays as a placeholder until obs.lua fills it.
ROOM_H, ROOM_W = 9, 15
MAX_ENEMIES = 24
MAX_PROJECTILES = 48
MAX_PICKUPS = 16

ENEMY_FEATS = 16
PROJ_FEATS = 10
PICKUP_FEATS = 8

PASSIVES_K = 256


def observation_space() -> spaces.Dict:
    return spaces.Dict({
        "player":     spaces.Box(-np.inf, np.inf, shape=(40,), dtype=np.float32),
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
        "global":      spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32),
        "last_action": spaces.Box(0.0, 1.0, shape=(len(ACTION_FACTORS),), dtype=np.float32),
    })


def zero_obs() -> dict[str, Any]:
    """Return a valid all-zeros observation. Used to pad any field the Lua side hasn't sent yet."""
    space = observation_space()

    def make(sp):
        if isinstance(sp, spaces.Dict):
            return {k: make(v) for k, v in sp.spaces.items()}
        if isinstance(sp, spaces.MultiBinary):
            return np.zeros(sp.shape, dtype=np.int8)
        return np.zeros(sp.shape, dtype=sp.dtype)

    return make(space)


# --- Encoders that convert the Lua JSON dict into the Dict observation ------

_PLAYER_FIELDS = (
    "x", "y", "vx", "vy",
    "hp_red", "hp_soul", "hp_black", "hp_max",
    "keys", "bombs", "coins",
    "damage", "fire_delay", "move_speed", "tear_range", "shot_speed", "luck",
    "can_shoot", "frame_count",
)

_GLOBAL_FIELDS = (
    "stage", "stage_type", "room_index", "room_type", "is_clear", "curses",
)


def encode_obs(raw: dict[str, Any], last_action: np.ndarray | None = None) -> dict[str, Any]:
    """Convert a JSON obs dict from Lua into a gym Dict observation.

    Missing fields are zero-filled. `last_action` is normalized to [0, 1] against ACTION_FACTORS - 1.
    """
    obs = zero_obs()

    p = raw.get("player") or {}
    for i, name in enumerate(_PLAYER_FIELDS):
        if i >= obs["player"].shape[0]:
            break
        obs["player"][i] = float(p.get(name, 0) or 0)

    g = raw.get("global") or {}
    for i, name in enumerate(_GLOBAL_FIELDS):
        if i >= obs["global"].shape[0]:
            break
        obs["global"][i] = float(g.get(name, 0) or 0)

    if last_action is not None:
        denom = np.maximum(ACTION_FACTORS - 1, 1).astype(np.float32)
        obs["last_action"][:] = np.asarray(last_action, dtype=np.float32) / denom

    return obs


def encode_action(action: np.ndarray | list[int]) -> dict[str, int]:
    """Convert a MultiDiscrete action array into the JSON action dict Lua expects."""
    a = np.asarray(action, dtype=np.int64).reshape(-1)
    return {ACTION_KEYS[i]: int(a[i]) for i in range(len(ACTION_KEYS))}
