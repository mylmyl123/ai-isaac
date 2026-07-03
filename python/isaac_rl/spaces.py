"""Observation / action spaces for the Isaac RL bridge.

Keep this in sync with `mods/isaac-rl-bridge/obs.lua`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


SCHEMA_VERSION = 2

# MultiDiscrete factors — mirrored in mods/isaac-rl-bridge/main.lua apply_action().
# Simplified 2026-07-02: removed use_active / drop_bomb / pill_card action heads
# (they were unused / harmful when triggered by random exploration — dropping
# a bomb hurts the player, using unknown pills is often negative). Reduces
# action space from 360 combos to 45 (8x smaller), speeds up convergence.
ACTION_FACTORS = np.array([9, 5], dtype=np.int64)
ACTION_KEYS = ("move", "shoot")


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

# Spatial obs (added schema v2). Preprocessed spatial features derived from
# player position within the room. Fed as a small dense vector; forces the
# network to reason about "where am I in this room" without needing to learn
# it from raw pixel coords. Features (in order):
#   [0, 1] player position normalized to [-1, 1] within room
#   [2, 3, 4, 5] normalized distance to each wall (left, up, right, down)
#   [6, 7] unit vector to nearest OPEN door, or (0, 0) if none open
SPATIAL_DIM = 8


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
        # Spatial features (added schema v2). See SPATIAL_DIM comment above.
        "spatial":    spaces.Box(-1.0, 1.0, shape=(SPATIAL_DIM,), dtype=np.float32),
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


def _compute_spatial(
    raw: dict[str, Any],
) -> np.ndarray:
    """Compute preprocessed spatial features (schema v2).

    Returns an (SPATIAL_DIM,) float32 vector:
      [0, 1] player position normalized to [-1, 1] within the room
      [2..5] normalized distance to each wall (L, U, R, D), clipped to [0, 1]
      [6, 7] unit vector to nearest OPEN door

    All features are zero if room_bounds is missing (backward-compatible with
    schema v1). The network's spatial_mlp will produce zero-mean features in
    that case and rely on the other obs paths.
    """
    out = np.zeros(SPATIAL_DIM, dtype=np.float32)

    bounds = raw.get("room_bounds")
    if not bounds:
        return out

    tl_x = float(bounds.get("tl_x", 0) or 0)
    tl_y = float(bounds.get("tl_y", 0) or 0)
    br_x = float(bounds.get("br_x", 1) or 1)
    br_y = float(bounds.get("br_y", 1) or 1)
    width = max(1.0, br_x - tl_x)
    height = max(1.0, br_y - tl_y)

    player = raw.get("player") or {}
    px = float(player.get("x", 0) or 0)
    py = float(player.get("y", 0) or 0)

    # Normalized position within room: -1 (top-left) to +1 (bottom-right).
    nx = 2.0 * (px - tl_x) / width - 1.0
    ny = 2.0 * (py - tl_y) / height - 1.0
    out[0] = np.clip(nx, -1.0, 1.0)
    out[1] = np.clip(ny, -1.0, 1.0)

    # Distance to each wall, normalized by room dimension.
    dl = np.clip((px - tl_x) / width, 0.0, 1.0)
    du = np.clip((py - tl_y) / height, 0.0, 1.0)
    dr = np.clip((br_x - px) / width, 0.0, 1.0)
    dd = np.clip((br_y - py) / height, 0.0, 1.0)
    out[2] = dl
    out[3] = du
    out[4] = dr
    out[5] = dd

    # Unit vector to nearest OPEN door. Door slots are LEFT, UP, RIGHT, DOWN.
    # Approximate door positions at wall midpoints. Filter for exists+open.
    doors = raw.get("doors") or []
    door_positions = [
        (tl_x, (tl_y + br_y) / 2.0),                  # LEFT
        ((tl_x + br_x) / 2.0, tl_y),                  # UP
        (br_x, (tl_y + br_y) / 2.0),                  # RIGHT
        ((tl_x + br_x) / 2.0, br_y),                  # DOWN
    ]
    best_dist = None
    best_dir = (0.0, 0.0)
    for slot in range(min(4, len(doors))):
        d = doors[slot]
        if not d or len(d) < 2:
            continue
        exists = bool(d[0])
        is_open = bool(d[1])
        if not exists or not is_open:
            continue
        dx, dy = door_positions[slot]
        vx = dx - px
        vy = dy - py
        dist = float(np.hypot(vx, vy))
        if dist < 1e-6:
            unit = (0.0, 0.0)
        else:
            unit = (vx / dist, vy / dist)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_dir = unit
    out[6] = best_dir[0]
    out[7] = best_dir[1]

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
    obs["spatial"] = _compute_spatial(raw)

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
        # Schema v2 addition. Backward-compat: older obs dicts without
        # "spatial" get zeros (matches _compute_spatial fallback behavior).
        "spatial": obs.get("spatial", np.zeros(SPATIAL_DIM, dtype=np.float32)),
    }
    for key in ("enemies", "projectiles", "pickups"):
        out[f"{key}_feats"] = obs[key]["feats"]
        out[f"{key}_mask"] = obs[key]["mask"].astype(np.float32)
    return out
