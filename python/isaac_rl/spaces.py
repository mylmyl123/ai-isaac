"""Observation / action spaces for the Isaac RL bridge.

Keep this in sync with `mods/isaac-rl-bridge/obs.lua`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


SCHEMA_VERSION = 2

# MultiDiscrete factors — mirrored in mods/isaac-rl-bridge/main.lua apply_action().
# 2026-07-02 REV: simplified from [9, 5, 2, 2, 2] to [9, 5] because random-init
#   exploration used the extra factors harmfully (bomb-drops damaged the agent,
#   random pill use was often negative).
# 2026-07-12 REV: RESTORED to [9, 5, 2, 2, 2] for Track A / BC-bootstrap. Human
#   demos use these factors purposefully; the RL fine-tune restores masked
#   heads on top of the BC-warm actor (mask forbids use_item when no active,
#   forbids drop_bomb when bombs=0, etc.).
# BREAKS OLD CHECKPOINTS: the actor head resizes from 14 logits (9+5) to 20
# logits (9+5+2+2+2). Consistent with the H_hard pivot per verdict.md — we're
# not resuming v3 anyway.
ACTION_FACTORS = np.array([9, 5, 2, 2, 2], dtype=np.int64)
ACTION_KEYS = ("move", "shoot", "use_item", "drop_bomb", "use_pillcard")


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

# 2026-07-12: PASSIVES_K bumped 256 -> 733 for Track A. Isaac Repentance has
# ~732 vanilla CollectibleType IDs. Old curated top-256 list silently ignored
# high-ID items (Sacred Orb=691, Angelic Prism=528, etc.). Identity mapping
# in mods/isaac-rl-bridge/tables.lua now covers all IDs.
PASSIVES_K = 733

# Character identity (Track A). 34 vanilla Isaac characters (0..33) + 1 unknown
# slot for tainted/DLC characters we haven't classified yet. One-hot in obs.
CHARACTER_K = 35

# Item slot dims (Track A). Isaac supports 2 active-item slots (primary +
# Schoolbag), 2 trinkets (primary + Mom's Purse), and 4 card / pill slots.
# Each slot's obs is [normalized_id, ..., has_flag]; see decoders below.
ACTIVE_SLOTS = 2
ACTIVE_FEATS = 3       # [item_id/730, charge/max, has_flag]
TRINKET_SLOTS = 2
TRINKET_FEATS = 2      # [trinket_id/200, has_flag]
CARD_SLOTS = 4
CARD_FEATS = 2         # [card_id/100, has_flag]
PILL_SLOTS = 4
PILL_FEATS = 2         # [pill_id/25, has_flag]

# Transformation counters (Track A). Isaac Repentance has 15 transformations
# (Guppy=0 ... Super Bum=14). Each returns 0..N progress items collected;
# transformation triggers at 3+ items. We store normalized counters in obs.
TRANSFORMATION_COUNT = 15

# Door features (Track A expansion). Was 6 [exists, open, locked, boss,
# treasure, secret]; now 18 [exists, open, locked, then 15 one-hot flags for
# room types (boss, treasure, secret, shop, arcade, curse, sacrifice, devil,
# angel, library, miniboss, challenge, dungeon, planetarium, chest)].
DOOR_FEATS = 18

# Spatial obs (added schema v2). Preprocessed spatial features derived from
# player position within the room. Fed as a small dense vector; forces the
# network to reason about "where am I in this room" without needing to learn
# it from raw pixel coords. Features (in order):
#   [0, 1] player position normalized to [-1, 1] within room
#   [2, 3, 4, 5] normalized distance to each wall (left, up, right, down)
#   [6, 7] unit vector to nearest OPEN door, or (0, 0) if none open
SPATIAL_DIM = 8

# Player history frame-stacking (added 2026-07-02).
# Last N frames of player state, oldest-first: [nx, ny, vx, vy] per frame.
# Provides short-term motion context beyond what the GRU's internal state
# provides (redundant but explicit; helps early BC learning of dynamics).
# Computed Python-side using a per-env rolling buffer in the env wrapper.
HISTORY_FRAMES = 4
HISTORY_FEATS = 4                                    # (nx, ny, vx, vy)
PLAYER_HISTORY_DIM = HISTORY_FRAMES * HISTORY_FEATS  # 4 * 4 = 16

# B4: Per-episode latent variable z ~ N(0, I). Sampled at reset,
# constant for the whole episode. Encourages strategic diversity.
Z_DIM = 16


def observation_space() -> spaces.Dict:
    return spaces.Dict({
        "player":     spaces.Box(-np.inf, np.inf, shape=(PLAYER_DIM,), dtype=np.float32),
        "passives":   spaces.MultiBinary(PASSIVES_K),
        "room_grid":  spaces.Box(0.0, 1.0, shape=(4, ROOM_H, ROOM_W), dtype=np.float32),
        "doors":      spaces.Box(0.0, 1.0, shape=(4, DOOR_FEATS), dtype=np.float32),
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
        # Player history (frame stacking, added 2026-07-02). Last N frames
        # of [nx, ny, vx, vy] flattened. Bootstrapped by the env wrapper.
        "player_history": spaces.Box(-np.inf, np.inf, shape=(PLAYER_HISTORY_DIM,), dtype=np.float32),
        # B4: Per-episode latent variable (Gaussian). Same across all steps
        # in the episode; changes at reset. Injected by the env wrapper.
        "z": spaces.Box(-np.inf, np.inf, shape=(Z_DIM,), dtype=np.float32),
        # ---- Track A (2026-07-12): character + item slots + transformations ----
        # New obs keys. All zero-fill when raw JSON is missing the field
        # (backward compat with recordings before mod expansion).
        "character":       spaces.MultiBinary(CHARACTER_K),
        "active_items":    spaces.Box(0.0, 1.0, shape=(ACTIVE_SLOTS, ACTIVE_FEATS), dtype=np.float32),
        "trinkets":        spaces.Box(0.0, 1.0, shape=(TRINKET_SLOTS, TRINKET_FEATS), dtype=np.float32),
        "cards":           spaces.Box(0.0, 1.0, shape=(CARD_SLOTS, CARD_FEATS), dtype=np.float32),
        "pills":           spaces.Box(0.0, 1.0, shape=(PILL_SLOTS, PILL_FEATS), dtype=np.float32),
        "transformations": spaces.Box(0.0, 1.0, shape=(TRANSFORMATION_COUNT,), dtype=np.float32),
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
    # Phase 2 (2026-07-14): normalized fire-cooldown-remaining. Written
    # explicitly in encode_obs() (not a raw passthrough) as
    # fire_cooldown / max(1, MaxFireDelay), clipped to [0, 1]. Gives the
    # agent the countdown to next tear so it can learn shot timing on the
    # aim-and-shoot task. Slot 20 of PLAYER_DIM=40 (was unused zero-fill).
    "fire_cooldown_norm",
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
    """Decode doors: 18 features per door (Track A expansion).

    Layout: [exists, open, locked, then 15 one-hot flags for room types].
    Backward compat: if raw has only 6 features per door (old schema), fill
    the first 6 and leave the trailing 12 room-type flags zero.
    """
    out = np.zeros((4, DOOR_FEATS), dtype=np.float32)
    if not raw:
        return out
    for i in range(min(4, len(raw))):
        row = raw[i] or []
        for j in range(min(DOOR_FEATS, len(row))):
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


# --- Track A decoders (character, actives, trinkets, cards, pills, transformations) ---

def _decode_character(raw_player: dict | None) -> np.ndarray:
    """Decode player_type -> one-hot MultiBinary(CHARACTER_K).

    Isaac=0, Magdalene=1, Cain=2, Judas=3, ..., Tainted variants up to 33.
    Values >= CHARACTER_K-1 collapse into the unknown-slot at index
    CHARACTER_K-1. Missing raw returns all zeros.
    """
    out = np.zeros(CHARACTER_K, dtype=np.int8)
    if not raw_player:
        return out
    pt = raw_player.get("player_type")
    if pt is None:
        return out
    try:
        idx = int(pt)
    except (TypeError, ValueError):
        return out
    if idx < 0:
        return out
    if idx >= CHARACTER_K - 1:
        out[CHARACTER_K - 1] = 1
    else:
        out[idx] = 1
    return out


def _decode_active_items(raw_player: dict | None) -> np.ndarray:
    """Decode active-item slots. Box(ACTIVE_SLOTS, ACTIVE_FEATS).

    Per slot: [id/730, charge/max_charge, has_flag]. Slot 0 is primary
    (space bar); slot 1 is Schoolbag secondary.
    """
    out = np.zeros((ACTIVE_SLOTS, ACTIVE_FEATS), dtype=np.float32)
    if not raw_player:
        return out
    # Slot 0: primary
    id0 = float(raw_player.get("active_item_id", 0) or 0)
    ch0 = float(raw_player.get("active_charge", 0) or 0)
    mx0 = float(raw_player.get("active_max_charge", 0) or 0)
    out[0, 0] = min(1.0, max(0.0, id0 / 730.0))
    out[0, 1] = min(1.0, ch0 / max(1.0, mx0)) if mx0 > 0 else 0.0
    out[0, 2] = 1.0 if id0 > 0 else 0.0
    # Slot 1: Schoolbag
    id1 = float(raw_player.get("active_item_id_2", 0) or 0)
    ch1 = float(raw_player.get("active_charge_2", 0) or 0)
    # Mod does not expose max_charge_2; use slot-0 max as approximation.
    mx1 = mx0
    out[1, 0] = min(1.0, max(0.0, id1 / 730.0))
    out[1, 1] = min(1.0, ch1 / max(1.0, mx1)) if mx1 > 0 else 0.0
    out[1, 2] = 1.0 if id1 > 0 else 0.0
    return out


def _decode_trinkets(raw_player: dict | None) -> np.ndarray:
    out = np.zeros((TRINKET_SLOTS, TRINKET_FEATS), dtype=np.float32)
    if not raw_player:
        return out
    for i, key in enumerate(("trinket_id_1", "trinket_id_2")):
        if i >= TRINKET_SLOTS:
            break
        tid = float(raw_player.get(key, 0) or 0)
        out[i, 0] = min(1.0, max(0.0, tid / 200.0))
        out[i, 1] = 1.0 if tid > 0 else 0.0
    return out


def _decode_cards(raw_player: dict | None) -> np.ndarray:
    out = np.zeros((CARD_SLOTS, CARD_FEATS), dtype=np.float32)
    if not raw_player:
        return out
    for i in range(CARD_SLOTS):
        cid = float(raw_player.get(f"card_id_{i+1}", 0) or 0)
        out[i, 0] = min(1.0, max(0.0, cid / 100.0))
        out[i, 1] = 1.0 if cid > 0 else 0.0
    return out


def _decode_pills(raw_player: dict | None) -> np.ndarray:
    out = np.zeros((PILL_SLOTS, PILL_FEATS), dtype=np.float32)
    if not raw_player:
        return out
    for i in range(PILL_SLOTS):
        pid = float(raw_player.get(f"pill_id_{i+1}", 0) or 0)
        out[i, 0] = min(1.0, max(0.0, pid / 25.0))
        out[i, 1] = 1.0 if pid > 0 else 0.0
    return out


def _decode_transformations(raw_player: dict | None) -> np.ndarray:
    out = np.zeros(TRANSFORMATION_COUNT, dtype=np.float32)
    if not raw_player:
        return out
    arr = raw_player.get("transformations")
    if not arr:
        return out
    for i, v in enumerate(arr[:TRANSFORMATION_COUNT]):
        try:
            n = float(v or 0)
        except (TypeError, ValueError):
            continue
        out[i] = min(1.0, max(0.0, n / 10.0))
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

    # Phase 2: normalized fire-cooldown-remaining. The mod emits raw
    # `fire_cooldown` (frames until next tear) + `fire_delay` (== MaxFireDelay).
    # `fire_cooldown_norm` is not a raw JSON field, so the loop above left its
    # slot at 0; compute it here as cooldown / max(1, MaxFireDelay), clipped to
    # [0, 1]. 0 == ready to fire, 1 == just fired / full cooldown.
    idx_cd = _PLAYER_FIELDS.index("fire_cooldown_norm")
    if idx_cd < PLAYER_DIM:
        raw_cd = float(p.get("fire_cooldown", 0) or 0)
        max_delay = float(p.get("fire_delay", 0) or 0)
        norm_cd = raw_cd / max(1.0, max_delay) if max_delay > 0 else 0.0
        obs["player"][idx_cd] = float(min(1.0, max(0.0, norm_cd)))

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

    # Track A obs keys (2026-07-12). All zero-fill if raw JSON lacks the
    # fields (older demos recorded before mod expansion still parse cleanly).
    obs["character"] = _decode_character(p)
    obs["active_items"] = _decode_active_items(p)
    obs["trinkets"] = _decode_trinkets(p)
    obs["cards"] = _decode_cards(p)
    obs["pills"] = _decode_pills(p)
    obs["transformations"] = _decode_transformations(p)

    for key, dim, feat_dim in [
        ("enemies", MAX_ENEMIES, ENEMY_FEATS),
        ("projectiles", MAX_PROJECTILES, PROJ_FEATS),
        ("pickups", MAX_PICKUPS, PICKUP_FEATS),
    ]:
        feats, mask = _copy_entity_feats(raw.get(key), dim, feat_dim)
        obs[key] = {"feats": feats, "mask": mask}

    if last_action is not None:
        # Backward-compat: last_action from callers passing short arrays (len 2)
        # is zero-padded up to len(ACTION_FACTORS) before normalization.
        denom = np.maximum(ACTION_FACTORS - 1, 1).astype(np.float32)
        la = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if la.shape[0] < denom.shape[0]:
            padded = np.zeros(denom.shape[0], dtype=np.float32)
            padded[:la.shape[0]] = la
            la = padded
        elif la.shape[0] > denom.shape[0]:
            la = la[:denom.shape[0]]
        obs["last_action"][:] = la / denom

    return obs


def encode_action(action: np.ndarray | list[int]) -> dict[str, int]:
    """Convert a factor-index array to the {name: int} dict the mod expects.

    Accepts short actions (fewer factors than ACTION_KEYS) for backward
    compatibility with test fixtures written before Track A. Missing factors
    default to 0 (no press / idle).
    """
    a = np.asarray(action, dtype=np.int64).reshape(-1)
    n = min(len(a), len(ACTION_KEYS))
    out = {ACTION_KEYS[i]: int(a[i]) for i in range(n)}
    # Zero-fill any factors past what the caller provided (BC/RL callers pass
    # all K factors; older tests may pass just move+shoot).
    for i in range(n, len(ACTION_KEYS)):
        out[ACTION_KEYS[i]] = 0
    return out


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
        # Player history (frame stacking). Backward compat: zeros if missing.
        "player_history": obs.get("player_history", np.zeros(PLAYER_HISTORY_DIM, dtype=np.float32)),
        "z": obs.get("z", np.zeros(Z_DIM, dtype=np.float32)),
        # Track A obs keys.
        "character": obs.get("character", np.zeros(CHARACTER_K, dtype=np.int8)).astype(np.float32),
        "active_items": obs.get("active_items", np.zeros((ACTIVE_SLOTS, ACTIVE_FEATS), dtype=np.float32)),
        "trinkets": obs.get("trinkets", np.zeros((TRINKET_SLOTS, TRINKET_FEATS), dtype=np.float32)),
        "cards": obs.get("cards", np.zeros((CARD_SLOTS, CARD_FEATS), dtype=np.float32)),
        "pills": obs.get("pills", np.zeros((PILL_SLOTS, PILL_FEATS), dtype=np.float32)),
        "transformations": obs.get("transformations", np.zeros(TRANSFORMATION_COUNT, dtype=np.float32)),
    }
    for key in ("enemies", "projectiles", "pickups"):
        out[f"{key}_feats"] = obs[key]["feats"]
        out[f"{key}_mask"] = obs[key]["mask"].astype(np.float32)
    return out
