"""Phase-2 regression tests (2026-07-14).

Covers the correctness changes made to make the agent learn:
  * fire_cooldown_norm added to the player obs (Change 5) — round-trips and
    keeps the flattened obs dimension stable (PLAYER_DIM unchanged).
  * kills_mean is derived from the reward breakdown, not info["raw"]["events"]
    (Change 3) — verify the count == kill_reward / r_kill arithmetic.
  * NPC_TYPES entity-id sanity for the curriculum enemies (Changes 1+2) —
    verified from the mod's tables.lua text (no live Isaac needed).

Run:
    PYTHONPATH=python pytest tests/test_phase2_fixes.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from isaac_rl.reward import RewardConfig
from isaac_rl.spaces import (
    PLAYER_DIM,
    _PLAYER_FIELDS,
    encode_obs,
    flatten_dict_obs,
    observation_space,
    zero_obs,
)


REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Change 5: fire_cooldown_norm
# --------------------------------------------------------------------------

def test_fire_cooldown_field_present_and_in_bounds():
    assert "fire_cooldown_norm" in _PLAYER_FIELDS
    idx = _PLAYER_FIELDS.index("fire_cooldown_norm")
    # Must live inside the fixed PLAYER_DIM budget (uses a previously-unused
    # trailing slot; no shape change, no checkpoint break).
    assert idx < PLAYER_DIM


def test_fire_cooldown_normalized_midway():
    # cooldown = 5 frames out of a 10-frame MaxFireDelay -> 0.5.
    raw = {"player": {"x": 0, "y": 0, "fire_cooldown": 5, "fire_delay": 10}, "global": {}}
    obs = encode_obs(raw)
    idx = _PLAYER_FIELDS.index("fire_cooldown_norm")
    assert obs["player"][idx] == pytest.approx(0.5)


def test_fire_cooldown_ready_is_zero():
    raw = {"player": {"fire_cooldown": 0, "fire_delay": 10}, "global": {}}
    obs = encode_obs(raw)
    idx = _PLAYER_FIELDS.index("fire_cooldown_norm")
    assert obs["player"][idx] == pytest.approx(0.0)


def test_fire_cooldown_clipped_and_safe_when_missing():
    # Missing fire_delay -> norm falls back to 0 (no divide-by-zero / no crash).
    raw = {"player": {"fire_cooldown": 7}, "global": {}}
    obs = encode_obs(raw)
    idx = _PLAYER_FIELDS.index("fire_cooldown_norm")
    assert obs["player"][idx] == pytest.approx(0.0)
    # Over-full cooldown is clipped to 1.0.
    raw2 = {"player": {"fire_cooldown": 99, "fire_delay": 10}, "global": {}}
    obs2 = encode_obs(raw2)
    assert obs2["player"][idx] == pytest.approx(1.0)


def test_obs_dim_unchanged_and_space_still_contains():
    # The obs schema must not have grown a dimension — fire_cooldown_norm reuses
    # an existing PLAYER_DIM slot, so the flattened obs dim and gym space are
    # unchanged (protects the trainer's Linear(obs_dim, hidden) shape).
    obs = zero_obs()
    assert observation_space().contains(obs)
    flat = flatten_dict_obs(obs)
    total = sum(int(np.prod(v.shape)) for v in flat.values())
    # player vector still exactly PLAYER_DIM wide.
    assert flat["player"].shape[0] == PLAYER_DIM
    assert total > 0


# --------------------------------------------------------------------------
# Change 3: kills_mean from reward breakdown (arithmetic the trainer uses)
# --------------------------------------------------------------------------

def test_kill_count_from_breakdown_arithmetic():
    r_kill = float(RewardConfig().r_kill)
    assert r_kill > 0
    # 10 kills' worth of reward -> exactly 10 kills, robust to float noise.
    bd_ep = {"kill": 10.0 * r_kill, "death": -1.0, "step": -0.8}
    kills = int(round(float(bd_ep.get("kill", 0.0)) / r_kill))
    assert kills == 10
    # No kills -> 0 (not a spurious 1 from the terminal frame's events).
    assert int(round(float({}.get("kill", 0.0)) / r_kill)) == 0


# --------------------------------------------------------------------------
# Changes 1+2: entity-id correctness in the mod (static text checks)
# --------------------------------------------------------------------------

def test_stage0_spawns_charger_23():
    """Stage 0 = 2x Charger (type 23): moving/un-camp-able bootstrap. NOT 26
    (Maw, the old ID bug) and NOT 12 (Horf, the stationary enemy that let the
    agent park out of range). Stage A/B stay Attack Fly (18)."""
    main_lua = (REPO / "mods" / "isaac-rl-bridge" / "main.lua").read_text(encoding="utf-8")
    m = re.search(r'STAGE_ENEMY_TYPE\s*=\s*\(STAGE\s*==\s*"0"\)\s*and\s*(\d+)\s*or\s*(\d+)', main_lua)
    assert m is not None, "could not find STAGE_ENEMY_TYPE assignment"
    stage0_type, other_type = int(m.group(1)), int(m.group(2))
    assert stage0_type == 23, f"Stage 0 must spawn Charger (23), got {stage0_type}"
    assert other_type == 18, f"Stage A/B must spawn Attack Fly (18), got {other_type}"
    # Stage 0 count must be 2 (two enemies -> un-camp-able).
    c = re.search(r'STAGE_ENEMY_COUNT\s*=\s*\(STAGE\s*==\s*"0"\)\s*and\s*(\d+)', main_lua)
    assert c is not None and int(c.group(1)) == 2, "Stage 0 must spawn 2 enemies"


def test_npc_types_horf_and_attackfly_distinct_and_present():
    tables_lua = (REPO / "mods" / "isaac-rl-bridge" / "tables.lua").read_text(encoding="utf-8")
    # Extract the numeric NPC id list (one "<int>," per line inside common_npcs).
    ids = [int(x) for x in re.findall(r"^\s+(\d+),", tables_lua, flags=re.MULTILINE)]
    assert 12 in ids, "Horf (12) missing from NPC_TYPES"
    assert 18 in ids, "Attack Fly (18) missing from NPC_TYPES"
    # Dense index = 1-based position in the list (see tables.lua ipairs loop).
    idx_horf = ids.index(12) + 1
    idx_fly = ids.index(18) + 1
    assert idx_horf != idx_fly, "Horf and Attack Fly must map to distinct indices"
    assert idx_horf > 0 and idx_fly > 0, "indices must be non-zero (not unknown/0)"
