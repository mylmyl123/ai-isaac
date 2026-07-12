"""Track A (2026-07-12) reward tests: Plan C trap-item guard + quality-scaled pickup."""
from __future__ import annotations

import numpy as np

from isaac_rl.reward import RewardConfig, RewardShaper, TRAP_ITEM_PENALTIES


def _make_obs(active_item_id: int = 0, hp: int = 6) -> dict:
    """Minimal raw obs the shaper needs."""
    return {
        "player": {
            "hp_red": hp, "hp_soul": 0, "hp_black": 0, "hp_max": hp,
            "keys": 0, "bombs": 0, "coins": 0,
            "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0,
            "active_item_id": active_item_id,
        },
        "global": {"stage": 1, "room_type": 1, "is_clear": False},
        "events": [],
    }


def test_trap_table_contains_chaos_card() -> None:
    assert 475 in TRAP_ITEM_PENALTIES
    assert TRAP_ITEM_PENALTIES[475] == -20.0


def test_chaos_card_use_item_fires_penalty() -> None:
    shaper = RewardShaper(RewardConfig())
    obs = _make_obs(active_item_id=475)         # holding Chaos Card
    # Action layout: [move, shoot, use_item, drop_bomb, use_pillcard]
    action = [0, 0, 1, 0, 0]                    # press space
    reward, _, bd = shaper(obs, action=action)
    assert bd.get("trap_item", 0.0) == -20.0
    assert reward < -19.0                        # -20 dominates the step penalty


def test_chaos_card_without_use_item_no_penalty() -> None:
    shaper = RewardShaper(RewardConfig())
    obs = _make_obs(active_item_id=475)
    action = [0, 0, 0, 0, 0]                    # not pressing space
    _, _, bd = shaper(obs, action=action)
    assert bd.get("trap_item", 0.0) == 0.0


def test_use_item_on_safe_active_no_penalty() -> None:
    shaper = RewardShaper(RewardConfig())
    obs = _make_obs(active_item_id=105)         # D6 (Isaac's default, non-trap)
    action = [0, 0, 1, 0, 0]                    # press space
    _, _, bd = shaper(obs, action=action)
    assert bd.get("trap_item", 0.0) == 0.0


def test_pickup_collectible_quality_scaled_high() -> None:
    """Q4 (top-tier, e.g. Sacred Heart) yields +6.0 reward."""
    shaper = RewardShaper(RewardConfig())
    obs = _make_obs()
    obs["events"] = [{"kind": "pickup_collectible", "item_id": 182, "quality": 4}]
    _, _, bd = shaper(obs, action=[0, 0, 0, 0, 0])
    assert bd.get("pickup_collectible", 0.0) == 6.0


def test_pickup_collectible_quality_scaled_low() -> None:
    """Q0 (Wavy Cap, etc.) yields +0.5 reward instead of the old flat +2.0."""
    shaper = RewardShaper(RewardConfig())
    obs = _make_obs()
    obs["events"] = [{"kind": "pickup_collectible", "item_id": 550, "quality": 0}]
    _, _, bd = shaper(obs, action=[0, 0, 0, 0, 0])
    assert bd.get("pickup_collectible", 0.0) == 0.5


def test_pickup_collectible_missing_quality_falls_back_to_flat() -> None:
    """Older mod builds emitting no quality field still get +2.0 (old default)."""
    shaper = RewardShaper(RewardConfig())
    obs = _make_obs()
    obs["events"] = [{"kind": "pickup_collectible", "item_id": 105}]
    _, _, bd = shaper(obs, action=[0, 0, 0, 0, 0])
    assert bd.get("pickup_collectible", 0.0) == 2.0
