"""Tests for the 2026-07-09 reward-hierarchy rework.

Covers:
  - P0.1: crash_penalty vs mod_restart split (env-level; smoke-tested via reward.py)
  - P0.2: clear_idle_extra grace period + coefficient reduction
  - Phase B: state-delta pickup/use rewards (coin/bomb/key/heart)
  - Phase B: room-type first-entry rewards
  - Phase B: boss_kill event
  - Phase B: use_item event
  - Phase C: episode_behavior_metrics telemetry
  - Phase D: chain reward state machines (bomb-reveals-secret,
    shop-purchase, active-item-ready-and-used)
"""
from __future__ import annotations

import pytest

from isaac_rl.reward import (
    RewardConfig, RewardShaper,
    ROOM_TYPE_SHOP, ROOM_TYPE_TREASURE, ROOM_TYPE_SECRET, ROOM_TYPE_BOSS,
    ROOM_TYPE_DEVIL, ROOM_TYPE_ANGEL, ROOM_TYPE_DEFAULT, ROOM_TYPE_SUPER_SECRET,
)


# ---------------------------------------------------------------------
# P0.2 — clear_idle_extra grace period
# ---------------------------------------------------------------------

def test_clear_idle_extra_grace_period_not_fired_within_grace():
    """Within the first `clear_idle_grace_ticks` after room_clear, the extra
    idle penalty should NOT fire even if the player is idle. Previously this
    fired on every idle tick, contributing -2.83/ep on the 2026-07-08 run."""
    cfg = RewardConfig(clear_idle_grace_ticks=30)
    r = RewardShaper(cfg)
    obs_alive = {
        "player": {"hp_red": 6, "hp_max": 6, "vx": 0, "vy": 0},  # idle
        "global": {"is_clear": True},
        "events": [{"kind": "room_clear"}],
    }
    r(obs_alive)   # room_clear tick
    # Next 20 idle ticks in cleared room — within grace window.
    for _ in range(20):
        _, _, bd = r({
            "player": {"hp_red": 6, "hp_max": 6, "vx": 0, "vy": 0},
            "global": {"is_clear": True},
            "events": [],
        })
        assert "clear_idle_extra" not in bd


def test_clear_idle_extra_fires_after_grace():
    """After grace_ticks elapse, idle in cleared room SHOULD fire the penalty."""
    cfg = RewardConfig(clear_idle_grace_ticks=5)
    r = RewardShaper(cfg)
    r({
        "player": {"hp_red": 6, "hp_max": 6, "vx": 0, "vy": 0},
        "global": {"is_clear": True},
        "events": [{"kind": "room_clear"}],
    })
    # After exactly grace_ticks + 1 idle ticks, penalty should fire.
    for _ in range(cfg.clear_idle_grace_ticks + 1):
        _, _, bd_last = r({
            "player": {"hp_red": 6, "hp_max": 6, "vx": 0, "vy": 0},
            "global": {"is_clear": True},
            "events": [],
        })
    assert "clear_idle_extra" in bd_last
    assert bd_last["clear_idle_extra"] == cfg.r_clear_room_idle_extra


# ---------------------------------------------------------------------
# Phase B — coin/bomb/key/heart delta detection
# ---------------------------------------------------------------------

def test_coin_pickup_via_delta():
    """Coins going UP without an explicit event = pickup. Fires pickup_coin."""
    r = RewardShaper()
    r({"player": {"hp_red": 6, "hp_max": 6, "coins": 5}, "events": []})
    _, _, bd = r({"player": {"hp_red": 6, "hp_max": 6, "coins": 7}, "events": []})
    assert bd.get("pickup_coin") == pytest.approx(RewardConfig().r_pickup_coin * 2)


def test_coin_spend_default_no_reward_no_chain():
    """After 2026-07-09 v2: r_spend_coin default 0.0, no chain reward. Coins
    dropped in shop only fire the collectible-pickup reward downstream."""
    cfg = RewardConfig()  # default r_spend_coin = 0.0
    r = RewardShaper(cfg)
    r({
        "player": {"hp_red": 6, "hp_max": 6, "coins": 10},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                    "safe_grid_index": 42}],
    })
    _, _, bd_spend = r({"player": {"hp_red": 6, "hp_max": 6, "coins": 5}, "events": []})
    assert bd_spend.get("spend_coin", 0.0) == 0.0
    _, _, bd_chain = r({
        "player": {"hp_red": 6, "hp_max": 6, "coins": 5},
        "events": [{"kind": "pickup_collectible"}],
    })
    assert "chain_shop_purchase" not in bd_chain
    assert bd_chain.get("pickup_collectible") == cfg.r_pickup_collectible


def test_bomb_use_small_no_chain():
    """After 2026-07-09 v2: bomb-then-secret chain reward removed. The
    r_secret_first_entry (+5) is enough downstream signal."""
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 2}, "events": []})
    _, _, bd = r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 1}, "events": []})
    # Small token reward for bomb use.
    assert bd.get("use_bomb") == cfg.r_use_bomb
    assert cfg.r_use_bomb < 0.1, "bomb-use must be small to avoid spam incentive"
    for _ in range(10):
        r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 1}, "events": []})
    _, _, bd2 = r({
        "player": {"hp_red": 6, "hp_max": 6, "bombs": 1},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SECRET,
                    "safe_grid_index": 99}],
    })
    assert "chain_bomb_reveals_secret" not in bd2
    assert bd2.get("secret_first_entry") == cfg.r_secret_first_entry


def test_use_item_no_chain_regardless_of_charge():
    """After 2026-07-09 v2: use_item fires the base reward regardless of
    was_charged. Chain reward removed."""
    cfg = RewardConfig()
    for was_charged in (False, True):
        r = RewardShaper(cfg)
        _, _, bd = r({
            "player": {"hp_red": 6, "hp_max": 6},
            "events": [{"kind": "use_item", "was_charged": was_charged}],
        })
        assert bd.get("use_item") == cfg.r_use_item
        assert "chain_active_item_ready_and_used" not in bd


# ---------------------------------------------------------------------
# End-of-episode aggregate outcome bonuses (2026-07-09 v2)
# ---------------------------------------------------------------------

def test_depth_end_bonus_fires_on_termination():
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    # Simulate: reach floor 2, then die (fires terminated=True).
    r({"player": {"hp_red": 6, "hp_max": 6}, "events": [{"kind": "new_level", "stage": 1}]})
    r({"player": {"hp_red": 6, "hp_max": 6}, "events": [{"kind": "new_level", "stage": 2}]})
    _, term, bd = r({"player": {"hp_red": 0, "hp_max": 6}, "events": []})
    assert term
    assert bd.get("depth_end_bonus") == cfg.r_depth_end_bonus * 2


def test_diversity_end_bonus_sublinear():
    import math
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    # Visit 3 room types (default + shop + treasure).
    for i, rt in enumerate([ROOM_TYPE_DEFAULT, ROOM_TYPE_SHOP, ROOM_TYPE_TREASURE]):
        r({
            "player": {"hp_red": 6, "hp_max": 6},
            "events": [{"kind": "new_room", "is_new": True, "room_type": rt,
                        "safe_grid_index": i}],
        })
    _, term, bd = r({"player": {"hp_red": 0, "hp_max": 6}, "events": []})
    assert term
    expected = cfg.r_diversity_end_bonus * math.log(1 + 3)
    assert bd.get("diversity_end_bonus") == pytest.approx(expected)


def test_survival_end_bonus_zero_when_dead():
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({"player": {"hp_red": 6, "hp_max": 6}, "events": []})
    _, term, bd = r({"player": {"hp_red": 0, "hp_max": 6}, "events": []})
    assert term
    # Died → no survival bonus.
    assert "survival_end_bonus" not in bd


def test_survival_end_bonus_partial_hp():
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({"player": {"hp_red": 6, "hp_max": 6}, "events": []})
    # Force terminate via a death event but with non-zero HP (mod-side death).
    _, term, bd = r({
        "player": {"hp_red": 3, "hp_max": 6},
        "events": [{"kind": "death"}],
    })
    assert term
    # st.dead is now True — no survival bonus.
    assert "survival_end_bonus" not in bd


def test_efficiency_end_bonus():
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    # 2 rooms visited, 1 item collected.
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_DEFAULT,
                    "safe_grid_index": 1}],
    })
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_TREASURE,
                    "safe_grid_index": 2}, {"kind": "pickup_collectible"}],
    })
    _, term, bd = r({"player": {"hp_red": 0, "hp_max": 6}, "events": []})
    assert term
    # 1 item / 2 rooms = 0.5
    assert bd.get("efficiency_end_bonus") == pytest.approx(cfg.r_efficiency_end_bonus * 0.5)


def test_bomb_use_via_delta_fires_use_bomb():
    """Bomb count going down = bomb placed. Fires small use_bomb reward."""
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 2}, "events": []})
    _, _, bd = r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 1}, "events": []})
    assert bd.get("use_bomb") == cfg.r_use_bomb


def test_bomb_then_secret_room_no_chain_only_secret_reward():
    """After 2026-07-09 v2: chain_bomb_reveals_secret removed. Bombing a
    wall + entering the secret room fires secret_first_entry (+5) only.
    Emergent bomb-then-secret behavior must come from RND / value learning."""
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 2}, "events": []})
    r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 1}, "events": []})
    for _ in range(10):
        r({"player": {"hp_red": 6, "hp_max": 6, "bombs": 1}, "events": []})
    _, _, bd = r({
        "player": {"hp_red": 6, "hp_max": 6, "bombs": 1},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SECRET,
                    "safe_grid_index": 99}],
    })
    assert "chain_bomb_reveals_secret" not in bd
    assert bd.get("secret_first_entry") == cfg.r_secret_first_entry


# (test_bomb_then_secret_outside_window_does_not_fire_chain removed: chain
#  rewards removed 2026-07-09 v2, so window semantics are moot.)


def test_key_use_via_delta():
    r = RewardShaper()
    r({"player": {"hp_red": 6, "hp_max": 6, "keys": 1}, "events": []})
    _, _, bd = r({"player": {"hp_red": 6, "hp_max": 6, "keys": 0}, "events": []})
    assert bd.get("use_key") == RewardConfig().r_use_key


def test_heart_pickup_via_hp_increase():
    """HP going up (without new_room) = heart pickup. Fires pickup_heart."""
    r = RewardShaper()
    r({"player": {"hp_red": 4, "hp_max": 6}, "events": []})
    _, _, bd = r({"player": {"hp_red": 6, "hp_max": 6}, "events": []})
    # HP went from 4 → 6, delta=2. Reward = 2 × r_pickup_heart.
    assert bd.get("pickup_heart") == pytest.approx(RewardConfig().r_pickup_heart * 2)


# ---------------------------------------------------------------------
# Phase B — room-type first-entry rewards
# ---------------------------------------------------------------------

def test_room_type_first_entry_fires_once_per_type():
    """Each room type first-entry reward should fire ONCE per episode."""
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    # First entry to shop → fires shop_first_entry.
    _, _, bd1 = r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                    "safe_grid_index": 1}],
    })
    assert bd1.get("shop_first_entry") == cfg.r_shop_first_entry
    # Second entry to shop (different room) → does NOT fire again.
    _, _, bd2 = r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                    "safe_grid_index": 2}],
    })
    assert "shop_first_entry" not in bd2


def test_room_type_all_special_types_covered():
    """Check that all documented special room types can fire their reward."""
    from isaac_rl.reward import _ROOM_FIRST_ENTRY_TABLE
    cfg = RewardConfig()
    for room_type, (key, cfg_attr) in _ROOM_FIRST_ENTRY_TABLE.items():
        r = RewardShaper(cfg)
        _, _, bd = r({
            "player": {"hp_red": 6, "hp_max": 6},
            "events": [{"kind": "new_room", "is_new": True, "room_type": room_type,
                        "safe_grid_index": room_type * 100}],
        })
        expected = getattr(cfg, cfg_attr)
        assert bd.get(key) == expected, (
            f"Room type {room_type} ({key}) expected {expected}, got {bd.get(key)}"
        )


def test_default_room_type_no_first_entry_reward():
    """Default rooms should NOT fire any first-entry reward."""
    r = RewardShaper()
    _, _, bd = r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_DEFAULT,
                    "safe_grid_index": 1}],
    })
    for k in bd:
        assert not k.endswith("_first_entry"), f"unexpected first-entry reward: {k}"


# ---------------------------------------------------------------------
# Phase B — boss_kill + use_item events
# ---------------------------------------------------------------------

def test_boss_kill_via_damage_to_npc_is_boss_flag():
    """damage_to_npc with is_boss + killed=True fires both kill AND boss_kill."""
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    _, _, bd = r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{
            "kind": "damage_to_npc",
            "dmg": 100, "npc_max_hp": 100,
            "killed": True, "is_boss": True,
        }],
    })
    assert bd.get("kill") == cfg.r_kill
    assert bd.get("boss_kill") == cfg.r_boss_kill


def test_regular_kill_does_not_fire_boss_kill():
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    _, _, bd = r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{
            "kind": "damage_to_npc",
            "dmg": 10, "npc_max_hp": 10, "killed": True, "is_boss": False,
        }],
    })
    assert bd.get("kill") == cfg.r_kill
    assert "boss_kill" not in bd


def test_use_item_event_fires_reward():
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    _, _, bd = r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "use_item", "was_charged": False}],
    })
    assert bd.get("use_item") == cfg.r_use_item
    assert "chain_active_item_ready_and_used" not in bd


# ---------------------------------------------------------------------
# Phase C — behavior metrics
# ---------------------------------------------------------------------

def test_behavior_metrics_tracks_rooms_and_kills():
    r = RewardShaper()
    # 3 rooms visited, 2 kills, 1 boss kill.
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                    "safe_grid_index": 1}],
    })
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "damage_to_npc", "dmg": 10, "npc_max_hp": 10, "killed": True, "is_boss": True}],
    })
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_BOSS,
                    "safe_grid_index": 2}],
    })
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "damage_to_npc", "dmg": 10, "npc_max_hp": 10, "killed": True}],
    })
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_TREASURE,
                    "safe_grid_index": 3}],
    })
    metrics = r.episode_behavior_metrics()
    assert metrics["rooms_visited"] == 3
    assert metrics["kills"] == 2
    assert metrics["boss_kills"] == 1
    # 3 unique room types (shop, boss, treasure).
    assert metrics["room_types_seen"] == 3


def test_behavior_metrics_tracks_coin_and_bomb_activity():
    r = RewardShaper()
    r({"player": {"hp_red": 6, "hp_max": 6, "coins": 0, "bombs": 3}, "events": []})
    r({"player": {"hp_red": 6, "hp_max": 6, "coins": 5, "bombs": 3}, "events": []})  # +5 coins
    r({"player": {"hp_red": 6, "hp_max": 6, "coins": 3, "bombs": 3}, "events": []})  # -2 coins
    r({"player": {"hp_red": 6, "hp_max": 6, "coins": 3, "bombs": 2}, "events": []})  # -1 bomb
    r({"player": {"hp_red": 6, "hp_max": 6, "coins": 3, "bombs": 1}, "events": []})  # -1 bomb
    metrics = r.episode_behavior_metrics()
    assert metrics["coins_earned"] == 5
    assert metrics["coins_spent"] == 2
    assert metrics["bombs_used"] == 2


def test_behavior_metrics_reset_on_reset():
    r = RewardShaper()
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                    "safe_grid_index": 1}],
    })
    m1 = r.episode_behavior_metrics()
    assert m1["rooms_visited"] == 1
    r.reset()
    m2 = r.episode_behavior_metrics()
    assert m2["rooms_visited"] == 0
    assert m2["kills"] == 0


# ---------------------------------------------------------------------
# Regression — kills tracked correctly across ticks
# ---------------------------------------------------------------------

def test_kill_streak_tracking():
    """kill_streak grows with successive kills, resets on damage taken."""
    r = RewardShaper()
    r({"player": {"hp_red": 6, "hp_max": 6}, "events": []})
    # 3 kills in a row.
    for _ in range(3):
        r({
            "player": {"hp_red": 6, "hp_max": 6},
            "events": [{"kind": "damage_to_npc", "dmg": 10, "npc_max_hp": 10, "killed": True}],
        })
    assert r.episode_behavior_metrics()["max_kill_streak"] == 3
    # Take damage — streak resets, but max was 3.
    r({
        "player": {"hp_red": 4, "hp_max": 6},
        "events": [{"kind": "damage_to_player", "dmg": 2}],
    })
    r({
        "player": {"hp_red": 4, "hp_max": 6},
        "events": [{"kind": "damage_to_npc", "dmg": 10, "npc_max_hp": 10, "killed": True}],
    })
    metrics = r.episode_behavior_metrics()
    assert metrics["max_kill_streak"] == 3   # historical max preserved
    assert metrics["kills"] == 4
