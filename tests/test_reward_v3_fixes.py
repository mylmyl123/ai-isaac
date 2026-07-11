"""Tests for the 2026-07-09 v3 fixes:
  * seek_door capped to prevent reward-hacking
  * finalize_episode() fires aggregates on all terminal paths
  * mod_restart branch in env.py applies r_death + aggregates
"""
from __future__ import annotations

import pytest

from isaac_rl.reward import RewardConfig, RewardShaper, ROOM_TYPE_SHOP, ROOM_TYPE_TREASURE


def test_seek_door_capped_per_episode():
    """After 2026-07-09 v3: seek_door caps at max_seek_door_reward_per_episode.
    Prevents the 'hover near door forever' reward pump discovered in the
    40h v2 run (seek_door fired 99% of episodes for +6.3/ep on average)."""
    cfg = RewardConfig(
        r_seek_door_when_clear=0.05,
        max_seek_door_reward_per_episode=0.3,
        seek_door_speed_threshold=0.1,
    )
    r = RewardShaper(cfg)
    # Room is clear, agent moves toward an open right-door for many ticks.
    obs = {
        "player": {"hp_red": 6, "hp_max": 6, "vx": 1.0, "vy": 0.0},
        "global": {"is_clear": True},
        "doors": [
            [0]*6, [0]*6,
            [1, 1, 1.0, 0.0, 0.5, 0.5],   # slot 2: RIGHT door, exists+open
            [0]*6,
        ],
        "events": [],
    }
    total = 0.0
    for _ in range(200):
        _, _, bd = r(obs)
        total += bd.get("seek_door", 0.0)
    # Should never exceed the cap.
    assert total <= cfg.max_seek_door_reward_per_episode + 1e-6, (
        f"cap not enforced: total={total} > {cfg.max_seek_door_reward_per_episode}"
    )
    # And should approximately HIT the cap given 200 aligned ticks at 0.05/tick.
    assert total >= 0.99 * cfg.max_seek_door_reward_per_episode, (
        f"cap suspiciously low: total={total}"
    )


def test_finalize_episode_fires_aggregates_on_all_reasons():
    """finalize_episode() should emit aggregate bonuses regardless of the
    reason (except isaac_crash)."""
    cfg = RewardConfig()
    for reason in ("shaper_terminated", "truncated", "mod_restart"):
        r = RewardShaper(cfg)
        # Visit 2 room types.
        r({
            "player": {"hp_red": 6, "hp_max": 6},
            "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                        "safe_grid_index": 1}],
        })
        r({
            "player": {"hp_red": 6, "hp_max": 6},
            "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_TREASURE,
                        "safe_grid_index": 2}, {"kind": "pickup_collectible"}],
        })
        total, bd = r.finalize_episode(reason)
        assert total > 0, f"aggregates empty for reason={reason}: {bd}"
        assert "diversity_end_bonus" in bd
        assert "efficiency_end_bonus" in bd
        # Not dead, HP survives -> survival bonus.
        assert "survival_end_bonus" in bd


def test_finalize_episode_skips_isaac_crash():
    """isaac_crash means data is unreliable \u2014 no aggregates."""
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                    "safe_grid_index": 1}],
    })
    total, bd = r.finalize_episode("isaac_crash")
    assert total == 0.0
    assert bd == {}


def test_finalize_episode_survival_zero_when_dead():
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({"player": {"hp_red": 6, "hp_max": 6}, "events": []})
    # Trigger HP-based death detection.
    r({"player": {"hp_red": 0, "hp_max": 6}, "events": []})
    total, bd = r.finalize_episode("mod_restart")
    assert "survival_end_bonus" not in bd


def test_shaper_call_no_longer_emits_aggregates_inline():
    """2026-07-09 v3 moved aggregates OUT of shaper.__call__(). A shaper-\n    terminated call should no longer emit them inline."""
    cfg = RewardConfig()
    r = RewardShaper(cfg)
    r({
        "player": {"hp_red": 6, "hp_max": 6},
        "events": [{"kind": "new_room", "is_new": True, "room_type": ROOM_TYPE_SHOP,
                    "safe_grid_index": 1}],
    })
    # Death event triggers terminated=True in shaper.__call__.
    _, term, bd = r({
        "player": {"hp_red": 0, "hp_max": 6},
        "events": [{"kind": "death"}],
    })
    assert term
    # v3: aggregates are NOT in the per-tick breakdown anymore.
    for k in ("diversity_end_bonus", "depth_end_bonus", "survival_end_bonus", "efficiency_end_bonus"):
        assert k not in bd, f"v3 fix: {k} should be emitted only via finalize_episode()"
