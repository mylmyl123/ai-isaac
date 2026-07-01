"""End-to-end tests for the reward shaper."""
from __future__ import annotations

from isaac_rl.reward import RewardConfig, RewardShaper


def test_room_clear_pays_once():
    r = RewardShaper()
    obs = {"player": {"hp_red": 3}, "events": [{"kind": "room_clear"}]}
    reward, term, bd = r(obs)
    assert bd["room_clear"] == r.cfg.r_room_clear
    assert not term

    # A tick with no events pays only the idle penalty.
    reward2, _, bd2 = r({"player": {"hp_red": 3}, "events": []})
    assert "room_clear" not in bd2


def test_beat_mom_terminates():
    r = RewardShaper()
    # Simulate: enter stage 6, then descend to stage 7 → beat_mom bonus + terminate.
    r({"player": {}, "events": [{"kind": "new_level", "stage": 6}]})
    _, term, bd = r({"player": {}, "events": [{"kind": "new_level", "stage": 7}]})
    assert term
    assert bd.get("beat_mom") == RewardConfig().r_beat_mom
    assert bd.get("floor_cleared") == RewardConfig().r_floor_cleared


def test_death_terminates():
    r = RewardShaper()
    _, term, bd = r({"player": {}, "events": [{"kind": "death"}]})
    assert term
    assert bd.get("death") == RewardConfig().r_death


def test_new_room_first_entry_only():
    r = RewardShaper()
    e1 = {"kind": "new_room", "is_new": True, "safe_grid_index": 42, "room_type": 1}
    reward1, _, bd1 = r({"player": {}, "events": [e1]})
    assert bd1.get("new_room") == RewardConfig().r_new_room

    # Same room visited again — no is_new flag → no reward.
    e2 = {"kind": "new_room", "is_new": False, "safe_grid_index": 42, "room_type": 1}
    _, _, bd2 = r({"player": {}, "events": [e2]})
    assert "new_room" not in bd2


def test_damage_reward_capped_per_room():
    cfg = RewardConfig(max_damage_reward_per_room=0.5)
    r = RewardShaper(cfg)
    # 10 damage events on a max_hp=10 target — each pays r_damage_dealt_scale.
    # r_damage_dealt_scale=0.1 → uncapped would sum to 1.0. Capped at 0.5.
    total = 0.0
    for _ in range(10):
        reward, _, bd = r({
            "player": {},
            "events": [{"kind": "damage_to_npc", "dmg": 10, "npc_max_hp": 10}],
        })
        total += bd.get("damage_dealt", 0.0)
    assert total <= cfg.max_damage_reward_per_room + 1e-6


def test_hp_delta_penalizes_damage_taken():
    r = RewardShaper()
    # First tick establishes baseline.
    r({"player": {"hp_red": 6, "hp_soul": 0, "hp_black": 0}, "events": []})
    # Second tick: lost 2 red hearts.
    _, _, bd = r({"player": {"hp_red": 4, "hp_soul": 0, "hp_black": 0}, "events": []})
    assert bd.get("hp_delta_red") == 2 * RewardConfig().r_damage_taken_red
