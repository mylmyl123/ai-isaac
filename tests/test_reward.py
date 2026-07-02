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


# ---- Dense shaping (aggressive reward tuning) --------------------------------


def _make_enemy_at(dx: float, dy: float, hp: float = 1.0) -> dict:
    """Build an enemies dict with one enemy at (dx, dy) world-units from player."""
    # feats layout: [nx, ny, dx/480, dy/270, ...]
    feats = [[0.0, 0.0, dx / 480.0, dy / 270.0] + [0.0] * 12]
    return {"feats": feats, "mask": [1], "count": 1}


def test_aim_alignment_reward_fires_when_shooting_at_enemy():
    r = RewardShaper()
    # Enemy directly to the right of player.
    obs = {"player": {"hp_red": 3, "hp_max": 3}, "enemies": _make_enemy_at(200, 0), "events": []}
    # action[1] = 2 (shoot right)
    _, _, bd = r(obs, action=[0, 2, 0, 0, 0])
    assert bd.get("aim_at_enemy") == r.cfg.r_aim_at_enemy
    assert bd.get("shoot_when_enemy") == r.cfg.r_shoot_when_enemy_visible


def test_aim_alignment_no_reward_wrong_direction():
    r = RewardShaper()
    # Enemy right, but shoot left.
    obs = {"player": {"hp_red": 3, "hp_max": 3}, "enemies": _make_enemy_at(200, 0), "events": []}
    _, _, bd = r(obs, action=[0, 4, 0, 0, 0])   # shoot=4 = left
    assert "aim_at_enemy" not in bd
    # Still gets the "shooting when enemy visible" bonus.
    assert bd.get("shoot_when_enemy") == r.cfg.r_shoot_when_enemy_visible


def test_kite_distance_reward_at_ideal_range():
    r = RewardShaper()
    # Enemy 200px right — inside default kite range [100, 300].
    obs = {"player": {"hp_red": 3, "hp_max": 3}, "enemies": _make_enemy_at(200, 0), "events": []}
    _, _, bd = r(obs, action=[0, 0, 0, 0, 0])
    assert bd.get("at_kite_dist") == r.cfg.r_at_kite_dist_tick


def test_kite_distance_no_reward_when_too_close():
    r = RewardShaper()
    obs = {"player": {"hp_red": 3, "hp_max": 3}, "enemies": _make_enemy_at(50, 0), "events": []}
    _, _, bd = r(obs, action=[0, 0, 0, 0, 0])
    assert "at_kite_dist" not in bd


def test_idle_penalty_fires_when_stationary():
    r = RewardShaper()
    obs = {"player": {"hp_red": 3, "hp_max": 3, "vx": 0.0, "vy": 0.0}, "events": []}
    _, _, bd = r(obs)
    assert bd.get("idle_penalty") == r.cfg.r_idle_penalty


def test_idle_penalty_absent_when_moving():
    r = RewardShaper()
    obs = {"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0}, "events": []}
    _, _, bd = r(obs)
    assert "idle_penalty" not in bd


def test_full_hp_tick_reward():
    r = RewardShaper()
    obs = {"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0}, "events": []}
    _, _, bd = r(obs)
    assert bd.get("full_hp_tick") == r.cfg.r_full_hp_tick


def test_full_hp_tick_absent_when_hurt():
    r = RewardShaper()
    obs = {"player": {"hp_red": 2, "hp_max": 3, "vx": 5.0, "vy": 0.0}, "events": []}
    _, _, bd = r(obs)
    assert "full_hp_tick" not in bd


def test_pbrs_approach_positive_when_moving_toward_ideal_dist():
    r = RewardShaper()
    # Tick 1: enemy at 500 (far). Sets prev_potential.
    r({"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0},
       "enemies": _make_enemy_at(500, 0), "events": []})
    # Tick 2: enemy at 200 (ideal). Potential jumps up → positive PBRS.
    _, _, bd = r({"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0},
                  "enemies": _make_enemy_at(200, 0), "events": []})
    assert bd.get("pbrs_approach", 0.0) > 0.0


def test_room_clear_speed_bonus():
    r = RewardShaper()
    # Enter room (starts speed timer).
    r({"player": {"hp_red": 3, "hp_max": 3}, "events": [{"kind": "new_room", "is_new": True}]})
    # A few ticks later, room clears — should get the speed bonus.
    for _ in range(50):
        r({"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0}, "events": []})
    _, _, bd = r({"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0},
                  "events": [{"kind": "room_clear"}]})
    assert bd.get("room_clear_speed") == r.cfg.r_room_clear_speed_bonus


def test_room_clear_no_damage_bonus():
    r = RewardShaper()
    r({"player": {"hp_red": 3, "hp_max": 3}, "events": [{"kind": "new_room", "is_new": True}]})
    # Clear without taking damage.
    _, _, bd = r({"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0},
                  "events": [{"kind": "room_clear"}]})
    assert bd.get("room_clear_no_damage") == r.cfg.r_room_clear_no_damage


def test_stationary_penalty_fires_when_camping():
    r = RewardShaper()
    # Feed enough ticks at nearly the same position to trigger stationary_window.
    obs = {"player": {"hp_red": 3, "hp_max": 3, "vx": 1.0, "vy": 0.0,
                       "x": 100.0, "y": 100.0}, "events": []}
    for _ in range(r.cfg.stationary_window):
        r(obs)
    # Next tick should fire the stationary penalty.
    _, _, bd = r(obs)
    assert bd.get("stationary_penalty") == r.cfg.r_stationary_penalty


def test_stationary_penalty_absent_when_moving_around():
    r = RewardShaper()
    # Move by more than stationary_radius over the window.
    for i in range(r.cfg.stationary_window + 5):
        obs = {"player": {"hp_red": 3, "hp_max": 3, "vx": 5.0, "vy": 0.0,
                          "x": float(i * 10), "y": 100.0}, "events": []}
        _, _, bd = r(obs)
    # Final tick shouldn't be penalised.
    assert "stationary_penalty" not in bd
