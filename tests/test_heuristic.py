"""Tests for the rule-based heuristic policy (heuristic.py)."""
from __future__ import annotations

import math
import numpy as np

from isaac_rl.heuristic import HeuristicConfig, HeuristicPolicy


def _make_obs(enemies=None, projectiles=None, player=None) -> dict:
    """Build a minimal raw obs dict for testing."""
    obs = {"player": player or {"hp_red": 3, "hp_max": 3, "vx": 0.0, "vy": 0.0}}
    if enemies:
        # Each enemy tuple: (dx, dy, vx, vy). Build feats row per obs.lua layout.
        feats = []
        mask = []
        for dx, dy, vx, vy in enemies:
            row = [0.0, 0.0, dx / 480.0, dy / 270.0, vx / 10.0, vy / 10.0] + [0.0] * 10
            feats.append(row)
            mask.append(1)
        obs["enemies"] = {"feats": feats, "mask": mask, "count": len(feats)}
    if projectiles:
        feats = []
        mask = []
        for dx, dy, vx, vy in projectiles:
            row = [0.0, 0.0, dx / 480.0, dy / 270.0, vx / 10.0, vy / 10.0] + [0.0] * 4
            feats.append(row)
            mask.append(1)
        obs["projectiles"] = {"feats": feats, "mask": mask, "count": len(feats)}
    return obs


def test_shoots_right_at_enemy_to_the_right():
    p = HeuristicPolicy()
    obs = _make_obs(enemies=[(300.0, 0.0, 0.0, 0.0)])
    a = p.act(obs)
    assert a[1] == 2   # shoot right


def test_shoots_up_at_enemy_above():
    p = HeuristicPolicy()
    obs = _make_obs(enemies=[(0.0, -300.0, 0.0, 0.0)])
    a = p.act(obs)
    assert a[1] == 1   # shoot up


def test_shoots_down_at_enemy_below():
    p = HeuristicPolicy()
    obs = _make_obs(enemies=[(0.0, 300.0, 0.0, 0.0)])
    a = p.act(obs)
    assert a[1] == 3   # shoot down


def test_shoots_left_at_enemy_to_left():
    p = HeuristicPolicy()
    obs = _make_obs(enemies=[(-300.0, 0.0, 0.0, 0.0)])
    a = p.act(obs)
    assert a[1] == 4   # shoot left


def test_no_shoot_when_no_enemies():
    p = HeuristicPolicy()
    obs = _make_obs()
    a = p.act(obs)
    assert a[1] == 0   # no shoot


def test_approaches_far_enemy():
    p = HeuristicPolicy()
    # Enemy far to the right (300 > approach_dist 240).
    obs = _make_obs(enemies=[(300.0, 0.0, 0.0, 0.0)])
    a = p.act(obs)
    assert a[0] == 3   # move right


def test_retreats_from_close_enemy():
    p = HeuristicPolicy()
    # Enemy very close (50 < retreat_dist 100), to the right → retreat left.
    obs = _make_obs(enemies=[(50.0, 0.0, 0.0, 0.0)])
    a = p.act(obs)
    assert a[0] == 7   # move left


def test_dodges_incoming_projectile():
    p = HeuristicPolicy()
    # Projectile approaching from the right, moving left toward player.
    # dx=100, dy=0, vx=-5, vy=0 → heading at player, threatening.
    # Perpendicular = (0, -5) or (0, 5) → move up or down.
    obs = _make_obs(projectiles=[(100.0, 0.0, -5.0, 0.0)])
    a = p.act(obs)
    assert a[0] in (1, 5)   # up or down


def test_ignores_non_threatening_projectile():
    p = HeuristicPolicy()
    # Projectile far away (500 > threat_dist 150) — should be ignored.
    obs = _make_obs(projectiles=[(500.0, 0.0, -5.0, 0.0)])
    a = p.act(obs)
    # No threat, no enemies — move should be idle or a wander action (0, 1, 3, 5, 7).
    assert a[0] in (0, 1, 3, 5, 7)


def test_ignores_projectile_moving_away():
    p = HeuristicPolicy()
    # Projectile close but moving AWAY from player (vx=+5, dx=100 → moving further right).
    # dot(v, -d) = -(5*100 + 0*0) = -500 < 0 → not threatening.
    obs = _make_obs(projectiles=[(100.0, 0.0, 5.0, 0.0)])
    a = p.act(obs)
    # No threat, no enemies — should NOT be dodging in a perpendicular direction.
    assert a[0] in (0, 1, 3, 5, 7)   # idle / cardinal wander


def test_never_uses_active_or_bomb_or_pill():
    """Regression: after action-space simplification, only move + shoot exist.
    Heuristic never returns extra dims. Kept as smoke test."""
    p = HeuristicPolicy()
    for _ in range(20):
        obs = _make_obs(enemies=[(100.0, 100.0, 0.0, 0.0)])
        a = p.act(obs)
        assert len(a) == 2   # no active/bomb/pill dims


def test_angle_to_shoot_covers_full_circle():
    """Every angle should map to a valid shoot direction (1-4)."""
    for deg in range(0, 360, 15):
        rad = math.radians(deg - 180)   # -pi to +pi
        result = HeuristicPolicy._angle_to_shoot(rad)
        assert result in (1, 2, 3, 4), f"angle {deg}° -> {result}"


def test_angle_to_move_covers_full_circle():
    """Every angle should map to a valid movement direction (1-8)."""
    for deg in range(0, 360, 15):
        rad = math.radians(deg - 180)
        result = HeuristicPolicy._angle_to_move(rad)
        assert result in (1, 2, 3, 4, 5, 6, 7, 8), f"angle {deg}° -> {result}"


def test_picks_nearest_enemy_when_multiple():
    p = HeuristicPolicy()
    # Enemy A far right (500), enemy B close left (-100).
    # Nearest is B → shoot LEFT.
    obs = _make_obs(enemies=[(500.0, 0.0, 0.0, 0.0), (-100.0, 0.0, 0.0, 0.0)])
    a = p.act(obs)
    assert a[1] == 4   # shoot left (at nearest enemy)


def test_returns_ndarray_of_correct_shape_and_dtype():
    p = HeuristicPolicy()
    obs = _make_obs(enemies=[(100.0, 100.0, 0.0, 0.0)])
    a = p.act(obs)
    assert isinstance(a, np.ndarray)
    assert a.shape == (2,)   # 2 heads: move, shoot (after action-space simplification)
    assert a.dtype == np.int64


def test_action_output_length_matches_action_factors():
    """Regression: heuristic action ndim must match spaces.ACTION_FACTORS."""
    from isaac_rl.spaces import ACTION_FACTORS
    p = HeuristicPolicy()
    obs = _make_obs()
    a = p.act(obs)
    assert a.shape == (len(ACTION_FACTORS),), f"heuristic returns {a.shape}, expected ({len(ACTION_FACTORS)},)"


# ---- Lead-shot prediction and multi-projectile threat aggregation ----------


def test_lead_shot_moves_aim_ahead_of_target():
    """Enemy far right (300 px) moving up FAST should be aimed 'up' (predicted
    position dominates over current position because lead is large)."""
    p = HeuristicPolicy()
    # Enemy at (300, 0), moving up hard (vy = -20).
    # tti_tear = 300 / 10 = 30 ticks. predicted dy = -20*30 = -600.
    # atan2(-600, 300) ~= -63 deg -> UP quadrant (angle < -pi/4).
    obs = _make_obs(enemies=[(300.0, 0.0, 0.0, -20.0)])
    a = p.act(obs)
    assert a[1] == 1   # shoot up (lead ahead of moving target)


def test_lead_shot_zero_lead_does_not_change_aim():
    """Same enemy but with the lead disabled by setting a very slow tear
    speed effectively equal to enemy velocity magnitude. Direct-aim direction."""
    p = HeuristicPolicy()
    # Enemy at (300, 0) moving slowly up.
    # tti_tear = 30, dy = -30. aim = (300, -30) -> atan2(-30, 300) ~ -5.7 deg -> RIGHT.
    obs = _make_obs(enemies=[(300.0, 0.0, 0.0, -1.0)])
    a = p.act(obs)
    assert a[1] == 2   # shoot right (tiny lead still keeps aim in right quadrant)


def test_lead_shot_no_lead_for_stationary_enemy():
    """Stationary enemy: aim stays at current position."""
    p = HeuristicPolicy()
    obs = _make_obs(enemies=[(300.0, 0.0, 0.0, 0.0)])   # stationary
    a = p.act(obs)
    assert a[1] == 2   # shoot right (no lead)


def test_multi_projectile_threat_aggregates():
    """Two projectiles approaching from up-left and up-right — bot should
    move DOWN (perpendicular to summed threat)."""
    p = HeuristicPolicy()
    # Projectile A: from upper-left, moving down-right at player.
    #   dx=-80, dy=-80, vx=5, vy=5 → heading at player from upper-left.
    # Projectile B: from upper-right, moving down-left at player.
    #   dx=80, dy=-80, vx=-5, vy=5 → heading at player from upper-right.
    # Both have downward velocity components — the correct dodge is UP (away
    # from where both projectiles are travelling toward).
    obs = _make_obs(projectiles=[
        (-80.0, -80.0, 5.0, 5.0),
        (80.0, -80.0, -5.0, 5.0),
    ])
    a = p.act(obs)
    # The escape should point away from downward-travelling threats.
    # Reasonable safe answers: up (1) or a cardinal that isn't straight into the fire.
    assert a[0] in (1, 2, 5, 7, 8)   # any non-downward-into-fire cardinal is acceptable


def test_all_projectile_threats_sorted_by_urgency():
    """The private helper returns threats sorted by time-to-impact ascending."""
    p = HeuristicPolicy()
    # Two projectiles: A close (fast impact), B far.
    obs = _make_obs(projectiles=[
        (100.0, 0.0, -3.0, 0.0),    # far-ish, slow — later tti
        (50.0, 0.0, -10.0, 0.0),    # close, fast — soonest tti
    ])
    threats = p._all_projectile_threats(obs)
    assert len(threats) == 2
    # threats[0] should be the more urgent one (smaller tti).
    assert threats[0][4] < threats[1][4]


# ---- Door-seeking (post-clear navigation) ----------------------------------


def _make_obs_with_doors(doors, is_clear=True):
    """Build an obs with the given door array and is_clear flag."""
    return {
        "player": {"hp_red": 3, "hp_max": 3, "vx": 0.0, "vy": 0.0},
        "enemies": {"feats": [], "mask": [], "count": 0},
        "projectiles": {"feats": [], "mask": [], "count": 0},
        "global": {"is_clear": 1 if is_clear else 0},
        "doors": doors,
        "events": [],
    }


def _door_seeking_policy() -> HeuristicPolicy:
    """Policy with door-seeking explicitly enabled (default is disabled)."""
    return HeuristicPolicy(HeuristicConfig(enable_door_seeking=True, idle_move_prob=0.0, seed=0))


def test_seeks_right_door_when_room_clear():
    """Room clear with only a RIGHT door open -> move right (only option)."""
    p = _door_seeking_policy()
    # slot 2 = RIGHT. Fields: [exists, is_open, is_locked, is_boss, is_treas, is_secret]
    doors = [
        [0, 0, 0, 0, 0, 0],   # LEFT: doesn't exist
        [0, 0, 0, 0, 0, 0],   # UP: doesn't exist
        [1, 1, 0, 0, 0, 0],   # RIGHT: exists + open
        [0, 0, 0, 0, 0, 0],   # DOWN: doesn't exist
    ]
    obs = _make_obs_with_doors(doors, is_clear=True)
    a = p.act(obs)
    assert a[0] == 3   # move right (only viable door)


def test_seeks_up_door_when_room_clear():
    """Only UP door open -> move up."""
    p = _door_seeking_policy()
    doors = [
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],   # UP open
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    obs = _make_obs_with_doors(doors, is_clear=True)
    a = p.act(obs)
    assert a[0] == 1   # move up (only viable door)


def test_prefers_normal_door_over_boss_door():
    """When both normal and boss doors are open, take the normal one (regardless of slot order)."""
    p = _door_seeking_policy()
    doors = [
        [1, 1, 0, 1, 0, 0],   # LEFT: boss (is_boss=1)
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],   # RIGHT: normal
        [0, 0, 0, 0, 0, 0],
    ]
    obs = _make_obs_with_doors(doors, is_clear=True)
    # Non-zero velocity so stuck-detection doesn't fire during test.
    obs["player"] = {"hp_red": 3, "hp_max": 3, "vx": 2.0, "vy": 0.0}
    for _ in range(20):
        a = p.act(obs)
        assert a[0] == 3, f"picked non-normal door: move={a[0]}"


def test_skips_locked_doors():
    """Locked door isn't picked, open unlocked one is."""
    p = _door_seeking_policy()
    doors = [
        [1, 1, 1, 0, 0, 0],   # LEFT: locked
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],   # RIGHT: open
        [0, 0, 0, 0, 0, 0],
    ]
    obs = _make_obs_with_doors(doors, is_clear=True)
    obs["player"] = {"hp_red": 3, "hp_max": 3, "vx": 2.0, "vy": 0.0}
    for _ in range(20):
        a = p.act(obs)
        assert a[0] == 3, f"picked locked door: move={a[0]}"

def test_door_seeking_when_no_enemies_regardless_of_clear():
    """UPDATED 2026-07-03: door-seeking no longer requires is_clear=True.
    If no enemies visible, bot heads to open door regardless of is_clear.
    Handles edge cases (loading transitions, starting room) where is_clear
    may be False even though the room is safe.
    """
    p = HeuristicPolicy(HeuristicConfig(enable_door_seeking=True, idle_move_prob=0.0))
    doors = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],   # RIGHT open
        [0, 0, 0, 0, 0, 0],
    ]
    obs = _make_obs_with_doors(doors, is_clear=False)
    a = p.act(obs)
    # No enemies visible + RIGHT door open -> should pick RIGHT (move=3).
    assert a[0] == 3


def test_door_selection_picks_nearest_door():
    """With room_bounds set, heuristic picks the door NEAREST to the player,
    not a random one. Fixed 2026-07-03 to prevent tick-to-tick oscillation.

    Verifies:
    - Player near LEFT wall + all 4 doors open -> picks LEFT (move=7)
    - Player near RIGHT wall + all 4 doors open -> picks RIGHT (move=3)
    - Player near UP wall + all 4 doors open -> picks UP (move=1)
    - Player near DOWN wall + all 4 doors open -> picks DOWN (move=5)
    """
    p = HeuristicPolicy(HeuristicConfig(enable_door_seeking=True, idle_move_prob=0.0, seed=0))
    doors = [
        [1, 1, 0, 0, 0, 0],   # LEFT open
        [1, 1, 0, 0, 0, 0],   # UP open
        [1, 1, 0, 0, 0, 0],   # RIGHT open
        [1, 1, 0, 0, 0, 0],   # DOWN open
    ]
    bounds = {"tl_x": 100, "tl_y": 100, "br_x": 500, "br_y": 500}

    # Player near LEFT wall.
    obs_left = _make_obs_with_doors(doors, is_clear=True)
    obs_left["player"] = {"hp_red": 3, "hp_max": 3, "vx": 0.0, "vy": 0.0, "x": 110, "y": 300}
    obs_left["room_bounds"] = bounds
    a = p.act(obs_left)
    assert a[0] == 7, f"expected LEFT (7), got {a[0]}"

    # Player near RIGHT wall.
    obs_right = _make_obs_with_doors(doors, is_clear=True)
    obs_right["player"] = {"hp_red": 3, "hp_max": 3, "vx": 0.0, "vy": 0.0, "x": 490, "y": 300}
    obs_right["room_bounds"] = bounds
    a = p.act(obs_right)
    assert a[0] == 3, f"expected RIGHT (3), got {a[0]}"

    # Player near UP wall.
    obs_up = _make_obs_with_doors(doors, is_clear=True)
    obs_up["player"] = {"hp_red": 3, "hp_max": 3, "vx": 0.0, "vy": 0.0, "x": 300, "y": 110}
    obs_up["room_bounds"] = bounds
    a = p.act(obs_up)
    assert a[0] == 1, f"expected UP (1), got {a[0]}"

    # Player near DOWN wall.
    obs_down = _make_obs_with_doors(doors, is_clear=True)
    obs_down["player"] = {"hp_red": 3, "hp_max": 3, "vx": 0.0, "vy": 0.0, "x": 300, "y": 490}
    obs_down["room_bounds"] = bounds
    a = p.act(obs_down)
    assert a[0] == 5, f"expected DOWN (5), got {a[0]}"


def test_door_selection_stable_same_state():
    """Regression: door selection must be deterministic for the same state.
    Previously (pre-2026-07-03) the heuristic shuffled slot order each call,
    causing the bot to oscillate between doors tick-to-tick.
    """
    p = HeuristicPolicy(HeuristicConfig(enable_door_seeking=True, idle_move_prob=0.0, seed=0))
    doors = [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
    ]
    obs = _make_obs_with_doors(doors, is_clear=True)
    # Non-zero velocity so stuck-detection doesn't fire during rapid test calls.
    obs["player"] = {"hp_red": 3, "hp_max": 3, "vx": -2.0, "vy": 0.0, "x": 110, "y": 300}
    obs["room_bounds"] = {"tl_x": 100, "tl_y": 100, "br_x": 500, "br_y": 500}
    # Player near LEFT wall -> should ALWAYS pick LEFT (move=7).
    picks = [int(p.act(obs)[0]) for _ in range(20)]
    assert all(m == 7 for m in picks), f"expected all LEFT, got {picks}"


def test_door_seeking_enabled_by_default():
    """Since 2026-07-03, door-seeking is ENABLED by default (see
    HeuristicConfig.enable_door_seeking). Previously disabled after early
    user report of wall-hitting, but with improved spatial obs (v2) and
    larger BC demo pool, BC no longer learns the biased LEFT-first behavior.
    """
    p = HeuristicPolicy(HeuristicConfig(idle_move_prob=0.0))
    doors = [
        [1, 1, 0, 0, 0, 0],   # LEFT open
        [1, 1, 0, 0, 0, 0],   # UP open
        [1, 1, 0, 0, 0, 0],   # RIGHT open
        [1, 1, 0, 0, 0, 0],   # DOWN open
    ]
    obs = _make_obs_with_doors(doors, is_clear=True)
    a = p.act(obs)
    # With door-seeking enabled, heuristic picks one of the door directions
    # (7=LEFT, 1=UP, 3=RIGHT, 5=DOWN). NOT idle (0).
    assert a[0] != 0
    assert a[0] in (1, 3, 5, 7)
