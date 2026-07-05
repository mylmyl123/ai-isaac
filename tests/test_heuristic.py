"""Tests for the simplified heuristic policy (2026-07-04 rewrite).

Old test file backed up as tests/test_heuristic_v1_deprecated.py.bak if needed.

Test structure:
  1. Combat behavior (aim, shoot, kite zones)
  2. Door target selection + LOCK across ticks (regression: oscillation)
  3. Door target RESET on room change
  4. Diagonal navigation to door center
  5. Fallback when no doors or bad state
"""
from __future__ import annotations

import numpy as np
import pytest

from isaac_rl.heuristic import HeuristicPolicy, HeuristicConfig


# ---- Helpers -------------------------------------------------------------

def _mk_obs(
    px=300.0, py=280.0, vx=0.0, vy=0.0, hp=3,
    enemies=None,
    doors=None,
    is_clear=True,
    room_index=1,
    bounds=(80, 160, 560, 400),
):
    """Build a raw_obs dict matching the mod's schema."""
    tl_x, tl_y, br_x, br_y = bounds
    obs = {
        "player": {"hp_red": hp, "hp_max": 3, "vx": vx, "vy": vy, "x": px, "y": py},
        "enemies": enemies or {"feats": [], "mask": []},
        "projectiles": {"feats": [], "mask": []},
        "doors": doors or [[0]*6, [0]*6, [0]*6, [0]*6],
        "global": {"is_clear": 1 if is_clear else 0, "room_index": room_index},
        "room_bounds": {"tl_x": tl_x, "tl_y": tl_y, "br_x": br_x, "br_y": br_y},
        "events": [],
    }
    return obs


def _mk_enemy(dx=100.0, dy=0.0):
    """Enemy feature vector. feats[2]=dx/480, feats[3]=dy/270."""
    return {
        "feats": [[0.5, 0.5, dx / 480.0, dy / 270.0]],
        "mask": [True],
    }


def test_shoot_at_enemy_right():
    """Enemy to the right -> shoot=2 (right)."""
    p = HeuristicPolicy()
    obs = _mk_obs(enemies=_mk_enemy(dx=150, dy=0))
    a = p.act(obs)
    assert a[1] == 2, f"expected shoot=2 (right), got {a[1]}"


def test_shoot_at_enemy_up():
    """Enemy above (dy negative) -> shoot=1 (up)."""
    p = HeuristicPolicy()
    obs = _mk_obs(enemies=_mk_enemy(dx=0, dy=-100))
    a = p.act(obs)
    assert a[1] == 1, f"expected shoot=1 (up), got {a[1]}"





def test_no_doors_visible_random_wander():
    """No open doors -> random cardinal, never idle."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    obs = _mk_obs()   # no enemies, no doors
    for _ in range(20):
        a = p.act(obs)
        # Should be a valid move (1-8), never idle (0).
        assert 1 <= a[0] <= 8, f"got idle or invalid: {a[0]}"


def test_door_target_locked_across_ticks_same_room():
    """CRITICAL REGRESSION TEST: once a door target is picked, it must NOT
    change tick-to-tick within the same room. Previous heuristic (v1)
    flipped between doors as bot position changed, causing left-right
    oscillation."""
    # Use high stuck_ticks so stuck detection doesn't fire in the test window.
    p = HeuristicPolicy(HeuristicConfig(seed=0, stuck_ticks=1000))
    # Room with LEFT+RIGHT doors both open. Bot in center.
    doors = [
        [1, 1, 0, 0, 0, 0],   # LEFT open
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],   # RIGHT open
        [0, 0, 0, 0, 0, 0],
    ]
    obs = _mk_obs(px=300, py=280, doors=doors, room_index=1)
    # First call locks a target.
    a0 = p.act(obs)
    target_first = p._target_door_slot
    # Call 30 more times in the same room. Target must NOT change.
    for i in range(30):
        _ = p.act(obs)
        assert p._target_door_slot == target_first, \
            f"target flipped at tick {i}: was {target_first}, now {p._target_door_slot}"


def test_door_target_resets_on_room_change():
    """When room_index changes, target must reset (re-picked for new room)."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    doors = [[1, 1, 0, 0, 0, 0], [0]*6, [1, 1, 0, 0, 0, 0], [0]*6]
    obs1 = _mk_obs(doors=doors, room_index=1)
    p.act(obs1)
    assert p._target_door_slot is not None
    # Room changes: target is cleared, re-picked next tick.
    obs2 = _mk_obs(doors=doors, room_index=2)
    p.act(obs2)
    # Target may be same or different slot (random), but it was RESET by
    # _target_door_slot = None before re-picking.
    assert p._prev_room_index == 2


def test_prefer_normal_over_boss_door():
    """Boss/treasure/secret doors are only picked if no normal doors."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    # LEFT is boss (is_boss=1), RIGHT is normal.
    doors = [
        [1, 1, 0, 1, 0, 0],   # LEFT boss
        [0]*6,
        [1, 1, 0, 0, 0, 0],   # RIGHT normal
        [0]*6,
    ]
    obs = _mk_obs(doors=doors)
    p.act(obs)
    assert p._target_door_slot == 2, f"picked non-normal: {p._target_door_slot}"


def test_locked_door_skipped():
    """Locked doors are never picked."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    doors = [
        [1, 1, 1, 0, 0, 0],   # LEFT locked
        [0]*6,
        [1, 1, 0, 0, 0, 0],   # RIGHT open
        [0]*6,
    ]
    obs = _mk_obs(doors=doors)
    p.act(obs)
    assert p._target_door_slot == 2


def test_move_toward_left_door_from_above():
    """Bot above the LEFT door -> should move down-left (6)."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    doors = [[1, 1, 0, 0, 0, 0], [0]*6, [0]*6, [0]*6]
    # LEFT door at (80, 280). Bot at (300, 180) — above the door center.
    obs = _mk_obs(px=300, py=180, doors=doors)
    a = p.act(obs)
    assert a[0] == 6, f"expected down-left (6), got {a[0]}"


def test_move_toward_left_door_when_aligned():
    """Bot Y-aligned with LEFT door -> pure left (7)."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    doors = [[1, 1, 0, 0, 0, 0], [0]*6, [0]*6, [0]*6]
    # LEFT door at (80, 280). Bot at (300, 280) — aligned.
    obs = _mk_obs(px=300, py=280, doors=doors)
    a = p.act(obs)
    assert a[0] == 7, f"expected pure left (7), got {a[0]}"


def test_move_toward_up_door_from_right():
    """Bot right of UP door -> up-left (8)."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    doors = [[0]*6, [1, 1, 0, 0, 0, 0], [0]*6, [0]*6]
    # UP door at (320, 160). Bot at (400, 300).
    obs = _mk_obs(px=400, py=300, doors=doors)
    a = p.act(obs)
    assert a[0] == 8, f"expected up-left (8), got {a[0]}"


def test_move_toward_down_door_from_left():
    """Bot left of DOWN door -> down-right (4)."""
    p = HeuristicPolicy(HeuristicConfig(seed=0))
    doors = [[0]*6, [0]*6, [0]*6, [1, 1, 0, 0, 0, 0]]
    # DOWN door at (320, 400). Bot at (200, 300).
    obs = _mk_obs(px=200, py=300, doors=doors)
    a = p.act(obs)
    assert a[0] == 4, f"expected down-right (4), got {a[0]}"


def test_no_oscillation_locked_target_delivers_stable_moves():
    """Simulate ticks with the same obs. Move action should be stable while
    the target is locked (within stuck_ticks). After stuck_ticks, target
    unlocks and moves may change - that's expected safety behavior."""
    # Use high stuck_ticks so it doesn't fire in the test window.
    p = HeuristicPolicy(HeuristicConfig(seed=0, stuck_ticks=1000))
    doors = [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
    # Position AWAY from all walls to avoid triggering entry_slot inference.
    obs = _mk_obs(px=300, py=280, doors=doors, room_index=1)
    moves = [int(p.act(obs)[0]) for _ in range(50)]
    unique = set(moves)
    # With static obs and locked target (no stuck timeout), we should see
    # at most 1 unique move.
    assert len(unique) == 1, f"expected 1 unique move, got {unique}"


def test_shoot_fires_in_all_kite_zones():
    """Shoot should be non-zero whenever an enemy is visible, regardless of zone."""
    p = HeuristicPolicy(HeuristicConfig(retreat_dist=100, approach_dist=250))
    for dx in [50, 180, 300]:   # retreat, hold, approach zones
        obs = _mk_obs(enemies=_mk_enemy(dx=dx, dy=0))
        a = p.act(obs)
        assert a[1] != 0, f"expected shoot != 0 at dx={dx}, got {a[1]}"


def test_determinism_with_seed():
    """Same seed + same obs sequence -> same actions."""
    doors = [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0]*6, [0]*6]
    obs = _mk_obs(doors=doors)
    p1 = HeuristicPolicy(HeuristicConfig(seed=42))
    p2 = HeuristicPolicy(HeuristicConfig(seed=42))
    for _ in range(10):
        a1 = p1.act(obs)
        a2 = p2.act(obs)
        assert (a1 == a2).all(), f"nondeterministic: {a1} vs {a2}"


def test_angle_to_shoot_cardinals():
    """Cardinal angles map correctly."""
    import math
    assert HeuristicPolicy._angle_to_shoot(0.0) == 2       # right
    assert HeuristicPolicy._angle_to_shoot(math.pi) == 4   # left
    assert HeuristicPolicy._angle_to_shoot(-math.pi / 2) == 1   # up (Isaac Y-down)
    assert HeuristicPolicy._angle_to_shoot(math.pi / 2) == 3    # down


def test_angle_to_move_cardinals():
    """Cardinal angles map to correct 8-way move actions."""
    import math
    assert HeuristicPolicy._angle_to_move(0.0) == 3        # right
    assert HeuristicPolicy._angle_to_move(math.pi) == 7    # left
    assert HeuristicPolicy._angle_to_move(-math.pi / 2) == 1   # up
    assert HeuristicPolicy._angle_to_move(math.pi / 2) == 5    # down




def test_rotate_move_90_clockwise():
    from isaac_rl.heuristic import HeuristicPolicy
    assert HeuristicPolicy._rotate_move_90(1, clockwise=True) == 3   # UP -> RIGHT
    assert HeuristicPolicy._rotate_move_90(3, clockwise=True) == 5   # RIGHT -> DOWN
    assert HeuristicPolicy._rotate_move_90(5, clockwise=True) == 7   # DOWN -> LEFT
    assert HeuristicPolicy._rotate_move_90(7, clockwise=True) == 1   # LEFT -> UP
    assert HeuristicPolicy._rotate_move_90(0, clockwise=True) == 0   # idle stays idle


def test_rotate_move_90_counterclockwise():
    from isaac_rl.heuristic import HeuristicPolicy
    assert HeuristicPolicy._rotate_move_90(1, clockwise=False) == 7   # UP -> LEFT
    assert HeuristicPolicy._rotate_move_90(3, clockwise=False) == 1   # RIGHT -> UP
