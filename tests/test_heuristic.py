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
    p = HeuristicPolicy()
    for _ in range(20):
        obs = _make_obs(enemies=[(100.0, 100.0, 0.0, 0.0)])
        a = p.act(obs)
        assert a[2] == 0   # use_active
        assert a[3] == 0   # drop_bomb
        assert a[4] == 0   # pill_card


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
    assert a.shape == (5,)
    assert a.dtype == np.int64
