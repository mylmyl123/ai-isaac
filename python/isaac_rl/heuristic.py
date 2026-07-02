"""Rule-based heuristic policy for Isaac RL.

Purpose: bootstrap PPO with sensible behavior. Runs faster and better than
random exploration on a compute budget too small for pure-RL from scratch.

Used two ways:
  1. Demo collection: run this policy for N steps, save (obs, action) trajectories
     to disk for behavior-cloning pretraining (see bc.py).
  2. Standalone play: works fine as a non-learning bot for demos / eval.

The rules are intentionally simple. PPO fine-tunes past the heuristic ceiling
once BC pretraining has given it a competent starting point.

Action space (see spaces.py IsaacActionSpace):
    action[0] = move: 0=idle, 1=up, 2=up-right, 3=right, 4=down-right,
                      5=down, 6=down-left, 7=left, 8=up-left
    action[1] = shoot: 0=none, 1=up, 2=right, 3=down, 4=left
    action[2] = use_active: 0/1
    action[3] = drop_bomb: 0/1
    action[4] = pill_card: 0/1

Heuristic always sets use_active=drop_bomb=pill_card=0. These are hard to use
correctly without game knowledge (using D6 wastes charge, dropping bomb hurts
self, pills can be negative). PPO discovers them later on its own.

Obs feature layouts (from mods/isaac-rl-bridge/obs.lua):

Enemies (feats[i], 16 dims each):
    [0]  nx (normalized room x, 0-1)
    [1]  ny
    [2]  dx / 480   -- world-unit x offset from player
    [3]  dy / 270
    [4]  vx / 10    -- velocity
    [5]  vy / 10
    [6]  hp / max_hp
    [7]  is_boss
    ...

Projectiles (feats[i], 10 dims each):
    [0]  nx
    [1]  ny
    [2]  dx / 480
    [3]  dy / 270
    [4]  vx / 10
    [5]  vy / 10
    ...
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HeuristicConfig:
    # Distance thresholds (world units)
    engage_dist: float = 200.0        # try to stay at this distance from enemies
    retreat_dist: float = 100.0       # if closer than this, retreat
    approach_dist: float = 240.0      # if farther than this, approach
    projectile_threat_dist: float = 150.0  # projectile is threatening if within this range

    # Speeds
    projectile_speed_min: float = 3.0  # ignore near-stationary projectiles

    # Fallback behavior when no enemies visible
    idle_move_prob: float = 0.4        # probability of moving when idle
    idle_move_choices: tuple = (1, 3, 5, 7)  # cardinal directions

    # Deterministic seed for stochastic tie-breaks
    seed: int = 0


class HeuristicPolicy:
    """Rule-based Isaac player. Stateless per-tick decisions."""

    def __init__(self, config: HeuristicConfig | None = None):
        self.cfg = config or HeuristicConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def act(self, raw_obs: dict[str, Any]) -> np.ndarray:
        """Return action ndarray of shape (5,), dtype int64.

        raw_obs is the Lua-side JSON dict (with keys 'player', 'enemies',
        'projectiles', 'room', 'events', etc.) — same structure that gets
        fed to RewardShaper.
        """
        cfg = self.cfg

        enemy = self._nearest_enemy(raw_obs)
        threat = self._most_urgent_projectile_threat(raw_obs)

        # ---- Movement decision -----------------------------------------
        move = 0

        if threat is not None:
            # Dodge: move perpendicular to projectile velocity.
            # Rotate proj velocity 90° to get sideways-relative-to-shot direction.
            # Pick the perpendicular direction pointing away from the projectile.
            pvx, pvy, pdx, pdy, _ = threat
            # Perpendicular = (-vy, vx) or (vy, -vx). Pick whichever has a
            # component pointing away from the projectile's origin.
            perp1 = (-pvy, pvx)
            perp2 = (pvy, -pvx)
            # Vector from projectile to player = (-pdx, -pdy). Dodge in the
            # perpendicular that's most aligned with player-away-from-projectile.
            away_x, away_y = -pdx, -pdy
            score1 = perp1[0] * away_x + perp1[1] * away_y
            score2 = perp2[0] * away_x + perp2[1] * away_y
            perp = perp1 if score1 >= score2 else perp2
            move = self._angle_to_move(math.atan2(perp[1], perp[0]))

        elif enemy is not None:
            edx, edy, edist, _, _ = enemy
            if edist < cfg.retreat_dist:
                # Too close — move away from enemy.
                move = self._angle_to_move(math.atan2(-edy, -edx))
            elif edist > cfg.approach_dist:
                # Too far — approach.
                move = self._angle_to_move(math.atan2(edy, edx))
            else:
                # In the sweet spot — strafe perpendicular to enemy line-of-sight.
                # Choose left or right strafe pseudo-randomly (stable per-tick).
                if self._rng.random() < 0.5:
                    move = self._angle_to_move(math.atan2(-edx, edy))     # rotate +90°
                else:
                    move = self._angle_to_move(math.atan2(edx, -edy))     # rotate -90°

        else:
            # No enemies, no threats. Wander a bit so we discover new rooms.
            if self._rng.random() < cfg.idle_move_prob:
                move = int(self._rng.choice(cfg.idle_move_choices))

        # ---- Shoot decision --------------------------------------------
        shoot = 0
        if enemy is not None:
            edx, edy, _, _, _ = enemy
            shoot = self._angle_to_shoot(math.atan2(edy, edx))

        # No use_active / no bomb / no pill for the heuristic.
        return np.array([move, shoot, 0, 0, 0], dtype=np.int64)

    # ---- feature extraction helpers ------------------------------------

    def _nearest_enemy(self, raw_obs: dict[str, Any]) -> tuple[float, float, float, float, float] | None:
        """Return (dx, dy, dist, vx, vy) to nearest visible enemy, in world units. None if no enemies."""
        enemies = raw_obs.get("enemies") or {}
        feats = enemies.get("feats") or []
        mask = enemies.get("mask") or []
        best: tuple[float, float, float, float, float] | None = None
        for i, f in enumerate(feats):
            if i >= len(mask) or not mask[i] or not f or len(f) < 6:
                continue
            dx = float(f[2]) * 480.0
            dy = float(f[3]) * 270.0
            dist = math.hypot(dx, dy)
            if best is None or dist < best[2]:
                vx = float(f[4]) * 10.0
                vy = float(f[5]) * 10.0
                best = (dx, dy, dist, vx, vy)
        return best

    def _most_urgent_projectile_threat(
        self, raw_obs: dict[str, Any]
    ) -> tuple[float, float, float, float, float] | None:
        """Find the most urgent enemy projectile heading toward the player.

        Returns (proj_vx, proj_vy, proj_dx, proj_dy, time_to_impact) or None.
        A projectile is threatening if:
          * Within projectile_threat_dist of the player.
          * Moving fast enough (speed > projectile_speed_min).
          * Its velocity has a positive component pointing at the player
            (dot(velocity, -displacement) > 0).
        """
        cfg = self.cfg
        projectiles = raw_obs.get("projectiles") or {}
        feats = projectiles.get("feats") or []
        mask = projectiles.get("mask") or []
        best: tuple[float, float, float, float, float] | None = None

        for i, f in enumerate(feats):
            if i >= len(mask) or not mask[i] or not f or len(f) < 6:
                continue
            dx = float(f[2]) * 480.0
            dy = float(f[3]) * 270.0
            vx = float(f[4]) * 10.0
            vy = float(f[5]) * 10.0
            dist = math.hypot(dx, dy)
            speed = math.hypot(vx, vy)
            if dist > cfg.projectile_threat_dist:
                continue
            if speed < cfg.projectile_speed_min:
                continue
            # Toward-player check: dot(velocity, player_from_projectile).
            # Player from projectile = -(dx, dy). Projectile heading toward
            # player when dot(v, -d) > 0 => (-vx * dx) + (-vy * dy) > 0.
            toward_score = -(vx * dx + vy * dy)
            if toward_score <= 0:
                continue
            # Time-to-impact (rough): closing distance / closing speed.
            closing_speed = toward_score / max(1.0, dist)
            tti = dist / max(1.0, closing_speed)
            if best is None or tti < best[4]:
                best = (vx, vy, dx, dy, tti)
        return best

    # ---- angle -> action helpers ---------------------------------------

    @staticmethod
    def _angle_to_shoot(angle: float) -> int:
        """Map atan2(dy, dx) angle (Isaac Y down) to shoot action 0-4.

        1=up, 2=right, 3=down, 4=left. Returns 0 for no-shoot only if caller
        wants that; this helper always returns a cardinal.
        """
        # Normalize to (-pi, pi]
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi

        if -math.pi / 4 <= angle <= math.pi / 4:
            return 2  # right
        elif math.pi / 4 < angle <= 3 * math.pi / 4:
            return 3  # down (Isaac Y-axis increases downward)
        elif -3 * math.pi / 4 <= angle < -math.pi / 4:
            return 1  # up
        else:
            return 4  # left

    @staticmethod
    def _angle_to_move(angle: float) -> int:
        """Map atan2 angle to 8-way movement action 1-8. 0 = idle (not returned here)."""
        # Normalize to [0, 2pi)
        if angle < 0:
            angle += 2 * math.pi

        # Divide 2pi into 8 sectors of pi/4 each, offset by pi/8 so the
        # cardinal directions land in the middle of a sector.
        sector = int((angle + math.pi / 8) / (math.pi / 4)) % 8
        # sector 0 = right, 1 = down-right, 2 = down, ..., 7 = up-right
        # Isaac action map:  1=up, 2=up-right, 3=right, 4=down-right,
        #                    5=down, 6=down-left, 7=left, 8=up-left
        mapping = {
            0: 3,   # right
            1: 4,   # down-right
            2: 5,   # down
            3: 6,   # down-left
            4: 7,   # left
            5: 8,   # up-left
            6: 1,   # up
            7: 2,   # up-right
        }
        return mapping[sector]
