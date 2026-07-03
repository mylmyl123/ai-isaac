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
    tear_speed: float = 10.0           # rough estimate of Isaac tear speed for lead-shot prediction

    # Multi-projectile threat aggregation: instead of dodging only the most
    # urgent projectile, sum threat vectors weighted by inverse time-to-impact
    # so we dodge a direction that avoids multiple projectiles when several
    # converge on the player at once.
    max_projectiles_for_threat: int = 8

    # Fallback behavior when no enemies visible
    idle_move_prob: float = 0.4        # probability of moving when idle
    idle_move_choices: tuple = (1, 3, 5, 7)  # cardinal directions

    # Deterministic seed for stochastic tie-breaks
    seed: int = 0

    # Door-seeking behavior. Historically activated when the room was clear;
    # DISABLED by default after user report of bot walking into walls at 50K
    # steps (BC learned biased/wrong door direction due to sparse demo data
    # in cleared-room states — classic BC distribution-shift failure mode).
    # To re-enable: set enable_door_seeking=True below AND
    # r_seek_door_when_clear > 0 in reward config.
    enable_door_seeking: bool = False
    door_slot_to_move: tuple = (7, 1, 3, 5)   # LEFT, UP, RIGHT, DOWN -> move action
    prefer_normal_doors: bool = True


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
        threats = self._all_projectile_threats(raw_obs)
        is_clear = bool((raw_obs.get("global") or {}).get("is_clear", False))

        # ---- Movement decision -----------------------------------------
        move = 0

        if threats:
            # Multi-projectile dodge: aggregate all threat vectors weighted by
            # inverse time-to-impact. Escape direction is perpendicular to
            # the summed threat, pointing away from the centroid of incoming
            # fire. Handles "crossfire" better than dodging one bullet at a
            # time.
            escape_x, escape_y = 0.0, 0.0
            for pvx, pvy, pdx, pdy, tti in threats:
                weight = 1.0 / max(0.5, tti)
                perp_a = (-pvy, pvx)
                perp_b = (pvy, -pvx)
                away_x, away_y = -pdx, -pdy
                score_a = perp_a[0] * away_x + perp_a[1] * away_y
                score_b = perp_b[0] * away_x + perp_b[1] * away_y
                perp = perp_a if score_a >= score_b else perp_b
                escape_x += perp[0] * weight
                escape_y += perp[1] * weight
            if escape_x != 0.0 or escape_y != 0.0:
                move = self._angle_to_move(math.atan2(escape_y, escape_x))

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
            # No enemies, no threats. If door-seeking is ENABLED and the room
            # is clear, head to an open door. Otherwise fall back to random
            # wander so we still explore.
            if cfg.enable_door_seeking and is_clear:
                door_move = self._pick_door_move(raw_obs)
                if door_move is not None:
                    move = door_move
                elif self._rng.random() < cfg.idle_move_prob:
                    move = int(self._rng.choice(cfg.idle_move_choices))
            elif self._rng.random() < cfg.idle_move_prob:
                move = int(self._rng.choice(cfg.idle_move_choices))

        # ---- Shoot decision (with lead-shot prediction) ------------------
        # Instead of aiming at the enemy's current position, aim at where the
        # enemy WILL be by the time our tear arrives. Isaac tears travel
        # slowly (~cfg.tear_speed units/tick) so leading matters a lot for
        # moving enemies. Predicted position = current + velocity * tti.
        shoot = 0
        if enemy is not None:
            edx, edy, edist, evx, evy = enemy
            tti_tear = edist / max(1.0, cfg.tear_speed)
            aim_dx = edx + evx * tti_tear
            aim_dy = edy + evy * tti_tear
            shoot = self._angle_to_shoot(math.atan2(aim_dy, aim_dx))

        # Heuristic outputs a 2-dim action: [move, shoot]. The active/bomb/pill
        # heads were removed from the action space (see spaces.ACTION_FACTORS).
        return np.array([move, shoot], dtype=np.int64)

    # ---- door-seeking (post-clear navigation) --------------------------

    def _pick_door_move(self, raw_obs: dict[str, Any]) -> int | None:
        """When room is clear, choose a movement direction toward an open door.

        Doors obs is a [4, 6] array: rows are LEFT/UP/RIGHT/DOWN slots, columns
        are (exists, is_open, is_locked, is_boss, is_treasure, is_secret).
        Prefers normal doors over special-purpose ones when both are open.
        Returns a movement action (1..8) or None if no viable door.

        RANDOMIZED slot order: if we always iterated 0..3, BC would learn a
        LEFT-first bias ("go left in every cleared room") because slot 0 is
        LEFT. In rooms without a LEFT door, the trained network would then
        walk into the left wall. Shuffling the slot order per call spreads
        the demo distribution across all four cardinal directions.
        """
        cfg = self.cfg
        doors = raw_obs.get("doors")
        if not doors:
            return None

        n_slots = min(4, len(doors))
        slot_order = list(range(n_slots))
        self._rng.shuffle(slot_order)

        # Two passes: normal doors first (if prefer_normal_doors), then any open door.
        for pass_idx in (0, 1):
            for slot in slot_order:
                d = doors[slot]
                if not d or len(d) < 6:
                    continue
                exists = bool(d[0])
                is_open = bool(d[1])
                is_locked = bool(d[2])
                is_boss = bool(d[3])
                is_treasure = bool(d[4])
                is_secret = bool(d[5])
                if not exists or not is_open or is_locked:
                    continue
                if pass_idx == 0 and cfg.prefer_normal_doors and (is_boss or is_treasure or is_secret):
                    continue
                return cfg.door_slot_to_move[slot]
        return None

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
        """Backward-compat: return single most urgent threat (used by tests)."""
        threats = self._all_projectile_threats(raw_obs)
        return threats[0] if threats else None

    def _all_projectile_threats(
        self, raw_obs: dict[str, Any]
    ) -> list[tuple[float, float, float, float, float]]:
        """Return threatening projectiles as list of (vx, vy, dx, dy, time_to_impact).

        A projectile is threatening if:
          * Within projectile_threat_dist of the player.
          * Moving fast enough (speed > projectile_speed_min).
          * Velocity has a positive component pointing at the player.

        Sorted by time-to-impact (most urgent first), truncated to
        max_projectiles_for_threat.
        """
        cfg = self.cfg
        projectiles = raw_obs.get("projectiles") or {}
        feats = projectiles.get("feats") or []
        mask = projectiles.get("mask") or []
        threats: list[tuple[float, float, float, float, float]] = []

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
            toward_score = -(vx * dx + vy * dy)
            if toward_score <= 0:
                continue
            closing_speed = toward_score / max(1.0, dist)
            tti = dist / max(1.0, closing_speed)
            threats.append((vx, vy, dx, dy, tti))

        threats.sort(key=lambda t: t[4])
        return threats[:cfg.max_projectiles_for_threat]

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
