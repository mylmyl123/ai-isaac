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

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Set env var ISAAC_HEURISTIC_DEBUG=1 to log per-tick decisions to stderr.
# Useful for diagnosing "why is the bot walking randomly" issues.
_DEBUG = os.environ.get("ISAAC_HEURISTIC_DEBUG", "").lower() in ("1", "true", "yes")


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
    idle_move_prob: float = 0.7        # was 0.4 (2026-07-03) — boost so BC learns to
                                       # move more often in cleared rooms
    idle_move_choices: tuple = (1, 3, 5, 7)  # cardinal directions

    # Deterministic seed for stochastic tie-breaks
    seed: int = 0

    # Door-seeking behavior. RE-ENABLED 2026-07-03 (was disabled after user
    # report of BC bias in earlier build).
    # We now have:
    #   - Explicit spatial obs field (unit vector to nearest OPEN door) so
    #     the network doesn't need to infer door direction from raw obs.
    #   - Randomized slot iteration in _pick_door_move (see method comment)
    #     so BC doesn't learn a LEFT-first bias.
    #   - Larger demo dataset (500K+) so tail-case door directions are seen.
    #   - Room_bounds field in obs.lua, enabling potential-based door reward.
    # These fixes together make door-seeking demos SAFE to learn from.
    enable_door_seeking: bool = True
    door_slot_to_move: tuple = (7, 1, 3, 5)   # LEFT, UP, RIGHT, DOWN -> move action
    prefer_normal_doors: bool = True


class HeuristicPolicy:
    """Rule-based Isaac player. Stateless per-tick decisions."""

    def __init__(self, config: HeuristicConfig | None = None):
        self.cfg = config or HeuristicConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        # Stuck-detection state. Isaac's collision means "push into obstacle =
        # velocity stays near zero". Track consecutive stuck ticks so we can
        # switch direction when needed.
        self._stuck_ticks = 0
        self._last_action_move = 0
        # Room-transition tracking. When room_index changes, we know the bot
        # just crossed a door. Remember which door we came IN through so we
        # can avoid immediately going back through it.
        self._prev_room_index: int | None = None
        self._entered_from_slot: int | None = None   # 0=LEFT,1=UP,2=RIGHT,3=DOWN

    def act(self, raw_obs: dict[str, Any]) -> np.ndarray:
        """Return action ndarray of shape (5,), dtype int64.

        raw_obs is the Lua-side JSON dict (with keys 'player', 'enemies',
        'projectiles', 'room', 'events', etc.) — same structure that gets
        fed to RewardShaper.
        """
        cfg = self.cfg

        # Update room-transition tracking. If we just entered a new room,
        # remember which slot we came IN through. Bot is closest to that
        # door at spawn, so we'd naturally pick it as "nearest" — avoiding
        # it prevents immediate backtracking.
        cur_room = None
        gg = raw_obs.get("global") or {}
        cur_room = gg.get("room_index") or gg.get("safe_grid_index")
        if cur_room is not None and cur_room != self._prev_room_index:
            # Room changed. Infer entry slot from player position: whichever
            # wall the player is nearest to is the wall they came through.
            bounds = raw_obs.get("room_bounds")
            if bounds:
                player = raw_obs.get("player") or {}
                px = float(player.get("x", 0) or 0)
                py = float(player.get("y", 0) or 0)
                tl_x = float(bounds.get("tl_x", 0) or 0)
                tl_y = float(bounds.get("tl_y", 0) or 0)
                br_x = float(bounds.get("br_x", 1) or 1)
                br_y = float(bounds.get("br_y", 1) or 1)
                dl = px - tl_x
                du = py - tl_y
                dr = br_x - px
                dd = br_y - py
                nearest = min(dl, du, dr, dd)
                # 0=LEFT, 1=UP, 2=RIGHT, 3=DOWN
                if nearest == dl:
                    self._entered_from_slot = 0
                elif nearest == du:
                    self._entered_from_slot = 1
                elif nearest == dr:
                    self._entered_from_slot = 2
                else:
                    self._entered_from_slot = 3
            self._prev_room_index = cur_room
            self._stuck_ticks = 0   # reset stuck counter on room change

        # Stuck detection: if last-tick action was non-idle but player velocity
        # is near zero, we're pushing into an obstacle. Increment counter;
        # when it exceeds threshold, force a direction change this tick.
        player = raw_obs.get("player") or {}
        vx = float(player.get("vx", 0) or 0)
        vy = float(player.get("vy", 0) or 0)
        actual_speed = math.hypot(vx, vy)
        if self._last_action_move != 0 and actual_speed < 0.5:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0

        enemy = self._nearest_enemy(raw_obs)
        threats = self._all_projectile_threats(raw_obs)
        is_clear = bool((raw_obs.get("global") or {}).get("is_clear", False))

        # Wall-avoidance: compute a bitmask of "forbidden" cardinal directions
        # based on player proximity to each wall. Prevents BC from learning
        # "push into wall" — the classic corner-hugging pathology.
        forbidden = self._forbidden_directions(raw_obs)
        # If we've been stuck for 3+ ticks, forbid the last action too.
        if self._stuck_ticks >= 3 and self._last_action_move != 0:
            forbidden.add(self._last_action_move)

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
            # No enemies, no threats. Head to an open door if door-seeking
            # is enabled. We no longer gate on is_clear: if there are no
            # visible enemies, we might as well progress through a door.
            # This handles edge cases (starting room, loading transitions)
            # where is_clear may be False but doors are available.
            if cfg.enable_door_seeking:
                door_move = self._pick_door_move(raw_obs)
                if door_move is not None and door_move not in forbidden:
                    move = door_move
                    if _DEBUG:
                        log.info("[heuristic] no-enemies -> door_move=%d (is_clear=%s)", move, is_clear)
                elif self._rng.random() < cfg.idle_move_prob:
                    move = self._safe_random_move(forbidden)
                    if _DEBUG:
                        log.info("[heuristic] no-enemies, no reachable door (door_move=%s), random=%d forbidden=%s doors=%s",
                                 door_move, move, sorted(forbidden), raw_obs.get("doors"))
            elif self._rng.random() < cfg.idle_move_prob:
                move = self._safe_random_move(forbidden)
                if _DEBUG:
                    log.info("[heuristic] door-seek disabled, random=%d", move)

        # ---- Wall-avoidance final filter --------------------------------
        # Regardless of how `move` was chosen (combat dodging, kiting, door-
        # seeking, random wander), if the chosen action would push into a
        # wall we're already against, override to a non-forbidden direction.
        # Prevents BC from learning the corner-hugging pathology.
        if move != 0 and move in forbidden:
            # Try the two rotations adjacent to the chosen move (±45°).
            adjacent = ((move - 1 - 1) % 8 + 1, (move - 1 + 1) % 8 + 1)
            candidates = [a for a in adjacent if a not in forbidden]
            if candidates:
                move = self._rng.choice(candidates)
            else:
                # All adjacent also forbidden. Fall back to any legal cardinal.
                move = self._safe_random_move(forbidden)

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
        self._last_action_move = int(move)
        return np.array([move, shoot], dtype=np.int64)

    # ---- wall-avoidance (prevent corner-hugging in BC demos) -----------

    def _forbidden_directions(self, raw_obs: dict[str, Any]) -> set[int]:
        """Return the set of movement actions that would push into a wall.

        Uses room_bounds + player position to detect wall proximity. When
        the player is within wall_proximity_threshold of a wall, moves that
        would push into that wall are marked forbidden.

        Prevents BC from learning "push into wall" — the classic corner-
        hugging pathology. Without this, the random-wander and door-seek
        fallbacks can push into walls indefinitely, and the network learns
        "if I'm at a wall, my action is 'move into wall'" (because that's
        what the heuristic did there).

        Move action codes (from ACTION_FACTORS[0]):
          0=idle, 1=up, 2=up-right, 3=right, 4=down-right,
          5=down, 6=down-left, 7=left, 8=up-left
        """
        forbidden: set[int] = set()
        bounds = raw_obs.get("room_bounds")
        if not bounds:
            return forbidden   # no bounds info — conservative fallback
        player = raw_obs.get("player") or {}
        px = float(player.get("x", 0) or 0)
        py = float(player.get("y", 0) or 0)
        tl_x = float(bounds.get("tl_x", 0) or 0)
        tl_y = float(bounds.get("tl_y", 0) or 0)
        br_x = float(bounds.get("br_x", 1) or 1)
        br_y = float(bounds.get("br_y", 1) or 1)
        # Wall proximity threshold: 5% of room dimension. Was 15% (too
        # aggressive) which prevented door-crossing (doors are AT walls).
        # 5% catches genuine wall-pressing without blocking door approach.
        wx = (br_x - tl_x) * 0.05
        wy = (br_y - tl_y) * 0.05
        near_left  = (px - tl_x) < wx
        near_up    = (py - tl_y) < wy
        near_right = (br_x - px) < wx
        near_down  = (br_y - py) < wy

        # KEY FIX: only forbid wall-directed moves if there is NO OPEN DOOR
        # on that wall. Doors are holes in the wall — the bot MUST approach
        # a wall to cross a door. Blocking wall-approach at door locations
        # traps the bot in loops circling the room center.
        doors = raw_obs.get("doors") or []
        def door_open(slot: int) -> bool:
            if slot >= len(doors):
                return False
            d = doors[slot]
            if not d or len(d) < 2:
                return False
            return bool(d[0]) and bool(d[1])
        # Door slot: 0=LEFT, 1=UP, 2=RIGHT, 3=DOWN
        left_has_door  = door_open(0)
        up_has_door    = door_open(1)
        right_has_door = door_open(2)
        down_has_door  = door_open(3)

        # If near a wall AND no door there, forbid moves into that wall.
        # Up-moves: 1,2,8. Down-moves: 4,5,6. Left-moves: 6,7,8. Right-moves: 2,3,4.
        if near_up and not up_has_door:
            forbidden.update({1, 2, 8})
        if near_down and not down_has_door:
            forbidden.update({4, 5, 6})
        if near_left and not left_has_door:
            forbidden.update({6, 7, 8})
        if near_right and not right_has_door:
            forbidden.update({2, 3, 4})
        return forbidden

    def _safe_random_move(self, forbidden: set[int]) -> int:
        """Pick a random cardinal move avoiding forbidden directions."""
        cfg = self.cfg
        allowed = [m for m in cfg.idle_move_choices if m not in forbidden]
        if not allowed:
            # All 4 cardinals blocked (bot at corner + wall thresholds cover
            # the whole room). Just idle rather than push into a wall.
            return 0
        return int(self._rng.choice(allowed))

    # ---- door-seeking (post-clear navigation) --------------------------

    def _pick_door_move(self, raw_obs: dict[str, Any]) -> int | None:
        """When no visible enemies, choose a movement direction toward an open door.

        Doors obs is a [4, 6] array: rows are LEFT/UP/RIGHT/DOWN slots, columns
        are (exists, is_open, is_locked, is_boss, is_treasure, is_secret).
        Prefers normal doors over special-purpose ones when both are open.
        Returns a movement action (1..8) or None if no viable door.

        DETERMINISTIC selection based on distance to player: picks the CLOSEST
        open door. Previously used random shuffle which caused the bot to
        oscillate between doors each tick (never committing to one).

        BC bias mitigation: instead of randomizing per-call, we rely on the
        natural distribution of nearest-door-across-rooms to spread demos
        across all four cardinal directions. Different rooms have different
        nearest doors, so BC still sees diverse directions.
        """
        cfg = self.cfg
        doors = raw_obs.get("doors")
        if not doors:
            return None

        # Compute nearest open-door slot based on player position.
        bounds = raw_obs.get("room_bounds")
        player = raw_obs.get("player") or {}
        px = float(player.get("x", 0) or 0)
        py = float(player.get("y", 0) or 0)
        if bounds:
            tl_x = float(bounds.get("tl_x", 0) or 0)
            tl_y = float(bounds.get("tl_y", 0) or 0)
            br_x = float(bounds.get("br_x", 1) or 1)
            br_y = float(bounds.get("br_y", 1) or 1)
            mid_x = (tl_x + br_x) / 2.0
            mid_y = (tl_y + br_y) / 2.0
            door_positions = [
                (tl_x, mid_y),   # LEFT
                (mid_x, tl_y),   # UP
                (br_x, mid_y),   # RIGHT
                (mid_x, br_y),   # DOWN
            ]
        else:
            # No room_bounds — fall back to fixed slot order (deterministic).
            door_positions = None

        n_slots = min(4, len(doors))
        # Avoid the door we just came IN through (prevents immediate backtrack).
        # Only skip when there ARE other viable doors; if the entry door is
        # the only open door, we have to use it.
        entered_from = self._entered_from_slot

        # Two passes: normal doors first (if prefer_normal_doors), then any open door.
        for pass_idx in (0, 1):
            # Collect eligible doors with their distances.
            eligible: list[tuple[float, int]] = []
            for slot in range(n_slots):
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
                if door_positions is not None:
                    dx, dy = door_positions[slot]
                    dist = math.hypot(px - dx, py - dy)
                else:
                    dist = float(slot)   # fallback: slot-index order
                eligible.append((dist, slot))
            if not eligible:
                continue
            # Filter out the door we came in through, unless it's the only one.
            if entered_from is not None:
                non_entry = [(d, s) for d, s in eligible if s != entered_from]
                if non_entry:
                    eligible = non_entry
            # Pick nearest. If multiple doors are tied within a small epsilon
            # (e.g. player at exact room center with all 4 doors open),
            # randomize among ties to avoid a persistent LEFT-bias in BC.
            eligible.sort()
            best_dist = eligible[0][0]
            tied = [slot for dist, slot in eligible if dist - best_dist < 1e-3]
            best_slot = int(self._rng.choice(tied)) if len(tied) > 1 else tied[0]
            return cfg.door_slot_to_move[best_slot]
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
