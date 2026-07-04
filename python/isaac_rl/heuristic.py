"""Simple heuristic policy for BC bootstrap and PPO kickstarting.

DESIGN PRINCIPLES (2026-07-04 REWRITE):

The previous heuristic (615 lines, 8+ patches over one session) accumulated
compounding bugs from stacking multiple state-based decision layers with
contradictory objectives. Audit found 4 HIGH-severity oscillation bugs:

  1. Stuck-detection flipped between opposite directions.
  2. Wall-avoidance forbade the direction of an open door.
  3. Nearest-door target flipped at tie boundaries as player moved.
  4. Entry-slot inference from "nearest wall" was often wrong.

Rewrite principles:

  * STATELESS per-tick where possible. State variables reset on clear events
    (room change), never on tick counts or "stuck" heuristics.
  * LOCK the door target per room. Recompute only on room change.
  * NO wall-avoidance. If the door is on the wall, we walk to the wall.
    Isaac's collision handles stopping at obstacles; BC learns from that.
  * NO stuck-detection. If bot gets stuck, PPO's entropy explores around it.
  * KITING has 3 states only: too close (retreat), too far (approach), hold.
  * SHOOT always when enemy visible. Isaac tears are cheap.

Total: ~200 lines including all helpers and docstrings.

This heuristic is NOT optimal — it's meant to give BC a "not-random" starting
point. PPO refines from here. If the heuristic is 40% competent, BC bootstraps
PPO to a state where entropy exploration finds better strategies faster than
random init + reward shaping alone.

If demos look bad (bot obviously stuck), that's OK for BC. What we can't
tolerate is OSCILLATION (bot flipping between actions each tick), because BC
learns oscillation as the "correct" policy and PPO amplifies it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# Action space:
#   move: 0=idle, 1=up, 2=up-right, 3=right, 4=down-right,
#         5=down, 6=down-left, 7=left, 8=up-left
#   shoot: 0=none, 1=up, 2=right, 3=down, 4=left
#
# Isaac Y-axis increases DOWNWARD (top-left origin), which is why:
#   angle < 0 => enemy above bot => shoot UP (action=1)
#   angle in [pi/4, 3pi/4] => enemy below bot => shoot DOWN (action=3)


@dataclass
class HeuristicConfig:
    # Combat distance thresholds (world units).
    retreat_dist: float = 100.0
    approach_dist: float = 260.0
    # Below retreat_dist: bot flees enemy.
    # retreat_dist..approach_dist: bot holds position, shoots.
    # Above approach_dist: bot advances toward enemy.

    # Door alignment threshold (world units). Once the bot's perpendicular
    # coordinate is within this of the door center, push straight through.
    # Wider than align_thresh in v1 (was 20) — hysteresis prevents jitter.
    door_align_thresh: float = 50.0

    # RNG seed for tie-breaking.
    seed: int = 0

    # Which slot maps to which move action.
    # Slots: 0=LEFT, 1=UP, 2=RIGHT, 3=DOWN
    # Moves: 7=LEFT,  1=UP,  3=RIGHT,  5=DOWN
    door_slot_to_move: tuple = (7, 1, 3, 5)


class HeuristicPolicy:
    """Stateless-per-tick controller for Isaac.

    State stored across ticks:
      _prev_room_index: last room seen (to detect room transitions)
      _target_door_slot: currently-locked door target (0..3 or None)
      _rng: numpy RNG for tie-breaking

    That's it. No stuck counters, no direction memory, no forbidden-direction
    sets, no entry-slot inference.
    """

    def __init__(self, config: HeuristicConfig | None = None):
        self.cfg = config or HeuristicConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._prev_room_index: int | None = None
        self._target_door_slot: int | None = None
        # Debug logging toggle.
        import os
        self._debug = bool(os.environ.get("ISAAC_HEURISTIC_DEBUG", "").strip())

    # ---- Main decision --------------------------------------------------

    def act(self, raw_obs: dict[str, Any]) -> np.ndarray:
        """Return action ndarray [move, shoot]."""
        # Detect room change: reset locked door target.
        cur_room = None
        gg = raw_obs.get("global") or {}
        cur_room = gg.get("room_index") or gg.get("safe_grid_index")
        if cur_room is not None and cur_room != self._prev_room_index:
            self._prev_room_index = cur_room
            self._target_door_slot = None   # force re-pick in new room

        # ---- Aim + shoot at nearest enemy ----
        enemy = self._nearest_enemy(raw_obs)
        shoot = 0
        move = 0

        if enemy is not None:
            edx, edy, edist = enemy
            # Shoot in the direction of the enemy (Isaac only has 4 cardinal
            # shoot directions; angle_to_shoot maps to the nearest).
            shoot = self._angle_to_shoot(math.atan2(edy, edx))

            # 3-zone kiting: too close -> flee, too far -> approach, else hold.
            if edist < self.cfg.retreat_dist:
                # Flee: move directly away from enemy.
                move = self._angle_to_move(math.atan2(-edy, -edx))
            elif edist > self.cfg.approach_dist:
                # Approach: move toward enemy.
                move = self._angle_to_move(math.atan2(edy, edx))
            else:
                # Hold position, keep shooting. Do not idle — small random
                # nudge to avoid becoming pinned by touch damage. Perpendicular
                # to enemy so we sidestep, not toward.
                # Rotate enemy direction by 90 degrees (either +90 or -90).
                sign = 1.0 if self._rng.random() < 0.5 else -1.0
                move = self._angle_to_move(math.atan2(sign * edx, -sign * edy))

            if self._debug:
                log.info("[heuristic] combat: enemy dist=%.0f -> move=%d shoot=%d",
                         edist, move, shoot)
            return np.array([move, shoot], dtype=np.int64)

        # ---- No enemies: navigate to a door ----
        # Pick a target door if we don't have one for this room.
        if self._target_door_slot is None:
            self._target_door_slot = self._pick_target_door(raw_obs)

        if self._target_door_slot is not None:
            move = self._move_toward_door(raw_obs, self._target_door_slot)
        else:
            # No open doors visible. Wander randomly.
            move = int(self._rng.integers(1, 9))

        if self._debug:
            log.info("[heuristic] no-enemies: door_slot=%s -> move=%d",
                     self._target_door_slot, move)
        return np.array([move, shoot], dtype=np.int64)

    # ---- Door target selection (called once per room) --------------------

    def _pick_target_door(self, raw_obs: dict[str, Any]) -> int | None:
        """Choose ONE door slot to target for this room. Locked in until room change.

        Prefers normal doors over boss/treasure/secret (which typically require
        specific conditions). Among viable doors, picks randomly to avoid a
        LEFT-bias in demos.
        """
        doors = raw_obs.get("doors")
        if not doors:
            return None
        # First pass: normal open unlocked doors only.
        # Second pass: any open unlocked doors (boss/treasure/secret included).
        for pass_idx in (0, 1):
            candidates: list[int] = []
            for slot in range(min(4, len(doors))):
                d = doors[slot]
                if not d or len(d) < 6:
                    continue
                exists = bool(d[0])
                is_open = bool(d[1])
                is_locked = bool(d[2])
                is_special = bool(d[3]) or bool(d[4]) or bool(d[5])
                if not exists or not is_open or is_locked:
                    continue
                if pass_idx == 0 and is_special:
                    continue
                candidates.append(slot)
            if candidates:
                # Random pick among viable doors. RNG-seeded so runs are
                # reproducible. Randomization here (rather than nearest-door)
                # ensures demos see all door slots as targets over many rooms,
                # preventing BC from learning a positional bias.
                return int(self._rng.choice(candidates))
        return None

    # ---- Door approach (called each tick until door crossed) -------------

    def _move_toward_door(self, raw_obs: dict[str, Any], slot: int) -> int:
        """Return the move action that best approaches the door center.

        Uses room_bounds to compute the door's world position. Navigates
        diagonally when player is off-axis, cardinal when aligned.
        """
        bounds = raw_obs.get("room_bounds")
        player = raw_obs.get("player") or {}
        px = float(player.get("x", 0) or 0)
        py = float(player.get("y", 0) or 0)

        if not bounds:
            # No bounds info — just push in the cardinal direction of the slot.
            return int(self.cfg.door_slot_to_move[slot])

        tl_x = float(bounds.get("tl_x", 0) or 0)
        tl_y = float(bounds.get("tl_y", 0) or 0)
        br_x = float(bounds.get("br_x", 1) or 1)
        br_y = float(bounds.get("br_y", 1) or 1)
        mid_x = (tl_x + br_x) / 2.0
        mid_y = (tl_y + br_y) / 2.0

        # Door center positions in world coordinates.
        door_pos = [
            (tl_x, mid_y),   # 0 = LEFT
            (mid_x, tl_y),   # 1 = UP
            (br_x, mid_y),   # 2 = RIGHT
            (mid_x, br_y),   # 3 = DOWN
        ]
        target_x, target_y = door_pos[slot]
        vec_x = target_x - px
        vec_y = target_y - py

        thresh = self.cfg.door_align_thresh
        # For LEFT/RIGHT doors: y-align is what matters. For UP/DOWN: x-align.
        if slot == 0:   # LEFT
            if vec_y > thresh:    return 6   # down-left
            elif vec_y < -thresh: return 8   # up-left
            else:                 return 7   # pure left
        elif slot == 2:  # RIGHT
            if vec_y > thresh:    return 4   # down-right
            elif vec_y < -thresh: return 2   # up-right
            else:                 return 3   # pure right
        elif slot == 1:  # UP
            if vec_x > thresh:    return 2   # up-right
            elif vec_x < -thresh: return 8   # up-left
            else:                 return 1   # pure up
        elif slot == 3:  # DOWN
            if vec_x > thresh:    return 4   # down-right
            elif vec_x < -thresh: return 6   # down-left
            else:                 return 5   # pure down
        return int(self.cfg.door_slot_to_move[slot])

    # ---- Enemy detection ------------------------------------------------

    def _nearest_enemy(self, raw_obs: dict[str, Any]) -> tuple[float, float, float] | None:
        """Return (dx, dy, dist) to nearest visible enemy, or None."""
        enemies = raw_obs.get("enemies") or {}
        feats = enemies.get("feats") or []
        mask = enemies.get("mask") or []
        best: tuple[float, float, float] | None = None
        for i, f in enumerate(feats):
            if i >= len(mask) or not mask[i]:
                continue
            if not f or len(f) < 4:
                continue
            # feats[2], feats[3] are (dx / 480, dy / 270) — reconstruct.
            dx = float(f[2]) * 480.0
            dy = float(f[3]) * 270.0
            d = math.hypot(dx, dy)
            if best is None or d < best[2]:
                best = (dx, dy, d)
        return best

    # ---- Angle -> action mappings ---------------------------------------

    @staticmethod
    def _angle_to_shoot(angle: float) -> int:
        """Map atan2(dy, dx) to shoot action 1-4 (up, right, down, left).

        Isaac Y-axis increases downward, so angle sectors:
          [-pi/4, pi/4]     -> RIGHT (2)
          [pi/4, 3pi/4]     -> DOWN (3)   (positive y = below in Isaac)
          [3pi/4, pi] or [-pi, -3pi/4] -> LEFT (4)
          [-3pi/4, -pi/4]   -> UP (1)
        """
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        if -math.pi / 4 <= angle <= math.pi / 4:
            return 2  # right
        elif math.pi / 4 < angle <= 3 * math.pi / 4:
            return 3  # down
        elif -3 * math.pi / 4 <= angle < -math.pi / 4:
            return 1  # up
        else:
            return 4  # left

    @staticmethod
    def _angle_to_move(angle: float) -> int:
        """Map atan2(dy, dx) to 8-way move action 1-8. Never returns 0 (idle)."""
        if angle < 0:
            angle += 2 * math.pi
        # 8 sectors of pi/4, offset by pi/8 so cardinals land in sector centers.
        sector = int((angle + math.pi / 8) / (math.pi / 4)) % 8
        # Sector -> action (Isaac Y-down):
        #   0=right, 1=down-right, 2=down, 3=down-left,
        #   4=left, 5=up-left, 6=up, 7=up-right
        # Action codes: 1=up, 2=up-right, 3=right, 4=down-right,
        #               5=down, 6=down-left, 7=left, 8=up-left
        return {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 1, 7: 2}[sector]
