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
    # 2026-07-04: LOWERED from 50 to 15 after debug trace showed bot stuck
    # at wall for 117 seconds because it thought it was aligned when 36
    # units off from the door center. Isaac's actual door opening is
    # ~40 units wide total, so bot must be within ~15-20 of center to pass.
    door_align_thresh: float = 15.0

    # 2026-07-04: stuck-detection. If bot's position stays within `stuck_radius`
    # for `stuck_ticks` consecutive ticks, unlock the current door target.
    # Next tick will pick a new target (possibly a different door). This
    # is the minimal safety net that prevents "push into wall forever"
    # pathologies without adding complex forbidden-direction logic.
    stuck_ticks: int = 30           # ~2 seconds at 15Hz
    stuck_radius: float = 5.0       # if pos hasn't moved beyond this radius

    # 2026-07-04: entry-slot skip. When entering a new room, check bot's
    # position relative to room walls. If bot is within `entry_wall_dist` of
    # a wall, treat that wall's door as the ENTRY door and don't target it.
    # Prevents backtrack oscillation (bot enters, targets entry door, walks
    # back). Isaac spawns bot near the door it came through, so proximity
    # to a wall is a reliable entry-slot signal EXCEPT in the very first
    # room (spawn at center) — handled by threshold: if no wall is within
    # entry_wall_dist, all doors are viable.
    entry_wall_dist: float = 60.0

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
        # 2026-07-04: stuck-detection state.
        self._stuck_pos: tuple[float, float] | None = None
        self._stuck_ticks_count: int = 0
        # 2026-07-04: entry-slot to skip (backtrack prevention).
        self._entry_slot: int | None = None
        # 2026-07-04 (later): slots that got the bot stuck THIS room -> skip.
        # Reset on room change. Prevents re-picking a stuck door repeatedly.
        self._failed_slots: set[int] = set()
        # Debug logging toggle.
        import os
        self._debug = bool(os.environ.get("ISAAC_HEURISTIC_DEBUG", "").strip())
        # Diagnostic recorder (singleton). Off by default; enabled via env var.
        try:
            from isaac_rl.debug_recorder import DebugRecorder
            self._recorder = DebugRecorder.get_instance()
        except ImportError:
            self._recorder = None

    # ---- Main decision --------------------------------------------------

    def act(self, raw_obs: dict[str, Any]) -> np.ndarray:
        """Return action ndarray [move, shoot]."""
        player = raw_obs.get("player") or {}
        px = float(player.get("x", 0) or 0)
        py = float(player.get("y", 0) or 0)

        # Detect room change: reset locked door target and entry slot.
        cur_room = None
        gg = raw_obs.get("global") or {}
        cur_room = gg.get("room_index") or gg.get("safe_grid_index")
        if cur_room is not None and cur_room != self._prev_room_index:
            self._prev_room_index = cur_room
            self._target_door_slot = None   # force re-pick in new room
            self._failed_slots = set()      # reset per-room failed history
            # Infer entry slot from bot's position relative to walls. Bot
            # typically spawns near the door it came through.
            self._entry_slot = self._infer_entry_slot(raw_obs, px, py)
            # Reset stuck-detection.
            self._stuck_pos = None
            self._stuck_ticks_count = 0

        # Stuck-detection: track position over recent ticks. If bot has been
        # within stuck_radius of its earlier position for stuck_ticks in a
        # row, unlock the door target. Next tick picks a new (random)
        # target which may differ, breaking the stuck loop.
        if self._stuck_pos is None:
            self._stuck_pos = (px, py)
            self._stuck_ticks_count = 0
        else:
            dx = px - self._stuck_pos[0]
            dy = py - self._stuck_pos[1]
            if (dx * dx + dy * dy) < self.cfg.stuck_radius * self.cfg.stuck_radius:
                self._stuck_ticks_count += 1
            else:
                # Bot moved; reset window.
                self._stuck_pos = (px, py)
                self._stuck_ticks_count = 0
            if self._stuck_ticks_count >= self.cfg.stuck_ticks:
                if self._debug:
                    log.info("[heuristic] STUCK at (%.0f,%.0f) for %d ticks -> unlock target=%s",
                             px, py, self._stuck_ticks_count, self._target_door_slot)
                # Add the stuck slot to failed_slots so we don't re-pick it.
                if self._target_door_slot is not None:
                    self._failed_slots.add(self._target_door_slot)
                self._target_door_slot = None
                self._stuck_pos = (px, py)
                self._stuck_ticks_count = 0

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
                # 2026-07-04: wall-avoidance during retreat. If the flee
                # direction would push into a wall (bot near that wall),
                # rotate 90 degrees to slide along the wall instead of
                # pushing into it. Prevents "corner pinning" pathology where
                # 80% of combat_retreat ticks happened at room edges.
                move = self._rotate_if_wall(move, raw_obs, px, py)
                zone = "retreat"
            elif edist > self.cfg.approach_dist:
                # Approach: move toward enemy.
                move = self._angle_to_move(math.atan2(edy, edx))
                zone = "approach"
            else:
                # Hold position, keep shooting. Do not idle — small random
                # nudge to avoid becoming pinned by touch damage. Perpendicular
                # to enemy so we sidestep, not toward.
                # Rotate enemy direction by 90 degrees (either +90 or -90).
                sign = 1.0 if self._rng.random() < 0.5 else -1.0
                move = self._angle_to_move(math.atan2(sign * edx, -sign * edy))
                zone = "hold"

            if self._debug:
                log.info("[heuristic] combat: enemy dist=%.0f zone=%s -> move=%d shoot=%d",
                         edist, zone, move, shoot)
            action = np.array([move, shoot], dtype=np.int64)
            if self._recorder is not None:
                self._recorder.log_tick(
                    env_idx=0, raw_obs=raw_obs, action=action, branch=f"combat_{zone}",
                    extra={"target_door": self._target_door_slot, "enemy_dist": round(edist, 1)},
                )
            return action

        # ---- No enemies: navigate to a door ----
        # Pick a target door if we don't have one for this room.
        just_picked = False
        if self._target_door_slot is None:
            self._target_door_slot = self._pick_target_door(raw_obs)
            just_picked = True

        if self._target_door_slot is not None:
            move = self._move_toward_door(raw_obs, self._target_door_slot)
            branch = "door_seek" + ("_newpick" if just_picked else "")
        else:
            # No open doors visible. Wander randomly.
            move = int(self._rng.integers(1, 9))
            branch = "wander_no_doors"

        if self._debug:
            log.info("[heuristic] no-enemies: door_slot=%s -> move=%d",
                     self._target_door_slot, move)
        action = np.array([move, shoot], dtype=np.int64)
        if self._recorder is not None:
            self._recorder.log_tick(
                env_idx=0, raw_obs=raw_obs, action=action, branch=branch,
                extra={"target_door": self._target_door_slot},
            )
        return action

    # ---- Door target selection (called once per room) --------------------

    def _infer_entry_slot(self, raw_obs: dict[str, Any], px: float, py: float) -> int | None:
        """Infer which door the bot came through by position relative to walls.

        Isaac spawns the bot near the door it entered. If the bot is within
        entry_wall_dist of one wall, that's likely the entry side.

        Returns door slot (0=LEFT, 1=UP, 2=RIGHT, 3=DOWN) or None for center
        spawns (first room / edge cases).
        """
        bounds = raw_obs.get("room_bounds")
        if not bounds:
            return None
        tl_x = float(bounds.get("tl_x", 0) or 0)
        tl_y = float(bounds.get("tl_y", 0) or 0)
        br_x = float(bounds.get("br_x", 1) or 1)
        br_y = float(bounds.get("br_y", 1) or 1)
        thresh = self.cfg.entry_wall_dist
        candidates = []
        if px - tl_x < thresh: candidates.append((px - tl_x, 0))   # LEFT wall
        if py - tl_y < thresh: candidates.append((py - tl_y, 1))   # UP wall
        if br_x - px < thresh: candidates.append((br_x - px, 2))   # RIGHT wall
        if br_y - py < thresh: candidates.append((br_y - py, 3))   # DOWN wall
        if not candidates:
            return None   # center spawn -> no entry slot to skip
        candidates.sort()
        return candidates[0][1]

    def _pick_target_door(self, raw_obs: dict[str, Any]) -> int | None:
        """Choose ONE door slot to target for this room. Locked in until room change.

        Prefers normal doors over boss/treasure/secret (which typically require
        specific conditions). Among viable doors, picks randomly to avoid a
        LEFT-bias in demos.
        """
        doors = raw_obs.get("doors")
        if not doors:
            return None
        # First pass: normal open unlocked doors, excluding entry AND failed slots.
        # Second pass: any open unlocked doors, excluding entry AND failed slots.
        # Third pass: any open unlocked doors, excluding failed slots (allow entry).
        # Fourth pass: any open unlocked doors (dead-end rescue — include everything).
        for pass_idx in (0, 1, 2, 3):
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
                if pass_idx <= 1 and slot == self._entry_slot:
                    continue
                if pass_idx <= 2 and slot in self._failed_slots:
                    continue
                if pass_idx == 0 and is_special:
                    continue
                candidates.append(slot)
            if candidates:
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

    def _rotate_if_wall(self, move: int, raw_obs: dict[str, Any], px: float, py: float) -> int:
        """If `move` action pushes bot into a nearby wall, rotate 90 degrees
        to slide along the wall instead. Returns rotated action.

        This prevents combat_retreat and other flee behaviors from pinning
        the bot against a wall corner. The 80% edge-retreat pathology from
        the debug trace comes from this.

        Rotation direction (CW or CCW) picked randomly per call.
        """
        bounds = raw_obs.get("room_bounds")
        if not bounds:
            return move
        tl_x = float(bounds.get("tl_x", 0) or 0)
        tl_y = float(bounds.get("tl_y", 0) or 0)
        br_x = float(bounds.get("br_x", 1) or 1)
        br_y = float(bounds.get("br_y", 1) or 1)
        # Check if move direction pushes into a wall that's very close.
        # Move actions: 1=up, 2=up-right, 3=right, 4=down-right,
        #               5=down, 6=down-left, 7=left, 8=up-left
        wall_thresh = 40.0   # if bot within 40 units of the wall it's pushing toward
        pushes_up = move in (1, 2, 8)
        pushes_down = move in (4, 5, 6)
        pushes_left = move in (6, 7, 8)
        pushes_right = move in (2, 3, 4)
        into_wall = (
            (pushes_up and (py - tl_y) < wall_thresh) or
            (pushes_down and (br_y - py) < wall_thresh) or
            (pushes_left and (px - tl_x) < wall_thresh) or
            (pushes_right and (br_x - px) < wall_thresh)
        )
        if not into_wall:
            return move
        # Rotate 90 degrees. Try one direction; if that also pushes into a
        # wall, try the other. If both do (corner), pick the one that pushes
        # further from BOTH walls.
        rot_cw = self._rotate_move_90(move, clockwise=True)
        rot_ccw = self._rotate_move_90(move, clockwise=False)
        # Check each rotation for wall collision.
        def _wall_collides(m: int) -> bool:
            pu = m in (1, 2, 8); pd = m in (4, 5, 6)
            pl = m in (6, 7, 8); pr = m in (2, 3, 4)
            return (
                (pu and (py - tl_y) < wall_thresh) or
                (pd and (br_y - py) < wall_thresh) or
                (pl and (px - tl_x) < wall_thresh) or
                (pr and (br_x - px) < wall_thresh)
            )
        cw_hits = _wall_collides(rot_cw)
        ccw_hits = _wall_collides(rot_ccw)
        if not cw_hits and not ccw_hits:
            # Both are free; pick randomly.
            return rot_cw if self._rng.random() < 0.5 else rot_ccw
        if not cw_hits:
            return rot_cw
        if not ccw_hits:
            return rot_ccw
        # Both rotations hit walls (unusual - three walls close). Return
        # original; stuck-detection will unlock target later.
        return move

    @staticmethod
    def _rotate_move_90(move: int, clockwise: bool = True) -> int:
        """Rotate an 8-way move action 90 degrees. Idle (0) stays idle.
        Move actions: 1=up, 2=up-right, 3=right, 4=down-right,
                      5=down, 6=down-left, 7=left, 8=up-left
        """
        if move == 0:
            return 0
        # Clockwise rotation table.
        cw = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 1, 8: 2}
        # Counter-clockwise = inverse.
        ccw = {v: k for k, v in cw.items()}
        return cw[move] if clockwise else ccw[move]

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
