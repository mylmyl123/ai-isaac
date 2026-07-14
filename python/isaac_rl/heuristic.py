"""MINIMAL heuristic for BC warm start (v3, 2026-07-04).

DESIGN: 3 rules, ~60 lines of logic. Nothing else.

  Rule 1: If enemy visible, shoot at nearest one AND move perpendicular to it.
  Rule 2: If no enemy, move toward a randomly-chosen open door.
  Rule 3: Door target locks per room. That's the only state.

DELETED from v2:
  - Stuck-detection (bot gets stuck sometimes; PPO exploration escapes)
  - Wall-rotation during retreat (BC learns from "sometimes stuck at wall")
  - Failed-slot memory (adds complexity for marginal benefit)
  - Entry-slot inference (edge case; sometimes helps, sometimes wrong)
  - 3-zone kite (simplified to perpendicular-sidestep always)
  - Diagonal navigation math for door approach (simple 4-way)
  - Approach mode (bot always sidesteps when enemies visible)

PHILOSOPHY: The heuristic doesn't need to be GOOD. It needs to give BC
"not-random" data. BC learns coarse patterns. PPO refines. Every extra
rule in the heuristic is:
  1. A potential bug source
  2. A pattern BC will faithfully reproduce (including the bugs)
  3. Time NOT spent on RL improvements

If bot gets stuck against a wall in demos, that's FINE. Reset happens
eventually (episode timeout, death, room clear, etc). BC learns to
associate stuck-state with "shoot randomly" and PPO improves it.
"""
from __future__ import annotations
import logging
import math
import os
from dataclasses import dataclass
from typing import Any
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class HeuristicConfig:
    seed: int = 0
    # These are kept as no-ops for backward-compat with existing tests/configs.
    # They don't do anything in v3.
    retreat_dist: float = 100.0
    approach_dist: float = 260.0
    door_align_thresh: float = 15.0
    stuck_ticks: int = 30
    stuck_radius: float = 5.0
    entry_wall_dist: float = 60.0
    door_slot_to_move: tuple = (7, 1, 3, 5)


class HeuristicPolicy:
    """Minimal 3-rule controller. See module docstring."""

    def __init__(self, config: HeuristicConfig | None = None):
        self.cfg = config or HeuristicConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._prev_room_index: int | None = None
        self._target_door_slot: int | None = None
        # For back-compat with old tests that inspect these.
        self._stuck_pos = None
        self._stuck_ticks_count = 0
        self._entry_slot = None
        self._failed_slots: set = set()
        self._debug = bool(os.environ.get("ISAAC_HEURISTIC_DEBUG", "").strip())
        try:
            from isaac_rl.debug_recorder import DebugRecorder
            self._recorder = DebugRecorder.get_instance()
        except ImportError:
            self._recorder = None

    def act(self, raw_obs: dict[str, Any]) -> np.ndarray:
        player = raw_obs.get("player") or {}
        px = float(player.get("x", 0) or 0)
        py = float(player.get("y", 0) or 0)

        # Room change: reset door target.
        gg = raw_obs.get("global") or {}
        cur_room = gg.get("room_index") or gg.get("safe_grid_index")
        if cur_room is not None and cur_room != self._prev_room_index:
            self._prev_room_index = cur_room
            self._target_door_slot = None

        # ---- Rule 1: enemy visible -> shoot + sidestep ----
        enemy = self._nearest_enemy(raw_obs)
        if enemy is not None:
            edx, edy, edist = enemy
            shoot = self._angle_to_shoot(math.atan2(edy, edx))
            # Always sidestep perpendicular to enemy. Simple, no kite zones.
            sign = 1.0 if self._rng.random() < 0.5 else -1.0
            move = self._angle_to_move(math.atan2(sign * edx, -sign * edy))
            action = np.array([move, shoot], dtype=np.int64)
            if self._recorder:
                self._recorder.log_tick(0, raw_obs, action, "combat",
                                        {"target_door": self._target_door_slot, "enemy_dist": round(edist, 1)})
            return action

        # ---- Rule 1.5 (2026-07-13): pickup/pedestal visible -> walk to it ----
        # If a collectible or consumable is in the room and no enemies are
        # visible, walking to it is strictly better than heading for a door.
        # This gives BC data on the 'grab items' behavior that pure door-
        # seeking heuristic v3 never demonstrated. Nearest pickup wins;
        # collectibles preferred over hearts/coins (higher long-run value).
        pickup = self._nearest_pickup(raw_obs, px, py)
        if pickup is not None:
            pdx, pdy, _pdist = pickup
            move = self._angle_to_move(math.atan2(pdy, pdx))
            action = np.array([move, 0], dtype=np.int64)
            if self._recorder:
                self._recorder.log_tick(0, raw_obs, action, "pickup",
                                        {"target_door": self._target_door_slot})
            return action

        # ---- Rule 2 + 3: no enemy -> move toward locked door target ----
        if self._target_door_slot is None:
            self._target_door_slot = self._pick_door(raw_obs)

        if self._target_door_slot is not None:
            move = self._move_toward_door(raw_obs, self._target_door_slot, px, py)
            branch = "door_seek"
        else:
            move = int(self._rng.integers(1, 9))
            branch = "wander"

        action = np.array([move, 0], dtype=np.int64)
        if self._recorder:
            self._recorder.log_tick(0, raw_obs, action, branch,
                                    {"target_door": self._target_door_slot})
        return action

    def _pick_door(self, raw_obs: dict[str, Any]) -> int | None:
        """Random open unlocked door slot. Prefers normal over special."""
        doors = raw_obs.get("doors") or []
        for pass_idx in (0, 1):
            candidates = []
            for slot in range(min(4, len(doors))):
                d = doors[slot] or []
                if len(d) < 6:
                    continue
                if not d[0] or not d[1] or d[2]:   # not exists / not open / locked
                    continue
                is_special = d[3] or d[4] or d[5]
                if pass_idx == 0 and is_special:
                    continue
                candidates.append(slot)
            if candidates:
                return int(self._rng.choice(candidates))
        return None

    def _move_toward_door(self, raw_obs: dict[str, Any], slot: int, px: float, py: float) -> int:
        """Move toward door center. Diagonal if off-axis, cardinal if aligned."""
        bounds = raw_obs.get("room_bounds")
        if not bounds:
            return int(self.cfg.door_slot_to_move[slot])
        tl_x = float(bounds.get("tl_x", 0) or 0)
        tl_y = float(bounds.get("tl_y", 0) or 0)
        br_x = float(bounds.get("br_x", 1) or 1)
        br_y = float(bounds.get("br_y", 1) or 1)
        mid_x = (tl_x + br_x) / 2.0
        mid_y = (tl_y + br_y) / 2.0
        door_pos = [(tl_x, mid_y), (mid_x, tl_y), (br_x, mid_y), (mid_x, br_y)]
        target_x, target_y = door_pos[slot]
        vec_x = target_x - px
        vec_y = target_y - py
        thresh = self.cfg.door_align_thresh
        if slot == 0:
            if vec_y > thresh: return 6
            elif vec_y < -thresh: return 8
            else: return 7
        elif slot == 2:
            if vec_y > thresh: return 4
            elif vec_y < -thresh: return 2
            else: return 3
        elif slot == 1:
            if vec_x > thresh: return 2
            elif vec_x < -thresh: return 8
            else: return 1
        elif slot == 3:
            if vec_x > thresh: return 4
            elif vec_x < -thresh: return 6
            else: return 5
        return int(self.cfg.door_slot_to_move[slot])

    def _nearest_pickup(self, raw_obs: dict[str, Any], px: float, py: float) -> tuple[float, float, float] | None:
        """Return (dx, dy, dist) to the nearest pickup in the room, or None.

        Uses the raw obs's `pickups` field (see mods/isaac-rl-bridge/obs.lua)
        which contains up to MAX_PICKUPS entries with world-space x/y. We
        return raw dx/dy (world units) so the caller can feed atan2 directly.

        Prefers collectibles (variant 100) over consumables when both exist
        — collectibles are permanent stat boosts; hearts/coins are marginal.
        """
        pickups = raw_obs.get("pickups") or {}
        feats = pickups.get("feats") or []
        mask = pickups.get("mask") or []
        best = None
        best_pri = 0
        for i, f in enumerate(feats):
            if i >= len(mask) or not mask[i] or not f or len(f) < 4:
                continue
            # obs.lua feats[i] schema: [variant, subtype, dx_norm, dy_norm, ...]
            variant = int(f[0])
            dx = float(f[2]) * 480.0
            dy = float(f[3]) * 270.0
            d = math.hypot(dx, dy)
            # Priority 2 for collectibles (variant 100), 1 for anything else.
            pri = 2 if variant == 100 else 1
            if best is None or pri > best_pri or (pri == best_pri and d < best[2]):
                best = (dx, dy, d)
                best_pri = pri
        return best

    def _nearest_enemy(self, raw_obs: dict[str, Any]) -> tuple[float, float, float] | None:
        enemies = raw_obs.get("enemies") or {}
        feats = enemies.get("feats") or []
        mask = enemies.get("mask") or []
        best = None
        for i, f in enumerate(feats):
            if i >= len(mask) or not mask[i] or not f or len(f) < 4:
                continue
            dx = float(f[2]) * 480.0
            dy = float(f[3]) * 270.0
            d = math.hypot(dx, dy)
            if best is None or d < best[2]:
                best = (dx, dy, d)
        return best

    @staticmethod
    def _angle_to_shoot(angle: float) -> int:
        if angle > math.pi: angle -= 2 * math.pi
        elif angle < -math.pi: angle += 2 * math.pi
        if -math.pi / 4 <= angle <= math.pi / 4: return 2
        elif math.pi / 4 < angle <= 3 * math.pi / 4: return 3
        elif -3 * math.pi / 4 <= angle < -math.pi / 4: return 1
        else: return 4

    @staticmethod
    def _angle_to_move(angle: float) -> int:
        if angle < 0: angle += 2 * math.pi
        sector = int((angle + math.pi / 8) / (math.pi / 4)) % 8
        return {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 1, 7: 2}[sector]

    # Back-compat stubs for old tests.
    def _infer_entry_slot(self, *args, **kwargs):
        return None

    def _pick_target_door(self, raw_obs):
        return self._pick_door(raw_obs)

    @staticmethod
    def _rotate_move_90(move: int, clockwise: bool = True) -> int:
        if move == 0: return 0
        cw = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 1, 8: 2}
        return cw[move] if clockwise else {v: k for k, v in cw.items()}[move]
