"""Reward shaper for Isaac RL.

Aggressive dense-shaping design for compute-limited training. Combines:
  * Terminal rewards (death, mom kill, floor clear) — high-magnitude anchors
  * Event rewards (kills, damage dealt/taken, pickups, room clear) — sparse gains
  * Dense per-tick shaping:
      - Aim alignment: reward for shooting in the direction of the nearest enemy.
      - Engagement distance: reward for being at kiting distance from enemies.
      - HP preservation: reward for maintaining full HP.
      - Anti-idle: penalty for standing still.
      - Approach potential (PBRS): reward for moving toward ideal engagement dist.
  * Compound bonuses:
      - Fast room clear (< N ticks) bonus.
      - No-damage room clear bonus.

All dense rewards are small (order 0.005 per tick) so they don't dominate the
terminal signals but cumulatively guide the policy toward productive behavior
during the ~1M-5M step regime where compute is limited.

Reference: Ng, Harada, Russell 1999 (PBRS). OpenAI Five (aim alignment).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardConfig:
    # ---- Terminal rewards (unchanged) ----------------------------------
    r_beat_mom: float = 50.0
    r_floor_cleared: float = 5.0
    r_death: float = -10.0

    # ---- Dense combat --------------------------------------------------
    r_damage_dealt_scale: float = 0.1
    r_kill: float = 0.5
    r_damage_taken_red: float = -1.0
    r_damage_taken_other: float = -0.5
    max_damage_reward_per_room: float = 3.0

    # ---- Room / exploration (BOOSTED from originals) -------------------
    r_room_clear: float = 2.0             # was 1.0 — clearing a room is a big deal
    r_new_room: float = 0.2               # was 0.05 — encourage exploration
    r_boss_room_first_entry: float = 0.5
    r_room_clear_speed_bonus: float = 1.0 # bonus if room cleared in < speed_clear_ticks
    speed_clear_ticks: int = 200          # ~13s at 15Hz
    r_room_clear_no_damage: float = 1.5   # bonus for clean clears (encourages dodging)

    # ---- Pickups (unchanged) -------------------------------------------
    r_pickup_heart: float = 0.2
    r_pickup_coin: float = 0.05
    r_pickup_key: float = 0.1
    r_pickup_bomb: float = 0.1
    r_pickup_collectible: float = 1.0

    # ---- Per-step cost -------------------------------------------------
    r_step: float = -0.003                # was -0.001 — 3x to discourage wandering

    # ---- NEW: Aim alignment (proven for shooting games) ----------------
    # When the policy fires the shoot action AND the shoot direction is the
    # correct cardinal quadrant toward the nearest enemy, reward per tick.
    # Turns "randomly shoot in 4 directions" from a lottery into a guided search.
    r_aim_at_enemy: float = 0.03
    r_shoot_when_enemy_visible: float = 0.005   # smaller reward just for shooting when enemies present

    # ---- NEW: Engagement-distance shaping ------------------------------
    # Bot learns to stay at kiting distance from enemies. Too close = enemies hit
    # you, too far = you miss shots. Sweet spot is 100-300 px in Isaac.
    r_at_kite_dist_tick: float = 0.005
    kite_dist_min: float = 100.0
    kite_dist_max: float = 300.0

    # ---- NEW: HP preservation ------------------------------------------
    # Dense reward per tick for being at full HP. Encourages proactive dodging
    # rather than only reacting after damage is taken.
    r_full_hp_tick: float = 0.005

    # ---- NEW: Anti-idle penalty ----------------------------------------
    # Prevents PPO from converging on the "stand still" local optimum where the
    # bot never engages enemies (which happens surprisingly often in early
    # training if step penalty is too small).
    r_idle_penalty: float = -0.01
    idle_speed_threshold: float = 0.5      # velocity magnitude below this = idle

    # ---- NEW: PBRS approach potential ----------------------------------
    # Potential-based reward shaping: F = gamma*Phi(s') - Phi(s) with
    # Phi(s) = 1.0 when at ideal distance, decaying with |dist - ideal|.
    # Provably doesn't change optimal policy (Ng, Harada, Russell 1999).
    # Positive when moving TOWARD ideal engagement distance.
    r_pbrs_scale: float = 0.1              # magnitude multiplier
    pbrs_gamma: float = 0.99               # match trainer gamma for PBRS guarantee
    pbrs_ideal_dist: float = 200.0
    pbrs_decay: float = 200.0              # potential decays over this distance

    # Damage-per-room cap so infinite spawners can't inflate reward.


# Room types where "boss cleared" implies a floor is done.
ROOM_TYPE_BOSS = 5


@dataclass
class RewardState:
    damage_reward_this_room: float = 0.0
    visited_boss_rooms: set = field(default_factory=set)
    prev_hp_red: float = 0.0
    prev_hp_soul: float = 0.0
    prev_hp_black: float = 0.0
    last_stage: int = 0
    dead: bool = False

    # NEW: dense-shaping state
    ticks_since_room_start: int = 0        # for speed-clear bonus
    damage_this_room_red: float = 0.0      # for no-damage clear bonus
    damage_this_room_other: float = 0.0
    prev_potential: float | None = None    # PBRS: previous state potential


class RewardShaper:
    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()
        self.state = RewardState()

    def reset(self) -> None:
        self.state = RewardState()

    # --- helper: cardinal direction from angle ---------------------------
    @staticmethod
    def _angle_to_shoot_action(angle_rad: float) -> int:
        """Map an angle (from atan2) to the shoot action (0=none, 1=up, 2=right, 3=down, 4=left).

        Isaac Y-axis increases downward, so:
          angle in [-pi/4, pi/4]     -> RIGHT (2)
          angle in [pi/4, 3pi/4]     -> DOWN (3)
          angle in [3pi/4, pi] or [-pi, -3pi/4] -> LEFT (4)
          angle in [-3pi/4, -pi/4]   -> UP (1)
        """
        if angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        elif angle_rad < -math.pi:
            angle_rad += 2 * math.pi

        if -math.pi / 4 <= angle_rad <= math.pi / 4:
            return 2  # right
        elif math.pi / 4 < angle_rad <= 3 * math.pi / 4:
            return 3  # down
        elif -3 * math.pi / 4 <= angle_rad < -math.pi / 4:
            return 1  # up
        else:
            return 4  # left

    def _nearest_enemy(self, raw_obs: dict[str, Any]) -> tuple[float, float, float] | None:
        """Return (dx, dy, dist) to the nearest visible enemy, in world units. None if no enemies."""
        enemies = raw_obs.get("enemies") or {}
        feats = enemies.get("feats") or []
        mask = enemies.get("mask") or []
        best: tuple[float, float, float] | None = None
        for i, f in enumerate(feats):
            if i >= len(mask) or not mask[i]:
                continue
            if not f or len(f) < 4:
                continue
            # feats[2], feats[3] are (dx / 480, dy / 270) — reconstruct world units.
            dx = float(f[2]) * 480.0
            dy = float(f[3]) * 270.0
            d = math.hypot(dx, dy)
            if best is None or d < best[2]:
                best = (dx, dy, d)
        return best

    def __call__(
        self,
        raw_obs: dict[str, Any],
        action: Any | None = None,
    ) -> tuple[float, bool, dict[str, float]]:
        """Return (reward, terminated, breakdown).

        action: optional int-array of shape (5,) or None. If provided, enables
        aim-alignment reward. Fields: [move(0-8), shoot(0-4), use_active,
        drop_bomb, pill_card]. See spaces.py.
        """
        cfg = self.cfg
        st = self.state
        reward = cfg.r_step
        breakdown: dict[str, float] = {"step": cfg.r_step}

        def add(name: str, x: float) -> None:
            nonlocal reward
            reward += x
            breakdown[name] = breakdown.get(name, 0.0) + x

        # ---- Player state -------------------------------------------------
        player = raw_obs.get("player") or {}
        cur_red = float(player.get("hp_red", 0) or 0)
        cur_soul = float(player.get("hp_soul", 0) or 0)
        cur_black = float(player.get("hp_black", 0) or 0)
        max_hp = float(player.get("hp_max", 0) or 0)
        vx = float(player.get("vx", 0) or 0)
        vy = float(player.get("vy", 0) or 0)
        speed = math.hypot(vx, vy)

        # ---- HP-delta damage-taken (also feeds damage-this-room) ----------
        if st.prev_hp_red or st.prev_hp_soul or st.prev_hp_black:
            d_red = st.prev_hp_red - cur_red
            d_other = (st.prev_hp_soul - cur_soul) + (st.prev_hp_black - cur_black)
            if d_red > 0:
                add("hp_delta_red", d_red * cfg.r_damage_taken_red)
                st.damage_this_room_red += d_red
            if d_other > 0:
                add("hp_delta_other", d_other * cfg.r_damage_taken_other)
                st.damage_this_room_other += d_other
        st.prev_hp_red = cur_red
        st.prev_hp_soul = cur_soul
        st.prev_hp_black = cur_black

        # ---- Events (kills, pickups, room/level, death) -------------------
        terminated = False
        room_cleared_this_tick = False
        for evt in raw_obs.get("events") or []:
            kind = evt.get("kind")

            if kind == "damage_to_npc":
                dmg = float(evt.get("dmg", 0) or 0)
                npc_max_hp = float(evt.get("npc_max_hp", 1) or 1) or 1.0
                gain = cfg.r_damage_dealt_scale * min(dmg, npc_max_hp) / npc_max_hp
                room_left = max(0.0, cfg.max_damage_reward_per_room - st.damage_reward_this_room)
                gain = min(gain, room_left)
                st.damage_reward_this_room += gain
                add("damage_dealt", gain)
                if evt.get("killed"):
                    add("kill", cfg.r_kill)

            elif kind == "damage_to_player":
                pass  # handled via HP-delta above

            elif kind == "pickup_heart":
                add("pickup_heart", cfg.r_pickup_heart)
            elif kind == "pickup_coin":
                add("pickup_coin", cfg.r_pickup_coin)
            elif kind == "pickup_key":
                add("pickup_key", cfg.r_pickup_key)
            elif kind == "pickup_bomb":
                add("pickup_bomb", cfg.r_pickup_bomb)
            elif kind == "pickup_collectible":
                add("pickup_collectible", cfg.r_pickup_collectible)

            elif kind == "new_room":
                if evt.get("is_new"):
                    add("new_room", cfg.r_new_room)
                # Reset per-room shaping state
                st.damage_reward_this_room = 0.0
                st.ticks_since_room_start = 0
                st.damage_this_room_red = 0.0
                st.damage_this_room_other = 0.0
                if evt.get("room_type") == ROOM_TYPE_BOSS:
                    sgi = evt.get("safe_grid_index")
                    if sgi is not None and sgi not in st.visited_boss_rooms:
                        st.visited_boss_rooms.add(sgi)
                        add("boss_room_first_entry", cfg.r_boss_room_first_entry)

            elif kind == "room_clear":
                add("room_clear", cfg.r_room_clear)
                room_cleared_this_tick = True
                # Compound bonuses
                if st.ticks_since_room_start > 0 and st.ticks_since_room_start <= cfg.speed_clear_ticks:
                    add("room_clear_speed", cfg.r_room_clear_speed_bonus)
                if st.damage_this_room_red == 0 and st.damage_this_room_other == 0:
                    add("room_clear_no_damage", cfg.r_room_clear_no_damage)

            elif kind == "new_level":
                stage = int(evt.get("stage", 0) or 0)
                if stage > st.last_stage and st.last_stage > 0:
                    add("floor_cleared", cfg.r_floor_cleared)
                if st.last_stage == 6 and stage == 7:
                    add("beat_mom", cfg.r_beat_mom)
                    terminated = True
                st.last_stage = stage

            elif kind == "death":
                if not st.dead:
                    st.dead = True
                    add("death", cfg.r_death)
                    terminated = True

            elif kind == "crash":
                # Env-side synthetic event on Isaac process crash
                add("crash_penalty", -1.0)
                terminated = True

        # ---- Dense per-tick shaping (skip if terminated this tick) --------
        if not terminated:
            st.ticks_since_room_start += 1

            # HP preservation: reward per tick at full HP.
            if max_hp > 0 and cur_red >= max_hp:
                add("full_hp_tick", cfg.r_full_hp_tick)

            # Anti-idle penalty.
            if speed < cfg.idle_speed_threshold:
                add("idle_penalty", cfg.r_idle_penalty)

            # Nearest-enemy dependent shaping (aim, kite distance, PBRS).
            enemy = self._nearest_enemy(raw_obs)
            if enemy is not None:
                edx, edy, edist = enemy

                # Engagement distance: reward for being in [kite_min, kite_max]
                if cfg.kite_dist_min <= edist <= cfg.kite_dist_max:
                    add("at_kite_dist", cfg.r_at_kite_dist_tick)

                # Aim alignment: if we're shooting AND direction matches nearest enemy
                if action is not None:
                    try:
                        shoot_action = int(action[1]) if hasattr(action, "__getitem__") else int(action.shoot if hasattr(action, "shoot") else 0)
                    except (IndexError, TypeError, ValueError):
                        shoot_action = 0
                    if shoot_action != 0:
                        # Small reward just for shooting when enemies are visible
                        add("shoot_when_enemy", cfg.r_shoot_when_enemy_visible)
                        # Big reward if aim matches enemy direction
                        ideal = self._angle_to_shoot_action(math.atan2(edy, edx))
                        if shoot_action == ideal:
                            add("aim_at_enemy", cfg.r_aim_at_enemy)

                # PBRS approach potential.
                # Phi(s) is high when at ideal_dist, decays with distance from ideal.
                phi = math.exp(-abs(edist - cfg.pbrs_ideal_dist) / max(1.0, cfg.pbrs_decay))
                if st.prev_potential is not None:
                    # F = gamma * Phi(s') - Phi(s), scaled
                    pbrs = cfg.r_pbrs_scale * (cfg.pbrs_gamma * phi - st.prev_potential)
                    add("pbrs_approach", pbrs)
                st.prev_potential = phi
            else:
                # No enemies: reset potential so we don't get spurious PBRS on next enemy encounter
                st.prev_potential = None

        return float(reward), bool(terminated), breakdown
