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
    r_death: float = -3.0                 # was -10.0 — reduced to prevent "delay-via-discount" camping
    # Note: -10 with gamma=0.99 means delaying death by 200 ticks makes the
    # discounted penalty 7x less bad (0.99^200 ~= 0.13). Bot exploits this by
    # camping in unreachable spots to defer death. -3 keeps death painful
    # without dominating the discounted value calculation.

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
    r_shoot_when_enemy_visible: float = 0.0     # DISABLED: was rewarding tear spam. Kept for config compatibility.

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

    # ---- NEW: Anti-camping (position-based) ----------------------------
    # Even with the idle penalty, PPO can find local optima where the bot
    # oscillates within a tiny radius ("wiggles" in a corner while shooting).
    # The idle penalty misses this because velocity is briefly non-zero. This
    # tracks the bots position over the last stationary_window ticks and
    # penalises staying inside a small radius — works even for wigglers.
    r_stationary_penalty: float = -0.02
    stationary_radius: float = 40.0        # world units
    stationary_window: int = 45            # ticks (~3s at 15Hz)

    # ---- NEW: Door-seeking (post-clear navigation) ---------------------
    # When the room is clear, the bot has no combat signal. Without this,
    # PPO only gets +r_new_room (0.2 by default) at the moment of crossing
    # a door — too sparse for random-walk to find reliably. Fires a small
    # per-tick reward when the bot's velocity is aligned with an open door
    # slot's direction. Combined with a boosted r_new_room, this gives a
    # dense gradient guiding the bot to exits.
    r_seek_door_when_clear: float = 0.05   # per tick, when velocity is toward an open door
    seek_door_speed_threshold: float = 0.2 # only credit motion when actually moving
    # Potential-based shaping on distance-to-nearest-door when room is clear.
    # Unlike r_seek_door_when_clear (which requires motion), this reward is
    # STATE-BASED: the bot gets reward simply for BEING closer to a door.
    # Fires per-tick as (prev_distance - current_distance) * scale. Positive
    # when closing distance, negative when moving away. Zero when stationary.
    # This is what breaks the "jitter in cleared room" pathology: the bot
    # gets clear signal that positions closer to doors are better, even
    # before it's moving toward one.
    r_door_distance_shaping: float = 0.3   # scales the delta-distance reward
    # Bigger idle penalty specifically when room is clear (there's no tactical
    # reason to hesitate; every idle tick is wasted time). Additive to the
    # normal idle_penalty.
    r_clear_room_idle_extra: float = -0.03

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
    prev_door_dist: float | None = None    # for r_door_distance_shaping
    pos_history: list = field(default_factory=list)  # rolling position window


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

            # Anti-camping (position-based). Tracks bots position over a rolling
            # window; if the max displacement across the window is below
            # stationary_radius, the bot is effectively camping and pays the
            # penalty. Fires even when the bot wiggles (idle_penalty would miss).
            cur_pos = (float(player.get("x", 0) or 0), float(player.get("y", 0) or 0))
            st.pos_history.append(cur_pos)
            if len(st.pos_history) > cfg.stationary_window:
                st.pos_history.pop(0)
            if len(st.pos_history) >= cfg.stationary_window:
                xs = [p[0] for p in st.pos_history]
                ys = [p[1] for p in st.pos_history]
                span = max(max(xs) - min(xs), max(ys) - min(ys))
                if span < cfg.stationary_radius:
                    add("stationary_penalty", cfg.r_stationary_penalty)

            # ---- Door-seeking when room is clear ------------------------
            # If room is clear and the bot's velocity is aligned with an open
            # door direction, give a per-tick reward. Combined with the (now
            # boosted) r_new_room, this gives a dense signal guiding the bot
            # to exits after finishing combat. Without it, cleared-room
            # navigation relies on random walk stumbling into a door — slow.
            is_clear = bool((raw_obs.get("global") or {}).get("is_clear", False))

            # ---- Potential-based shaping: distance to nearest open door ---
            # Fires per-tick as (prev_door_dist - current_door_dist) * scale.
            # State-based (not motion-based), so it fires even when the bot
            # is stationary near a door. Combined with r_seek_door_when_clear
            # (which requires motion), this breaks the "jitter in cleared
            # room" pathology: bot gets clear signal that door-adjacent
            # positions are better regardless of whether it's moving.
            # Only active in cleared rooms; during combat, positioning is
            # tactical, not door-focused.
            if is_clear:
                # Compute distance to nearest open door.
                doors_pbrs = raw_obs.get("doors") or []
                bounds_pbrs = raw_obs.get("room_bounds") or {}
                tl_x_p = float(bounds_pbrs.get("tl_x", 0) or 0)
                tl_y_p = float(bounds_pbrs.get("tl_y", 0) or 0)
                br_x_p = float(bounds_pbrs.get("br_x", 1) or 1)
                br_y_p = float(bounds_pbrs.get("br_y", 1) or 1)
                door_positions = [
                    (tl_x_p, (tl_y_p + br_y_p) / 2.0),                # LEFT
                    ((tl_x_p + br_x_p) / 2.0, tl_y_p),                # UP
                    (br_x_p, (tl_y_p + br_y_p) / 2.0),                # RIGHT
                    ((tl_x_p + br_x_p) / 2.0, br_y_p),                # DOWN
                ]
                px = float(player.get("x", 0) or 0)
                py = float(player.get("y", 0) or 0)
                # Room diagonal for normalization.
                room_diag = max(1.0, ((br_x_p - tl_x_p) ** 2 + (br_y_p - tl_y_p) ** 2) ** 0.5)
                min_dist = None
                for slot in range(min(4, len(doors_pbrs))):
                    d = doors_pbrs[slot]
                    if not d or len(d) < 2:
                        continue
                    if not bool(d[0]) or not bool(d[1]):  # exists & open
                        continue
                    dx_p, dy_p = door_positions[slot]
                    dist_p = ((px - dx_p) ** 2 + (py - dy_p) ** 2) ** 0.5 / room_diag
                    if min_dist is None or dist_p < min_dist:
                        min_dist = dist_p
                if min_dist is not None:
                    if st.prev_door_dist is not None:
                        # Delta reward: positive when closing distance.
                        delta = st.prev_door_dist - min_dist
                        add("door_pbrs", cfg.r_door_distance_shaping * delta)
                    st.prev_door_dist = min_dist
                # Extra idle penalty in cleared rooms (no reason to hesitate).
                if speed < cfg.idle_speed_threshold:
                    add("clear_idle_extra", cfg.r_clear_room_idle_extra)
            else:
                # Reset door-dist tracker on entering combat so re-clear
                # doesn't get a bogus huge delta.
                st.prev_door_dist = None

            if is_clear and speed >= cfg.seek_door_speed_threshold:
                doors = raw_obs.get("doors") or []
                # Door slot -> unit velocity direction we want.
                # 0=LEFT (-1,0), 1=UP (0,-1), 2=RIGHT (+1,0), 3=DOWN (0,+1)
                door_dirs = ((-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0))
                best_alignment = 0.0
                for slot in range(min(4, len(doors))):
                    d = doors[slot]
                    if not d or len(d) < 2:
                        continue
                    exists = bool(d[0])
                    is_open = bool(d[1])
                    if not exists or not is_open:
                        continue
                    dx_dir, dy_dir = door_dirs[slot]
                    # Normalized velocity dotted with door direction.
                    # speed is guaranteed > 0 above.
                    alignment = (vx * dx_dir + vy * dy_dir) / (speed + 1e-6)
                    if alignment > best_alignment:
                        best_alignment = alignment
                if best_alignment > 0:
                    # Scale by alignment so straight-line motion gets full reward
                    # and diagonal motion gets partial credit.
                    add("seek_door", cfg.r_seek_door_when_clear * best_alignment)

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
                        # Small reward just for shooting when enemies are visible.
                        # Now zero by default (was rewarding spam) but still
                        # tunable via config for special cases.
                        if cfg.r_shoot_when_enemy_visible != 0.0:
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
