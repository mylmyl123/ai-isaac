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
    r_new_room: float = 3.0               # was 0.2 — dramatically boosted (2026-07-03)
    # RATIONALE: 0.2 wasn't enough to overcome the risk aversion trap.
    # Bots learned "cleared room = safe -4.0 discounted idle return; moving
    # to door = might enter combat = might die (-3.0)" and picked idle.
    # At r_new_room=3.0, crossing is strongly incentivized even if the next
    # room has enemies. Combined with r_door_distance_shaping (PBRS),
    # this gives dense signal both en-route to and at the moment of crossing.
    # Backtrack penalty (2026-07-03): if bot re-enters a room it's been to
    # before, apply this penalty. Breaks the "walk-up-and-down between two
    # cleared rooms" oscillation loop that arises when door_pbrs rewards
    # any door approach without regard to novelty.
    r_backtrack: float = -0.5
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

    # ---- NEW (2026-07-04): Idle death termination -----------------------
    # User request after 605K-step analysis showed bot idled for 400K steps
    # in a bad local optimum. Instead of just penalizing idle per-tick, we
    # now TERMINATE the episode if the bot has been idle for too long,
    # applying a huge negative reward. This:
    #   1. Forces PPO to actually update policy (episode boundary = clear signal)
    #   2. Provides a discrete cliff cost that dominates any incremental
    #      idle-reward exploit
    #   3. Resets the bot to a fresh spawn, giving exploration another shot
    # Only fires when bot is IDLE (velocity ~ 0) AND not in combat (no enemies
    # visible). In combat, staying in place can be tactical (dodging in a
    # tight corner) so we don't terminate.
    r_idle_death: float = -20.0            # applied when idle_death_ticks reached
    idle_death_ticks: int = 300            # 300 ticks = ~20 seconds at 15Hz
    idle_death_require_no_enemies: bool = True   # don't terminate if enemies visible

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


# All reward-breakdown keys the RewardShaper can emit. Kept in sync with the
# add("...") calls in RewardShaper.__call__ below. Logging (TB) pre-populates
# per-episode aggregates with zeros for every key so "never fired" reward
# components appear as flat-zero traces instead of being invisible (which
# hid the exploration crisis on the 2026-07-06 run: kill/damage_dealt/
# new_room/room_clear had literally never fired in 711 episodes but there
# was no TB scalar to see it).
REWARD_BREAKDOWN_KEYS: tuple[str, ...] = (
    # per-tick / dense
    "step", "hp_delta_red", "hp_delta_other", "full_hp_tick",
    "idle_penalty", "stationary_penalty", "at_kite_dist", "aim_at_enemy",
    "shoot_when_enemy", "seek_door", "door_pbrs", "pbrs_approach",
    "clear_idle_extra",
    # events
    "damage_dealt", "kill",
    "pickup_heart", "pickup_coin", "pickup_key", "pickup_bomb", "pickup_collectible",
    "new_room", "backtrack", "boss_room_first_entry",
    "room_clear", "room_clear_speed", "room_clear_no_damage",
    "floor_cleared", "beat_mom",
    # terminals
    "death", "idle_death", "crash_penalty",
)


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
    consecutive_idle_ticks: int = 0        # 2026-07-04: for idle-death termination


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
        # Initialize early so the HP-based death detection below (inserted
        # 2026-07-08) can read `terminated`. The original declaration at
        # the start of the events loop is now redundant — kept as a no-op.
        terminated: bool = False

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
        # Snapshot the previous-tick HP totals BEFORE we overwrite them, so
        # the HP-based death detection below can check the "was alive last
        # tick" condition (needed to distinguish real death from the
        # first-obs state where prev is 0).
        prev_alive = st.prev_hp_red > 0 or st.prev_hp_soul > 0 or st.prev_hp_black > 0
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

        # ---- HP-based death detection (Python-side, mod-independent) ------
        # The mod is supposed to send a {kind: "death"} event on player
        # death, but its delivery is unreliable (see 2026-07-08 postmortem:
        # 100% of episodes ended via mod_socket_error, 0 via shaper
        # termination, despite the fact that the mod's fast-path AND render
        # fallback both call handle_player_death). Rather than depend on
        # the mod delivering the terminal frame, we terminate here whenever
        # the player's total HP reaches 0 after having been >0 in a prior
        # tick. Uses the same r_death reward as the mod-delivered event,
        # so downstream training is unaffected.
        #
        # Uses `prev_alive` snapshotted above (before the prev fields were
        # overwritten with this tick's HP). Also gates on max_hp > 0 to
        # exclude Lost-style 0-max characters. In practice we run as Isaac
        # (max_hp = 6), so this is belt-and-suspenders.
        if not terminated and not st.dead:
            total_cur = cur_red + cur_soul + cur_black
            if max_hp > 0 and total_cur <= 0 and prev_alive:
                st.dead = True
                add("death", cfg.r_death)
                terminated = True

        # ---- Events (kills, pickups, room/level, death) -------------------
        # `terminated` was hoisted to the top of __call__ so the HP-based
        # death check can read it; keeping this line as a no-op preserves
        # blame-friendly diffs and future-proofs against someone deleting
        # the hoist.
        terminated = terminated or False
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
                else:
                    # Backtrack: re-entering a room we've been to before.
                    # Break oscillation loops between cleared rooms.
                    add("backtrack", cfg.r_backtrack)
                # Reset per-room shaping state
                st.damage_reward_this_room = 0.0
                st.ticks_since_room_start = 0
                st.damage_this_room_red = 0.0
                st.damage_this_room_other = 0.0
                # CRITICAL: reset door PBRS state so cross-room delta doesn't
                # emit a bogus reward/penalty. Without this, the delta between
                # "nearly at LEFT door of prev room" (dist ≈ 0) and "far from
                # any door in new room" (dist ≈ 0.5) generates a -0.15 penalty,
                # or vice-versa. This drove room-oscillation loops.
                st.prev_door_dist = None
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
                # Idle-death: track consecutive idle ticks. If bot idles too
                # long WITHOUT enemies present, terminate the episode with a
                # huge negative reward. This forces PPO out of "stand still"
                # local optima that per-tick penalties alone can't escape.
                enemies_present = False
                enemies_obs = raw_obs.get("enemies") or {}
                mask = enemies_obs.get("mask") or []
                if any(bool(m) for m in mask):
                    enemies_present = True
                if cfg.idle_death_require_no_enemies and enemies_present:
                    # In combat: don't count toward idle-death (dodging is OK).
                    st.consecutive_idle_ticks = 0
                else:
                    st.consecutive_idle_ticks += 1
                    if st.consecutive_idle_ticks >= cfg.idle_death_ticks:
                        add("idle_death", cfg.r_idle_death)
                        terminated = True
                        st.consecutive_idle_ticks = 0   # reset for next episode
            else:
                st.consecutive_idle_ticks = 0

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
