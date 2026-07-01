"""Reward shaper that turns raw Lua events + obs deltas into a scalar reward.

The Lua side already labels events (damage_to_npc, damage_to_player, pickup_*, new_room,
new_level, room_clear, death). We only need to weight them, apply anti-farming caps,
and detect terminal states.

Kept as a plain callable class so the env can hold one instance per Isaac process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardConfig:
    # Terminal
    r_beat_mom: float = 50.0
    r_floor_cleared: float = 5.0
    r_death: float = -10.0

    # Dense combat
    r_damage_dealt_scale: float = 0.1     # per (dmg / max_hp)
    r_kill: float = 0.5
    r_damage_taken_red: float = -1.0      # per red heart lost
    r_damage_taken_other: float = -0.5    # per soul/black heart lost

    # Room / exploration
    r_room_clear: float = 1.0
    r_new_room: float = 0.05
    r_boss_room_first_entry: float = 0.5

    # Pickups
    r_pickup_heart: float = 0.2
    r_pickup_coin: float = 0.05
    r_pickup_key: float = 0.1
    r_pickup_bomb: float = 0.1
    r_pickup_collectible: float = 1.0

    # Cost per step (~15 Hz)
    r_step: float = -0.001

    # Damage-per-room cap so infinite spawners can't inflate reward.
    max_damage_reward_per_room: float = 3.0


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


class RewardShaper:
    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()
        self.state = RewardState()

    def reset(self) -> None:
        self.state = RewardState()

    def __call__(self, raw_obs: dict[str, Any]) -> tuple[float, bool, dict[str, float]]:
        """Return (reward, terminated, breakdown)."""
        cfg = self.cfg
        st = self.state
        reward = cfg.r_step
        breakdown: dict[str, float] = {}

        def add(name: str, x: float) -> None:
            nonlocal reward
            reward += x
            breakdown[name] = breakdown.get(name, 0.0) + x

        # Initialize baseline HP on first tick after a reset.
        player = raw_obs.get("player") or {}
        cur_red = float(player.get("hp_red", 0) or 0)
        cur_soul = float(player.get("hp_soul", 0) or 0)
        cur_black = float(player.get("hp_black", 0) or 0)

        # HP-delta damage-taken reward (in case events miss e.g. spike hits).
        if st.prev_hp_red or st.prev_hp_soul or st.prev_hp_black:
            d_red = st.prev_hp_red - cur_red
            d_other = (st.prev_hp_soul - cur_soul) + (st.prev_hp_black - cur_black)
            if d_red > 0:
                add("hp_delta_red", d_red * cfg.r_damage_taken_red)
            if d_other > 0:
                add("hp_delta_other", d_other * cfg.r_damage_taken_other)
        st.prev_hp_red = cur_red
        st.prev_hp_soul = cur_soul
        st.prev_hp_black = cur_black

        # Events posted by Lua this tick.
        terminated = False
        for evt in raw_obs.get("events") or []:
            kind = evt.get("kind")

            if kind == "damage_to_npc":
                dmg = float(evt.get("dmg", 0) or 0)
                max_hp = float(evt.get("npc_max_hp", 1) or 1) or 1.0
                gain = cfg.r_damage_dealt_scale * min(dmg, max_hp) / max_hp
                # Cap total damage-derived reward per room.
                room_left = max(0.0, cfg.max_damage_reward_per_room - st.damage_reward_this_room)
                gain = min(gain, room_left)
                st.damage_reward_this_room += gain
                add("damage_dealt", gain)
                if evt.get("killed"):
                    add("kill", cfg.r_kill)
                    if evt.get("is_boss"):
                        # Beating a boss = +kill on the boss entity. Handled per stage below.
                        pass

            elif kind == "damage_to_player":
                # HP-delta handles this. Skip to avoid double-counting.
                pass

            elif kind == "pickup_collectible":
                add("pickup_collectible", cfg.r_pickup_collectible)

            elif kind == "new_room":
                if evt.get("is_new"):
                    add("new_room", cfg.r_new_room)
                st.damage_reward_this_room = 0.0
                if evt.get("room_type") == ROOM_TYPE_BOSS:
                    sgi = evt.get("safe_grid_index")
                    if sgi is not None and sgi not in st.visited_boss_rooms:
                        st.visited_boss_rooms.add(sgi)
                        add("boss_room_first_entry", cfg.r_boss_room_first_entry)

            elif kind == "room_clear":
                add("room_clear", cfg.r_room_clear)

            elif kind == "new_level":
                stage = int(evt.get("stage", 0) or 0)
                if stage > st.last_stage and st.last_stage > 0:
                    add("floor_cleared", cfg.r_floor_cleared)
                # Special-case: descending past stage 6 with Mom dead counts as beating Mom
                # (the trapdoor after Mom leads to stage 7 Womb). Reward on new-level.
                if st.last_stage == 6 and stage == 7:
                    add("beat_mom", cfg.r_beat_mom)
                    terminated = True
                st.last_stage = stage

            elif kind == "death":
                if not st.dead:
                    st.dead = True
                    add("death", cfg.r_death)
                    terminated = True

        return float(reward), bool(terminated), breakdown
