"""Minimal 3-term reward for Isaac RL.

Post 2026-07-13 reset: the previous reward function had 51 terms accumulated
over 6 months. Un-debuggable — every training failure had 51 plausible root
causes, and every 'fix' was another shaping tweak that moved the bias.

Three non-zero terms only:

    r_kill  = +1     per NPC that dies
    r_death = -1     on player death
    r_step  = -0.001 per game tick (mild time pressure)

Reward shaping obscures whether learning is happening. If the agent can
learn 'kill things, don't die,' it will discover on its own that room
exploration and item pickups are instrumental. Adding shaping again should
require an ablation study proving it helps.

INTERFACE: this shaper is a drop-in replacement for the previous
RewardShaper — same `__call__`, `reset`, `finalize_episode`,
`episode_behavior_metrics`, and `state.dead` attributes. env.py still
works unchanged. All the removed reward terms are silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardConfig:
    """Three non-zero reward terms.

    Do not add more terms here without an experiment showing the addition
    helps. Reward-shaping accumulation caused the previous project reset.
    """
    r_kill: float = 1.0
    r_death: float = -1.0
    r_step: float = -0.001


@dataclass
class RewardState:
    """Minimal state persisted across a single episode.

    Only `dead` is used externally (by env.py's mod_restart path). Kept as
    a dataclass so `state.dead = True` reads naturally.
    """
    dead: bool = False


class RewardShaper:
    """Reads events from raw obs, returns scalar rewards.

    Callable signature matches env.py's usage:
        reward, terminated, breakdown = shaper(raw_obs, action=action)
    """

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()
        self.state = RewardState()
        # Episode-total breakdown accumulator (for TB per-episode reward split).
        self._episode_total: dict[str, float] = {}

    # ---- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Called at env.reset() start-of-episode."""
        self.state = RewardState()
        self._episode_total = {}

    def finalize_episode(self, reason: str) -> tuple[float, dict[str, float]]:
        """Called at env-side terminal-obs handoff. Returns (total, breakdown).

        Old shaper emitted outcome bonuses here (survival, depth, efficiency).
        This shaper emits nothing — 3-term reward means no end-of-episode bonus.
        Kept as a stub so env.py doesn't need to be touched.
        """
        return 0.0, {}

    def episode_behavior_metrics(self) -> dict[str, float]:
        """Called on info["behavior_metrics"]. TB scalars for the episode.

        Old shaper tracked ~20 metrics (kite time, aim alignment, ...).
        This shaper returns just the reward breakdown so TB still shows
        the per-episode kill / death / step splits.
        """
        return dict(self._episode_total)

    # ---- main call ---------------------------------------------------------

    def __call__(
        self,
        raw_obs: dict[str, Any] | None,
        action: Any = None,
    ) -> tuple[float, bool, dict[str, float]]:
        """Per-step reward computation.

        Args:
            raw_obs: the mod's most recent obs dict (may be None during crash
                     recovery).
            action: the action just taken (5-int MultiDiscrete). Unused in
                    this minimal shaper; kept for API compatibility.

        Returns:
            (reward, terminated, breakdown)
                reward: scalar reward for this step
                terminated: True if the player died this step
                breakdown: {"kill": x, "death": y, "step": z} — non-zero
                           terms only. Consumed by env.py for the TB
                           per-episode breakdown.
        """
        bd: dict[str, float] = {"step": self.cfg.r_step}

        events = (raw_obs or {}).get("events") or []
        terminated = False
        for ev in events:
            kind = ev.get("kind")
            if kind == "kill":
                bd["kill"] = bd.get("kill", 0.0) + self.cfg.r_kill
            elif kind == "death" and not self.state.dead:
                bd["death"] = bd.get("death", 0.0) + self.cfg.r_death
                self.state.dead = True
                terminated = True

        # Also detect death via hp_red==0 in the obs (belt+suspenders — the
        # 'death' event isn't always emitted before the mod_restart cycle).
        if not self.state.dead and raw_obs:
            player = raw_obs.get("player") or {}
            if player.get("is_dead") is True or (player.get("hp_red") == 0 and player.get("hp_soul", 0) == 0):
                bd["death"] = bd.get("death", 0.0) + self.cfg.r_death
                self.state.dead = True
                terminated = True

        total = sum(bd.values())
        # Fold into per-episode running total.
        for k, v in bd.items():
            self._episode_total[k] = self._episode_total.get(k, 0.0) + v

        return total, terminated, bd
