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

    PBRS (2026-07-14): optional potential-based shaping for the sparse-reward
    cold start. Ng et al. 1999 form F = gamma*Phi(s') - Phi(s); provably
    optimal-policy-invariant, so it only densifies the signal, never changes
    the optimum. DISABLED by default (pbrs_coef=0.0) — the 3-term reward is
    still the baseline. `gamma` MUST equal the trainer's gamma (PPOConfig /
    curriculum.yaml) for the invariance guarantee to hold.
    """
    r_kill: float = 1.0
    r_death: float = -1.0
    r_step: float = -0.001
    # ---- PBRS (potential-based reward shaping) ----
    pbrs_coef: float = 0.0            # 0.0 = off (pure 3-term baseline)
    gamma: float = 0.995              # must match PPOConfig.gamma


@dataclass
class RewardState:
    """Minimal state persisted across a single episode.

    `dead` is used externally (by env.py's mod_restart path). `last_phi` holds
    Phi(s) of the previous step for the PBRS telescoping term; None before the
    first potential is computed.
    """
    dead: bool = False
    last_phi: float | None = None


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

    # ---- PBRS potential ----------------------------------------------------

    def _potential(self, raw_obs: dict[str, Any] | None) -> float:
        """Phi(s) for potential-based shaping.

        Phi(s) = -(normalized distance from player to nearest live enemy),
        in [-~1.4, 0]: 0 when on top of the enemy (best), more negative when
        far. Maximizing discounted sum of F = gamma*Phi(s')-Phi(s) rewards
        CLOSING distance to the enemy, which densifies the aim/approach signal
        without a kill. Returns 0.0 when disabled or no enemy is visible (so
        F contributes nothing on enemy-less frames).

        Uses the enemy relative-offset features the mod already emits
        (obs.lua build_enemies: feats[2],feats[3] = (ex-px)/480, (ey-py)/270),
        so no new obs field is needed. Reads the raw JSON, not the encoded obs.
        """
        if self.cfg.pbrs_coef == 0.0 or not raw_obs:
            return 0.0
        enemies = raw_obs.get("enemies") or {}
        feats = enemies.get("feats") or []
        mask = enemies.get("mask") or []
        best = None
        for i, row in enumerate(feats):
            if not row or (i < len(mask) and not mask[i]):
                continue
            # feats[2]=rel_x/480, feats[3]=rel_y/270 (normalized offsets).
            try:
                rx = float(row[2]); ry = float(row[3])
            except (IndexError, TypeError, ValueError):
                continue
            d = (rx * rx + ry * ry) ** 0.5
            if best is None or d < best:
                best = d
        if best is None:
            return 0.0            # no visible enemy -> flat potential
        return -best

    def finalize_episode(self, reason: str) -> tuple[float, dict[str, float]]:
        """Called at env-side terminal-obs handoff. Returns (total, breakdown).

        PBRS terminal correction (CRITICAL for invariance): Ng-1999 requires
        Phi(terminal)=0, so the final transition contributes
        F = gamma*Phi(terminal) - Phi(s_last) = -Phi(s_last). The mod_restart
        and isaac_crash terminal paths in env.py bypass __call__ entirely and
        only call THIS method, so the terminal correction MUST live here or the
        uncancelled -Phi(s_last) residual silently breaks policy-invariance.
        Fires once per episode; guarded by last_phi being set + not already
        consumed.
        """
        bd: dict[str, float] = {}
        if self.cfg.pbrs_coef != 0.0 and self.state.last_phi is not None:
            # F_terminal = gamma*0 - Phi(s_last), scaled by pbrs_coef.
            f_term = self.cfg.pbrs_coef * (-self.state.last_phi)
            bd["pbrs"] = f_term
            self.state.last_phi = None   # consumed; don't double-apply
            for k, v in bd.items():
                self._episode_total[k] = self._episode_total.get(k, 0.0) + v
            return f_term, bd
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
            # 2026-07-14: Mod emits kills as {kind: 'damage_to_npc', killed: true}
            # NOT as {kind: 'kill'}. Recognize both for forward-compat.
            if kind == "kill" or (kind == "damage_to_npc" and ev.get("killed")):
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

        # PBRS shaping term F = gamma*Phi(s') - Phi(s), added per non-terminal
        # step (Ng et al. 1999). No-op when pbrs_coef=0. On a terminal step we
        # do NOT add the running term here — finalize_episode() applies the
        # -Phi(s_last) correction with Phi(terminal)=0 to keep the telescoping
        # sum (and thus policy-invariance) exact. last_phi carries Phi(s) across
        # steps; on the first step (last_phi None) there is no previous state so
        # the term is skipped and we just seed last_phi.
        if self.cfg.pbrs_coef != 0.0 and not terminated:
            phi_now = self._potential(raw_obs)
            if self.state.last_phi is not None:
                f = self.cfg.pbrs_coef * (self.cfg.gamma * phi_now - self.state.last_phi)
                bd["pbrs"] = bd.get("pbrs", 0.0) + f
            self.state.last_phi = phi_now

        total = sum(bd.values())
        # Fold into per-episode running total.
        for k, v in bd.items():
            self._episode_total[k] = self._episode_total.get(k, 0.0) + v

        return total, terminated, bd
