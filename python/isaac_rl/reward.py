"""Reward for Isaac RL.

Post 2026-07-13 reset: the previous reward had 51 terms accumulated over 6
months — un-debuggable. Rebuilt minimal, then iterated against observed
training exploits (each documented below and gated behind a flag).

Active terms (2026-07-15):

    r_kill  = +1      per NPC that dies
    r_death = -1      on player death
    r_step  = -0.001  per game tick (mild time pressure)
    r_idle  = -0.005  per tick once >idle_grace ticks since the last kill

The idle penalty ("time since last kill") is the fix for the park-and-spray
exploit found in the 2026-07-15 overnight sweep: rewarding hits directly
(r_hit) let the agent spray one fixed direction, graze a stationary enemy, and
survive forever without killing. r_idle instead makes NOT killing hurt more the
longer it goes on (flat penalty after a short grace window, frozen while no
enemy is present), so aiming-and-killing strictly dominates loitering. r_hit
and PBRS are kept but DISABLED by default (=0) so prior runs stay reproducible
as ablations.

INTERFACE: drop-in for the previous RewardShaper — same `__call__`, `reset`,
`finalize_episode`, `episode_behavior_metrics`, and `state.dead`. env.py
unchanged. Unknown/removed reward terms are silently ignored.
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
    # ---- Idle penalty (2026-07-15): "time since last kill" pressure ----
    # A flat penalty applied every step once it has been more than `idle_grace`
    # ticks since the last kill (counter resets to 0 on every kill) AND an enemy
    # is actually present to shoot. This is the fix for the park-and-spray
    # exploit: a policy that survives without killing bleeds reward the longer
    # it goes between kills, so the only way to stop the bleeding is to kill.
    # FLAT-after-grace (not a linear ramp) on purpose — a ramp is quadratic in
    # the cumulative sum and makes "die immediately" beat "keep trying" (the
    # suicide trap the original 51-term reward suffered).
    #
    # idle_grace=12: the mod respawns the next enemy ~15 game-ticks after the
    # room empties (main.lua poll), so max kill cadence is ~1 per 15-25 ticks. A
    # grace ABOVE that (the initial 30) let a lazy "kill every ~30 ticks, loiter
    # between" policy pay ZERO penalty — the shallow-gradient trap that sank the
    # prior rewards. 12 sits just under the respawn floor so loitering is
    # penalized while a prompt killer at the environment's max cadence is not.
    # The counter is FROZEN while no enemy is visible (see __call__) so the agent
    # is never penalized for the forced empty-room respawn gap it can't act on.
    # Verified at r_idle=-0.005, grace=12: a ~90-tick kill cycle nets +0.525,
    # never-killing a full 1800-tick episode bleeds strongly negative, and
    # killing beats both idling and suicide. r_idle=0.0 disables it.
    r_idle: float = -0.005
    idle_grace: int = 12
    # ---- Dense per-hit reward (Phase-2c cold-start fix) ----
    # DISABLED by default (r_hit=0.0). Rewarded every connecting tear, which the
    # 2026-07-15 overnight sweep showed causes a "park and spray one fixed
    # direction, graze the enemy, never kill" local optimum — superseded by the
    # idle penalty above. Kept (off) only so prior runs remain reproducible for
    # the paper's ablation. Do NOT enable alongside r_idle without an ablation.
    r_hit: float = 0.0
    # ---- PBRS (potential-based reward shaping) ----
    pbrs_coef: float = 0.0            # 0.0 = off (pure baseline)
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
    # Ticks since the last kill (or since episode start). Reset to 0 on every
    # kill; drives the idle penalty once it exceeds idle_grace.
    ticks_since_kill: int = 0


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

    def _enemy_present(self, raw_obs: dict[str, Any] | None) -> bool:
        """True if at least one live enemy is visible in the obs.

        Used to FREEZE the idle counter during the forced empty-room respawn
        gap (the mod respawns the next enemy a few ticks after the room clears),
        so a prompt killer isn't idle-penalized for a wait it cannot act on.
        Prefers the enemy mask; falls back to the feats list, then to a
        conservative True (no obs info -> keep counting so we never silently
        disable the penalty). Reads the raw JSON, matching _potential.
        """
        if not raw_obs:
            return True
        enemies = raw_obs.get("enemies")
        if not isinstance(enemies, dict):
            return True
        mask = enemies.get("mask")
        if isinstance(mask, list):
            return any(bool(m) for m in mask)
        feats = enemies.get("feats")
        if isinstance(feats, list):
            return any(bool(row) for row in feats)
        return True

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
        killed_this_step = False
        for ev in events:
            kind = ev.get("kind")
            # 2026-07-14: Mod emits kills as {kind: 'damage_to_npc', killed: true}
            # NOT as {kind: 'kill'}. Recognize both for forward-compat.
            if kind == "kill" or (kind == "damage_to_npc" and ev.get("killed")):
                bd["kill"] = bd.get("kill", 0.0) + self.cfg.r_kill
                killed_this_step = True
            elif kind == "damage_to_npc" and self.cfg.r_hit != 0.0:
                # Dense per-hit reward on a NON-lethal connect. A landed tear on
                # a (stationary) enemy means the shoot head aimed correctly.
                # Scale by damage fraction so total hit reward per enemy ~= r_hit
                # (fractions across the ~3 hits to kill sum to ~1), keeping
                # r_kill strictly dominant and preventing multi-hit farming.
                max_hp = float(ev.get("npc_max_hp", 0) or 0)
                dmg = float(ev.get("dmg", 0) or 0)
                frac = (dmg / max_hp) if max_hp > 0 else 0.0
                bd["hit"] = bd.get("hit", 0.0) + self.cfg.r_hit * min(1.0, frac)
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

        # Idle penalty: "time since last kill" pressure (2026-07-15). Reset the
        # counter on a kill (so the killing step itself is never penalized).
        # Otherwise, only advance/penalize while an enemy is actually PRESENT to
        # shoot — freezing the counter during the forced empty-room respawn gap
        # so a prompt killer isn't punished for a wait the environment imposes
        # (not its fault it can't kill a nonexistent target). Once the counter
        # exceeds idle_grace, apply the flat r_idle penalty every step. Skipped
        # on the terminal (death) step. No-op when r_idle == 0.0.
        if killed_this_step:
            self.state.ticks_since_kill = 0
        elif not terminated and self._enemy_present(raw_obs):
            self.state.ticks_since_kill += 1
            if self.cfg.r_idle != 0.0 and self.state.ticks_since_kill > self.cfg.idle_grace:
                bd["idle"] = bd.get("idle", 0.0) + self.cfg.r_idle

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
