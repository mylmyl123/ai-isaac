"""Tests for the idle penalty ("time since last kill" pressure, 2026-07-15).

This term replaces r_hit as the anti-park-and-spray mechanism. The properties
that MUST hold (any of these breaking reintroduces a known failure):
  * counter resets to 0 on a kill; the killing step itself is NOT penalized
  * penalty fires only AFTER idle_grace ticks without a kill
  * the terminal (death) step is not additionally idle-penalized
  * MAGNITUDE: a realistic kill cycle nets positive (killing beats idling),
    a never-kill episode is strongly negative (park-and-spray dominated),
    and dying-immediately does NOT beat a killing policy (no suicide trap)
  * r_idle=0.0 fully disables it (recovers prior behavior)

Run:
    PYTHONPATH=python pytest tests/test_idle_penalty.py -q
"""
from __future__ import annotations

import pytest

from isaac_rl.reward import RewardConfig, RewardShaper


def _step(sh, killed=False, dead=False, enemy=True):
    """Drive one shaper step. killed -> a kill event; dead -> a death event.
    enemy -> include a live enemy in the obs (the idle counter is frozen when
    no enemy is present, so tests that expect the penalty to advance must have
    one)."""
    events = []
    if killed:
        events.append({"kind": "damage_to_npc", "killed": True})
    if dead:
        events.append({"kind": "death"})
    raw = {"player": {"hp_red": 0 if dead else 3, "hp_soul": 0}, "events": events}
    if enemy:
        raw["enemies"] = {"feats": [[0, 0, 0.3, 0.2, 0, 0, 1] + [0] * 9], "mask": [1]}
    else:
        raw["enemies"] = {"feats": [], "mask": []}
    return sh(raw)


# --------------------------------------------------------------------------
# Behavior
# --------------------------------------------------------------------------

def test_no_penalty_within_grace():
    sh = RewardShaper(RewardConfig(r_idle=-0.005, idle_grace=12, r_step=-0.001))
    sh.reset()
    for _ in range(12):                       # ticks 1..12, all <= grace
        _, _, bd = _step(sh)
        assert "idle" not in bd
    # tick 13 is the first that exceeds grace
    _, _, bd = _step(sh)
    assert bd["idle"] == pytest.approx(-0.005)


def test_penalty_persists_after_grace():
    sh = RewardShaper(RewardConfig(r_idle=-0.005, idle_grace=12))
    sh.reset()
    for _ in range(12):
        _step(sh)
    for _ in range(5):
        _, _, bd = _step(sh)
        assert bd["idle"] == pytest.approx(-0.005)


def test_kill_resets_counter_and_killing_step_not_penalized():
    sh = RewardShaper(RewardConfig(r_idle=-0.005, idle_grace=12, r_kill=1.0))
    sh.reset()
    for _ in range(20):                       # go past grace so penalty is active
        _step(sh)
    # A killing step must reset the counter and NOT carry an idle penalty.
    _, _, bd = _step(sh, killed=True)
    assert "idle" not in bd
    assert bd["kill"] == pytest.approx(1.0)
    assert sh.state.ticks_since_kill == 0
    # Immediately after a kill we're back inside the grace window.
    for _ in range(12):
        _, _, bd = _step(sh)
        assert "idle" not in bd
    _, _, bd = _step(sh)                       # tick 13 since the kill
    assert bd["idle"] == pytest.approx(-0.005)


def test_terminal_step_not_idle_penalized():
    sh = RewardShaper(RewardConfig(r_idle=-0.005, idle_grace=12))
    sh.reset()
    for _ in range(20):
        _step(sh)
    _, term, bd = _step(sh, dead=True)
    assert term is True
    assert bd["death"] == pytest.approx(-1.0)
    assert "idle" not in bd                    # dead agent isn't also idle-penalized


def test_counter_frozen_while_no_enemy_present():
    """The idle counter must NOT advance while no enemy is on screen (the forced
    respawn gap) — otherwise a prompt killer is punished for a wait it can't
    act on."""
    sh = RewardShaper(RewardConfig(r_idle=-0.005, idle_grace=12))
    sh.reset()
    # 50 ticks with NO enemy: counter frozen, never any penalty.
    for _ in range(50):
        _, _, bd = _step(sh, enemy=False)
        assert "idle" not in bd
    assert sh.state.ticks_since_kill == 0
    # Enemy appears: counter now advances and penalty kicks in after grace.
    for _ in range(12):
        _, _, bd = _step(sh, enemy=True)
        assert "idle" not in bd
    _, _, bd = _step(sh, enemy=True)
    assert bd["idle"] == pytest.approx(-0.005)


def test_r_idle_zero_disables():
    sh = RewardShaper(RewardConfig(r_idle=0.0, idle_grace=12))
    sh.reset()
    for _ in range(100):
        _, _, bd = _step(sh)
        assert "idle" not in bd


def test_default_config_has_idle_active():
    cfg = RewardConfig()
    # Phase-3 defaults (2026-07-15): idle softened, kill raised.
    assert cfg.r_idle == -0.002 and cfg.idle_grace == 20
    assert cfg.r_kill == 3.0
    # And the superseded shaping terms are OFF by default.
    assert cfg.r_hit == 0.0 and cfg.pbrs_coef == 0.0


# --------------------------------------------------------------------------
# Magnitude / incentive properties (the reason the term exists)
# --------------------------------------------------------------------------

def _episode_return(sh, kill_ticks, total_ticks):
    """Simulate an episode where kills happen on the given tick indices.
    Returns the summed reward (no discounting — we check raw incentive sign)."""
    sh.reset()
    kills = set(kill_ticks)
    total = 0.0
    for t in range(total_ticks):
        r, _, _ = _step(sh, killed=(t in kills))
        total += r
    return total


def test_killing_beats_park_and_spray():
    cfg = RewardConfig(r_idle=-0.005, idle_grace=12, r_step=-0.001, r_kill=1.0)
    # Park-and-spray: never kill for the full 1800-tick episode.
    spray = _episode_return(RewardShaper(cfg), kill_ticks=[], total_ticks=1800)
    # Killer: a kill every ~90 ticks (20 kills over 1800 ticks).
    killer = _episode_return(RewardShaper(cfg), kill_ticks=range(90, 1800, 90), total_ticks=1800)
    assert spray < 0, f"park-and-spray should bleed reward, got {spray}"
    assert killer > spray, f"killer ({killer}) must beat spray ({spray})"
    assert killer > 0, f"a steady killer should net positive, got {killer}"


def test_no_suicide_trap():
    """Dying immediately (one -1) must NOT beat a policy that keeps killing.
    The linear-ramp version failed this; the flat-after-grace version must pass."""
    cfg = RewardConfig(r_idle=-0.005, idle_grace=12, r_step=-0.001, r_kill=1.0, r_death=-1.0)
    # Suicide: die on the first tick.
    sh = RewardShaper(cfg); sh.reset()
    r, term, _ = _step(sh, dead=True)
    suicide = r
    # Killer over a full episode.
    killer = _episode_return(RewardShaper(cfg), kill_ticks=range(90, 1800, 90), total_ticks=1800)
    assert killer > suicide, f"killing ({killer}) must beat suicide ({suicide})"


def test_faster_killing_scores_higher():
    """Killing more frequently should net more (the term rewards promptness)."""
    cfg = RewardConfig(r_idle=-0.005, idle_grace=12, r_step=-0.001, r_kill=1.0)
    slow = _episode_return(RewardShaper(cfg), kill_ticks=range(120, 1200, 120), total_ticks=1200)
    fast = _episode_return(RewardShaper(cfg), kill_ticks=range(60, 1200, 60), total_ticks=1200)
    assert fast > slow, f"faster killing ({fast}) should beat slower ({slow})"


def test_documented_cycle_magnitude():
    """Pin the exact 90-tick-cycle net stated in reward.py's docstring (+0.525)
    so the documented magnitude can't silently drift from the code again."""
    cfg = RewardConfig(r_idle=-0.005, idle_grace=12, r_step=-0.001, r_kill=1.0)
    net = _episode_return(RewardShaper(cfg), kill_ticks=[89], total_ticks=90)
    assert net == pytest.approx(0.525, abs=1e-6)
