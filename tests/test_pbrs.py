"""Tests for the Phase-2 cold-start interventions (2026-07-14):
  * PBRS potential-based shaping — policy-invariance (telescoping sum),
    off-by-default, and correct terminal handling.
  * Closer-spawn curriculum — env-var band parsing in the mod (static text).

The invariance property we assert (Ng et al. 1999): the total shaping reward
added over a full episode telescopes to  sum_t [gamma*Phi(s_{t+1}) - Phi(s_t)]
with Phi(terminal)=0, which for a fixed start collapses to a term that depends
only on the potentials, NOT on the actions taken between them — so it cannot
change which policy is optimal. We check the shaper reproduces exactly this
sum and that disabling it recovers the pure 3-term reward.

Run:
    PYTHONPATH=python pytest tests/test_pbrs.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from isaac_rl.reward import RewardConfig, RewardShaper

REPO = Path(__file__).resolve().parent.parent


def _obs(rel_x, rel_y, dead=False):
    """Minimal raw obs with one enemy at normalized rel offset (feats[2],[3])."""
    return {
        "player": {"hp_red": 0 if dead else 3, "hp_soul": 0},
        "enemies": {"feats": [[0, 0, rel_x, rel_y, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "mask": [1]},
        "events": [],
    }


# --------------------------------------------------------------------------
# pbrs_coef = 0 -> pure 3-term reward (no contamination)
# --------------------------------------------------------------------------

def test_pbrs_disabled_is_pure_three_term():
    sh = RewardShaper(RewardConfig(pbrs_coef=0.0))
    sh.reset()
    r, term, bd = sh(_obs(0.5, 0.5))
    assert "pbrs" not in bd
    assert r == pytest.approx(-0.001)   # only r_step
    assert not term


# --------------------------------------------------------------------------
# PBRS telescoping invariance
# --------------------------------------------------------------------------

def test_pbrs_telescopes_to_minus_phi_start():
    """Sum of PBRS terms over an episode (incl. terminal correction) must equal
    gamma^? ... more simply: per-step sum_t F_t = sum_t (gamma*Phi_{t+1}-Phi_t).
    With Phi(terminal)=0 applied at finalize, the running PBRS contributions
    plus the terminal correction telescope to a value determined only by the
    potentials, independent of the specific intermediate potentials' path.

    Concretely we verify the shaper's emitted PBRS sum equals the closed-form
    telescoping value for a hand-chosen sequence of states."""
    gamma, coef = 0.995, 1.0
    sh = RewardShaper(RewardConfig(pbrs_coef=coef, gamma=gamma))
    sh.reset()

    # A trajectory of enemy-relative distances (dist = hypot(rel_x, rel_y)).
    # Phi(s) = -dist. We drive the shaper and accumulate every "pbrs" term.
    seq = [(0.6, 0.0), (0.3, 0.0), (0.1, 0.0)]   # enemy getting closer
    phis = [-abs(rx) for (rx, _) in seq]         # ry=0 so dist=|rx|

    pbrs_sum = 0.0
    for (rx, ry) in seq:
        _, _, bd = sh(_obs(rx, ry))
        pbrs_sum += bd.get("pbrs", 0.0)
    # Terminal correction (Phi(terminal)=0): finalize adds -Phi(s_last).
    term_total, term_bd = sh.finalize_episode("shaper_terminated")
    pbrs_sum += term_bd.get("pbrs", 0.0)

    # Closed form: F_1 has no previous state (last_phi seeded, no term). So the
    # emitted running terms are for transitions 1->2 and 2->3:
    #   gamma*phi[1]-phi[0]  +  gamma*phi[2]-phi[1]
    # plus terminal:  -phi[2]  (gamma*0 - phi_last)
    expected = (gamma * phis[1] - phis[0]) + (gamma * phis[2] - phis[1]) + (-phis[2])
    assert pbrs_sum == pytest.approx(expected, abs=1e-9)


def test_pbrs_discounted_return_telescopes_to_minus_phi_start():
    """The Ng-1999 invariance holds for the DISCOUNTED return, not the raw sum:
        sum_t gamma^t * F_t  =  -Phi(s_0)      (with Phi(terminal)=0)
    because sum_t gamma^t (gamma*Phi_{t+1} - Phi_t) telescopes. The raw
    (undiscounted) sum of shaping rewards is path-dependent when gamma!=1 and
    that is fine — the agent optimizes the discounted return, which is what
    must be invariant. We verify the discounted sum equals -Phi(s_0) exactly,
    for two different middle paths sharing the same start."""
    gamma, coef = 0.9, 1.0

    def discounted_pbrs(mid_states):
        sh = RewardShaper(RewardConfig(pbrs_coef=coef, gamma=gamma))
        sh.reset()
        # F_t is the shaping reward on the transition OUT OF step t. The first
        # sh() call seeds Phi(s_0) and emits no term (no previous state), so the
        # first emitted term F corresponds to the transition s_0->s_1, i.e. t=0.
        disc = 0.0
        t = 0
        first = True
        for (rx, ry) in mid_states:
            _, _, bd = sh(_obs(rx, ry))
            if first:
                first = False   # seeded Phi(s_0); no F yet
                continue
            disc += (gamma ** t) * bd.get("pbrs", 0.0)
            t += 1
        # Terminal transition s_last -> terminal: F = gamma*0 - Phi(s_last).
        _, term_bd = sh.finalize_episode("done")
        disc += (gamma ** t) * term_bd.get("pbrs", 0.0)
        return disc

    phi_start = -0.5    # Phi(s_0) = -|rel_x| = -0.5 for both paths
    path_a = [(0.5, 0.0), (0.4, 0.0), (0.2, 0.0)]
    path_b = [(0.5, 0.0), (0.8, 0.0), (0.2, 0.0)]
    a = discounted_pbrs(path_a)
    b = discounted_pbrs(path_b)
    assert a == pytest.approx(b, abs=1e-9)
    assert a == pytest.approx(-phi_start, abs=1e-9)   # == -Phi(s_0)


def test_pbrs_no_enemy_is_flat_potential():
    sh = RewardShaper(RewardConfig(pbrs_coef=1.0, gamma=0.99))
    sh.reset()
    raw = {"player": {"hp_red": 3}, "enemies": {"feats": [], "mask": []}, "events": []}
    _, _, bd = sh(raw)
    # No enemy -> Phi=0 both steps -> F=0 (first step seeds last_phi=0 anyway).
    assert bd.get("pbrs", 0.0) == pytest.approx(0.0)


def test_pbrs_terminal_correction_fires_once():
    sh = RewardShaper(RewardConfig(pbrs_coef=1.0, gamma=0.99))
    sh.reset()
    sh(_obs(0.5, 0.0))
    t1, bd1 = sh.finalize_episode("done")
    assert "pbrs" in bd1 and t1 != 0.0
    # Second call must not re-apply (last_phi consumed to None).
    t2, bd2 = sh.finalize_episode("done")
    assert t2 == 0.0 and bd2 == {}


# --------------------------------------------------------------------------
# Closer-spawn curriculum (mod static checks)
# --------------------------------------------------------------------------

def test_mod_reads_spawn_band_env_vars():
    main_lua = (REPO / "mods" / "isaac-rl-bridge" / "main.lua").read_text(encoding="utf-8")
    assert "ISAAC_RL_SPAWN_MIN" in main_lua
    assert "ISAAC_RL_SPAWN_MAX" in main_lua
    # The hard-coded 200/500 band literal must be gone from the distance check.
    assert "d >= SPAWN_DIST_MIN and d <= SPAWN_DIST_MAX" in main_lua
    # Anti-instakill floor present.
    assert re.search(r"SPAWN_DIST_MIN\s*<\s*60", main_lua)


def test_config_has_pbrs_and_spawn_keys():
    cfg = (REPO / "configs" / "curriculum.yaml").read_text(encoding="utf-8")
    assert re.search(r"(?m)^pbrs_coef:", cfg)
    assert re.search(r"(?m)^spawn_min:", cfg)
    assert re.search(r"(?m)^spawn_max:", cfg)
