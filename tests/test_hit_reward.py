"""Tests for the dense per-hit reward (Phase-2c, 2026-07-14).

The hit reward must (a) fire on non-lethal tear connects, (b) be bounded so
total hit reward over one enemy's life ~= r_hit (so r_kill stays dominant and
hits can't be farmed), and (c) be a no-op when r_hit=0.

Run:
    PYTHONPATH=python pytest tests/test_hit_reward.py -q
"""
from __future__ import annotations

import pytest

from isaac_rl.reward import RewardConfig, RewardShaper


def _dmg_event(dmg, max_hp, killed=False):
    return {"kind": "damage_to_npc", "dmg": dmg, "npc_max_hp": max_hp, "killed": killed}


def _raw(events):
    return {"player": {"hp_red": 3}, "events": events}


def test_hit_reward_off_by_default_is_noop():
    sh = RewardShaper(RewardConfig())  # r_hit defaults to 0.0
    sh.reset()
    _, _, bd = sh(_raw([_dmg_event(3.5, 10.0)]))
    assert "hit" not in bd
    assert bd == {"step": pytest.approx(-0.001)}


def test_hit_reward_fires_on_nonlethal_connect():
    sh = RewardShaper(RewardConfig(r_hit=0.3))
    sh.reset()
    _, _, bd = sh(_raw([_dmg_event(3.5, 10.0, killed=False)]))
    # 0.3 * (3.5/10) = 0.105
    assert bd["hit"] == pytest.approx(0.3 * 0.35)
    assert "kill" not in bd


def test_kill_takes_precedence_over_hit_on_lethal_blow():
    sh = RewardShaper(RewardConfig(r_hit=0.3, r_kill=1.0))
    sh.reset()
    _, _, bd = sh(_raw([_dmg_event(4.0, 10.0, killed=True)]))
    # Killing blow credits r_kill, NOT r_hit (elif branch).
    assert bd.get("kill") == pytest.approx(1.0)
    assert "hit" not in bd


def test_total_hit_reward_bounded_by_r_hit_over_enemy_life():
    """Three hits totalling the enemy's max HP must sum to ~= r_hit, so a kill
    (r_kill=1.0) always out-earns the accumulated hits (r_hit=0.3)."""
    r_hit = 0.3
    sh = RewardShaper(RewardConfig(r_hit=r_hit, r_kill=1.0))
    sh.reset()
    total_hit = 0.0
    # 3 tears of 3.5 dmg vs 10 HP: fracs 0.35+0.35+0.30 = 1.0
    for dmg, killed in [(3.5, False), (3.5, False), (3.0, True)]:
        _, _, bd = sh(_raw([_dmg_event(dmg, 10.0, killed=killed)]))
        total_hit += bd.get("hit", 0.0)
    # Non-lethal hits summed to ~r_hit * (0.35+0.35) = 0.21; the lethal blow
    # paid r_kill not r_hit. So accumulated hit reward < r_hit < r_kill.
    assert total_hit == pytest.approx(r_hit * 0.70)
    assert total_hit < 1.0   # strictly less than one kill


def test_hit_reward_frac_clipped_to_one():
    # A single overkill tear (dmg > max_hp) contributes at most r_hit, not more.
    sh = RewardShaper(RewardConfig(r_hit=0.3))
    sh.reset()
    _, _, bd = sh(_raw([_dmg_event(99.0, 10.0, killed=False)]))
    assert bd["hit"] == pytest.approx(0.3)   # min(1.0, 9.9) * 0.3


def test_hit_reward_safe_when_max_hp_missing():
    sh = RewardShaper(RewardConfig(r_hit=0.3))
    sh.reset()
    _, _, bd = sh(_raw([{"kind": "damage_to_npc", "dmg": 3.5, "killed": False}]))
    # No npc_max_hp -> frac 0 -> no hit reward, no crash.
    assert bd.get("hit", 0.0) == pytest.approx(0.0)


def test_config_and_ppoconfig_have_r_hit():
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    assert re.search(r"(?m)^r_hit:", (repo / "configs" / "curriculum.yaml").read_text())
    from isaac_rl.cleanrl_ppo import PPOConfig
    assert hasattr(PPOConfig(), "r_hit")
