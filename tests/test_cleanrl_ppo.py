"""Smoke tests for the post-2026-07-13 CleanRL PPO pipeline.

These are pure-Python tests — they don't spawn Isaac. They verify the
new modules import cleanly, config loads, reward shaper works, and the
policy network forward pass produces correct-shape outputs.
"""
import numpy as np
import pytest
import torch

from isaac_rl.cleanrl_ppo import ActorCritic, PPOConfig, Rollout
from isaac_rl.reward import RewardConfig, RewardShaper
from isaac_rl.spaces import ACTION_FACTORS


def test_reward_config_has_three_terms():
    """Regression: reward config must not accumulate terms again."""
    cfg = RewardConfig()
    fields = [f for f in cfg.__dataclass_fields__ if f.startswith("r_")]
    assert len(fields) == 3, f"expected 3 r_* terms, got {len(fields)}: {fields}"
    assert set(fields) == {"r_kill", "r_death", "r_step"}


def test_reward_shaper_kill_and_step():
    s = RewardShaper()
    total, terminated, bd = s(
        {"events": [{"kind": "kill"}], "player": {"hp_red": 3}},
        action=None,
    )
    assert total == pytest.approx(1.0 - 0.001)
    assert terminated is False
    assert bd["kill"] == 1.0
    assert bd["step"] == pytest.approx(-0.001)


def test_reward_shaper_death_via_event():
    s = RewardShaper()
    total, terminated, _ = s({"events": [{"kind": "death"}], "player": {}}, action=None)
    assert terminated is True
    assert total == pytest.approx(-1.0 - 0.001)
    # Death must fire exactly once per episode.
    total2, terminated2, bd2 = s({"events": [{"kind": "death"}], "player": {}}, action=None)
    assert "death" not in bd2
    assert total2 == pytest.approx(-0.001)


def test_reward_shaper_death_via_hp():
    """HP=0 detection as fallback when 'death' event is missing."""
    s = RewardShaper()
    _, terminated, bd = s({"events": [], "player": {"hp_red": 0, "hp_soul": 0}}, action=None)
    assert terminated is True
    assert "death" in bd


def test_reward_shaper_ignores_unknown_events():
    s = RewardShaper()
    total, _, _ = s(
        {"events": [{"kind": "damage_dealt"}, {"kind": "new_room"}, {"kind": "room_clear"}]},
        action=None,
    )
    # Only r_step should fire.
    assert total == pytest.approx(-0.001)


def test_ppo_config_defaults_sensible():
    cfg = PPOConfig()
    assert cfg.rollout_length >= 64
    assert 0 < cfg.gamma < 1
    assert 0 < cfg.gae_lambda < 1
    assert 0 < cfg.clip_coef < 1
    assert cfg.stage in ("A", "B", "C", "D", "E")


def test_actor_critic_forward_shape():
    obs_dim = 128
    n_factors = len(ACTION_FACTORS)
    net = ActorCritic(obs_dim=obs_dim, hidden_dim=64, n_layers=2)
    x = torch.zeros(4, obs_dim)
    dists, v = net.forward(x)
    assert len(dists) == n_factors
    for k, d in enumerate(dists):
        assert d.logits.shape == (4, int(ACTION_FACTORS[k]))
    assert v.shape == (4,)


def test_actor_critic_act_shape():
    obs_dim = 128
    net = ActorCritic(obs_dim=obs_dim, hidden_dim=64, n_layers=2)
    x = torch.randn(3, obs_dim)
    actions, logp, ent, v = net.act(x)
    assert actions.shape == (3, len(ACTION_FACTORS))
    assert logp.shape == (3,)
    assert ent.shape == (3,)
    assert v.shape == (3,)


def test_actor_critic_evaluate_shape():
    obs_dim = 128
    n_factors = len(ACTION_FACTORS)
    net = ActorCritic(obs_dim=obs_dim, hidden_dim=64, n_layers=2)
    x = torch.randn(5, obs_dim)
    # Fake actions within each factor's range.
    actions = torch.stack(
        [torch.randint(low=0, high=int(n), size=(5,)) for n in ACTION_FACTORS],
        dim=-1,
    )
    logp, ent, v = net.evaluate(x, actions)
    assert logp.shape == (5,)
    assert ent.shape == (5,)
    assert v.shape == (5,)


def test_rollout_gae_shape():
    T, N, D = 16, 2, 8
    rb = Rollout(T, N, D, n_factors=len(ACTION_FACTORS), device=torch.device("cpu"))
    rb.rewards = torch.randn(T, N)
    rb.dones = torch.zeros(T, N)
    rb.values = torch.randn(T, N)
    next_v = torch.randn(N)
    next_done = torch.zeros(N)
    adv, ret = rb.compute_gae(next_v, next_done, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (T, N)
    assert ret.shape == (T, N)
