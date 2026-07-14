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
    # Legacy schema: {kind: 'kill'}
    total, terminated, bd = s(
        {"events": [{"kind": "kill"}], "player": {"hp_red": 3}},
        action=None,
    )
    assert total == pytest.approx(1.0 - 0.001)
    assert terminated is False
    assert bd["kill"] == 1.0
    assert bd["step"] == pytest.approx(-0.001)


def test_reward_shaper_kill_via_damage_to_npc():
    """Regression: mod actually emits {kind: damage_to_npc, killed: true}, not
    {kind: kill}. Schema mismatch caused zero kills through 5000 steps of
    Stage A prior to fix."""
    s = RewardShaper()
    total, _, bd = s(
        {"events": [{"kind": "damage_to_npc", "killed": True, "dmg": 3, "npc_type": 18}],
         "player": {"hp_red": 3}},
        action=None,
    )
    assert bd.get("kill") == 1.0, "damage_to_npc + killed=True must count as a kill"
    # Non-lethal damage_to_npc should NOT count as a kill.
    s2 = RewardShaper()
    _, _, bd2 = s2(
        {"events": [{"kind": "damage_to_npc", "killed": False, "dmg": 1}],
         "player": {"hp_red": 3}},
        action=None,
    )
    assert "kill" not in bd2, "damage_to_npc without killed=True is not a kill"


def test_action_masking_active_factors():
    """Phase-1 fix: unused action factors on Stage 0/A/B should be masked out
    of the loss so entropy bonus doesn't leak into useless heads.

    Verifies:
    - active_factors=2 (move + shoot only) samples all 5 factors but zeros
      factors [2, 3, 4]
    - Log prob and entropy are summed over only the 2 active factors
    - active_factors=5 (default) uses all factors
    """
    import torch
    from isaac_rl.cleanrl_ppo import ActorCritic
    from isaac_rl.spaces import ACTION_FACTORS

    torch.manual_seed(0)
    obs_dim = 32
    for k in (2, len(ACTION_FACTORS)):
        net = ActorCritic(obs_dim, hidden_dim=64, n_layers=1, active_factors=k)
        x = torch.randn(8, obs_dim)
        actions, logp, ent, v = net.act(x)
        assert actions.shape == (8, len(ACTION_FACTORS))
        # Masked factors must all be 0.
        if k < len(ACTION_FACTORS):
            assert (actions[:, k:] == 0).all(), f"masked factors must be 0, got {actions[:, k:]}"
        # Entropy scale: sum over k active factors of log(n_choices_i).
        # For k=2: log(9)+log(5) ≈ 3.80. For k=5: log(9)+log(5)+log(2)*3 ≈ 5.88.
        max_ent = sum([
            float(torch.log(torch.tensor(float(ACTION_FACTORS[i]))))
            for i in range(k)
        ])
        assert ent.mean().item() <= max_ent + 0.01, \
            f"entropy {ent.mean().item()} exceeds max {max_ent} for k={k}"

        # Round-trip through evaluate() with the same actions.
        logp2, ent2, v2 = net.evaluate(x, actions)
        assert torch.allclose(logp, logp2, atol=1e-5), "act() vs evaluate() logprob mismatch"


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
