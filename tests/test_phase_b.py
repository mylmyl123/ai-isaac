"""Tests for Phase B additions:
- B1: distributional value function (twohot encoding)
- B2: curriculum scheduler
- B3: predict-future-rewards aux head
- B4: latent variable z conditioning
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from isaac_rl.curriculum import CurriculumScheduler
from isaac_rl.model import IsaacPolicy, PolicyConfig
from isaac_rl.spaces import (
    ENEMY_FEATS, GLOBAL_DIM, MAX_ENEMIES, MAX_PICKUPS, MAX_PROJECTILES,
    PASSIVES_K, PICKUP_FEATS, PLAYER_DIM, PLAYER_HISTORY_DIM, PROJ_FEATS,
    ROOM_H, ROOM_W, SPATIAL_DIM, Z_DIM,
    ACTION_FACTORS, DOOR_FEATS, CHARACTER_K,
    ACTIVE_SLOTS, ACTIVE_FEATS, TRINKET_SLOTS, TRINKET_FEATS,
    CARD_SLOTS, CARD_FEATS, PILL_SLOTS, PILL_FEATS, TRANSFORMATION_COUNT,
)


def _random_batch(B: int = 2, z_dim: int = 16) -> dict[str, torch.Tensor]:
    b = {
        "player":            torch.randn(B, PLAYER_DIM),
        "global":            torch.randn(B, GLOBAL_DIM),
        "passives":          torch.zeros(B, PASSIVES_K),
        "last_action":       torch.zeros(B, len(ACTION_FACTORS)),
        "enemies_feats":     torch.randn(B, MAX_ENEMIES, ENEMY_FEATS),
        "enemies_mask":      torch.zeros(B, MAX_ENEMIES),
        "projectiles_feats": torch.randn(B, MAX_PROJECTILES, PROJ_FEATS),
        "projectiles_mask":  torch.zeros(B, MAX_PROJECTILES),
        "pickups_feats":     torch.randn(B, MAX_PICKUPS, PICKUP_FEATS),
        "pickups_mask":      torch.zeros(B, MAX_PICKUPS),
        "room_grid":         torch.zeros(B, 4, ROOM_H, ROOM_W),
        "doors":             torch.zeros(B, 4, DOOR_FEATS),
        "spatial":           torch.zeros(B, SPATIAL_DIM),
        "player_history":    torch.zeros(B, PLAYER_HISTORY_DIM),
        # Track A (2026-07-12) keys.
        "character":         torch.zeros(B, CHARACTER_K),
        "active_items":      torch.zeros(B, ACTIVE_SLOTS, ACTIVE_FEATS),
        "trinkets":          torch.zeros(B, TRINKET_SLOTS, TRINKET_FEATS),
        "cards":             torch.zeros(B, CARD_SLOTS, CARD_FEATS),
        "pills":             torch.zeros(B, PILL_SLOTS, PILL_FEATS),
        "transformations":   torch.zeros(B, TRANSFORMATION_COUNT),
    }
    if z_dim > 0:
        b["z"] = torch.zeros(B, z_dim)
    return b


# ---- B1: distributional value ---------------------------------------------


def test_twohot_target_sums_to_one():
    """Twohot-encoded target distributions must sum to 1."""
    policy = IsaacPolicy(PolicyConfig(value_atoms=51))
    returns = torch.tensor([-5.0, -0.5, 0.0, 0.3, 5.0, 15.0])
    target = policy.value_twohot_target(returns)
    sums = target.sum(-1)
    for s in sums:
        assert abs(float(s) - 1.0) < 1e-5


def test_twohot_target_correct_atoms():
    """A return exactly on an atom -> all mass on that atom."""
    # 51 atoms from -20 to 20 -> atom step delta = 40/50 = 0.8, atoms at
    # -20, -19.2, -18.4, ..., 0, ..., 20. Zero is on an atom.
    policy = IsaacPolicy(PolicyConfig(value_atoms=51, value_v_min=-20.0, value_v_max=20.0))
    returns = torch.tensor([0.0])
    target = policy.value_twohot_target(returns)
    # Atom index for 0: (0 - (-20)) / 0.8 = 25
    assert target[0, 25] > 0.99   # ~1
    assert target[0, 24] < 0.01   # rest is 0


def test_twohot_clamps_outside_range():
    """A return outside [v_min, v_max] gets clamped to boundary atom."""
    policy = IsaacPolicy(PolicyConfig(value_atoms=51, value_v_min=-20.0, value_v_max=20.0))
    returns = torch.tensor([100.0, -100.0])
    target = policy.value_twohot_target(returns)
    # Should have all mass on the boundary atoms.
    assert target[0, -1] > 0.99   # +100 -> last atom
    assert target[1, 0] > 0.99    # -100 -> first atom


def test_value_from_head_returns_scalar():
    """_value_from_head must return a scalar per batch element even with atoms."""
    policy = IsaacPolicy(PolicyConfig(value_atoms=51))
    logits = torch.randn(3, 51)
    v = policy._value_from_head(logits)
    assert v.shape == (3,)


def test_value_from_head_scalar_mode():
    """value_atoms=1 -> pass-through."""
    policy = IsaacPolicy(PolicyConfig(value_atoms=1))
    logits = torch.tensor([[2.5], [-1.0], [0.0]])
    v = policy._value_from_head(logits)
    assert v.shape == (3,)
    assert abs(float(v[0]) - 2.5) < 1e-6


# ---- B2: curriculum scheduler ---------------------------------------------


class _StubCfg:
    ent_coef = 0.02
    lr = 3e-4


def test_curriculum_empty_stages_no_op():
    sched = CurriculumScheduler([])
    cfg = _StubCfg()
    changed = sched.apply(cfg, reward_shaper=None, global_step=1000)
    assert not changed
    assert cfg.ent_coef == 0.02


def test_curriculum_applies_first_stage():
    stages = [
        {"until_step": 500_000, "overrides": {"ent_coef": 0.05}},
        {"until_step": 1_500_000, "overrides": {"ent_coef": 0.02}},
    ]
    sched = CurriculumScheduler(stages)
    cfg = _StubCfg()
    sched.apply(cfg, reward_shaper=None, global_step=100_000)
    assert cfg.ent_coef == 0.05


def test_curriculum_transitions_between_stages():
    stages = [
        {"until_step": 500_000, "overrides": {"ent_coef": 0.05}},
        {"until_step": 1_500_000, "overrides": {"ent_coef": 0.02}},
    ]
    sched = CurriculumScheduler(stages)
    cfg = _StubCfg()
    changed1 = sched.apply(cfg, reward_shaper=None, global_step=100_000)
    assert changed1
    assert cfg.ent_coef == 0.05
    changed2 = sched.apply(cfg, reward_shaper=None, global_step=500_001)
    assert changed2
    assert cfg.ent_coef == 0.02
    # No change on subsequent call within same stage.
    changed3 = sched.apply(cfg, reward_shaper=None, global_step=800_000)
    assert not changed3


def test_curriculum_past_all_stages():
    stages = [{"until_step": 500_000, "overrides": {"ent_coef": 0.05}}]
    sched = CurriculumScheduler(stages)
    cfg = _StubCfg()
    sched.apply(cfg, reward_shaper=None, global_step=100_000)
    assert cfg.ent_coef == 0.05
    # Past the only stage -> current_stage returns None, ent_coef unchanged.
    changed = sched.apply(cfg, reward_shaper=None, global_step=600_000)
    assert changed   # stage_idx went from 0 to -1
    # cfg.ent_coef stays at the last-applied value (0.05); this is expected
    # behavior. If you want to revert to base, add a final stage explicitly.


# ---- B3: predict-future-rewards aux head ---------------------------------


def test_reward_pred_head_output_shape():
    policy = IsaacPolicy(PolicyConfig(reward_pred_horizon=8))
    batch = _random_batch(B=2)
    # Convert single-frame batch into T=1 sequence.
    seq = {k: v.unsqueeze(0) for k, v in batch.items()}
    dones = torch.zeros(1, 2)
    init = policy.initial_hidden(2, torch.device("cpu"))
    logits, values, aux, reward_pred, value_logits = policy.sequence_forward(seq, dones, init)
    assert reward_pred.shape == (2, 8)   # T*B rows, N horizon


# ---- B4: latent variable z conditioning ----------------------------------


def test_z_input_shape_dependence():
    """Different z should produce different trunk outputs."""
    policy = IsaacPolicy(PolicyConfig(z_dim=16))
    batch = _random_batch(B=2, z_dim=16)
    batch["z"] = torch.zeros(2, 16)
    trunk1 = policy.encode(batch)
    batch["z"] = torch.ones(2, 16)
    trunk2 = policy.encode(batch)
    # Trunk outputs must differ (z was actually consumed).
    diff = (trunk1 - trunk2).abs().mean()
    assert float(diff) > 1e-6


def test_z_dim_zero_disables():
    """z_dim=0 -> policy accepts obs without z key."""
    policy = IsaacPolicy(PolicyConfig(z_dim=0))
    batch = _random_batch(B=2, z_dim=0)
    # Should not raise.
    trunk = policy.encode(batch)
    assert trunk.shape == (2, policy.cfg.trunk_dim)
