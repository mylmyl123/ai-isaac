"""Tests for Priority-1 improvements (2026-07-02):

- symlog / symexp transformation
- Spatial feature encoding
- Kickstarting KL computation
- LayerNorm + orthogonal init sanity
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from isaac_rl.model import IsaacPolicy, PolicyConfig
from isaac_rl.ppo import symlog, symexp
from isaac_rl.spaces import (
    MAX_ENEMIES, MAX_PICKUPS, MAX_PROJECTILES,
    ENEMY_FEATS, PICKUP_FEATS, PROJ_FEATS,
    PLAYER_DIM, GLOBAL_DIM, PASSIVES_K, ROOM_H, ROOM_W,
    SPATIAL_DIM,
    PLAYER_HISTORY_DIM,
    _compute_spatial,
    zero_obs,
)


# ---- symlog / symexp -------------------------------------------------------


def test_symlog_zero_maps_to_zero():
    assert float(symlog(torch.tensor(0.0))) == 0.0


def test_symlog_sign_preserved():
    x = torch.tensor([-50.0, -3.0, -0.5, 0.5, 3.0, 50.0])
    y = symlog(x)
    assert torch.all(torch.sign(y) == torch.sign(x))


def test_symlog_monotonic():
    x = torch.linspace(-10, 10, 100)
    y = symlog(x)
    diffs = y[1:] - y[:-1]
    assert torch.all(diffs > 0), "symlog must be strictly monotonic"


def test_symlog_compresses_large_magnitudes():
    """symlog(50) ~= 3.93 (log(51)), much smaller than 50."""
    y = float(symlog(torch.tensor(50.0)))
    assert 3.9 < y < 4.0


def test_symlog_near_identity_for_small_values():
    """For |x| << 1, symlog(x) ~= x (Taylor expansion)."""
    x = torch.tensor([-0.01, -0.001, 0.001, 0.01])
    y = symlog(x)
    for i in range(4):
        assert abs(float(y[i]) - float(x[i])) < 0.001


def test_symlog_symexp_inverse():
    x = torch.tensor([-50.0, -3.0, -0.1, 0.0, 0.1, 3.0, 50.0])
    recovered = symexp(symlog(x))
    for orig, rec in zip(x, recovered):
        assert abs(float(orig) - float(rec)) < 1e-4


# ---- Spatial features ------------------------------------------------------


def test_spatial_features_zero_without_bounds():
    """No room_bounds field -> all zeros (backward compat with schema v1)."""
    raw = {"player": {"x": 300.0, "y": 300.0}}
    feats = _compute_spatial(raw)
    assert feats.shape == (SPATIAL_DIM,)
    assert np.all(feats == 0)


def test_spatial_features_player_at_center():
    """Player at room center -> normalized pos (0, 0), equal wall dists."""
    raw = {
        "player": {"x": 340.0, "y": 300.0},
        "room_bounds": {"tl_x": 80.0, "tl_y": 160.0, "br_x": 600.0, "br_y": 440.0},
        "doors": [],
    }
    feats = _compute_spatial(raw)
    # Position normalized to [-1, 1]: at center should be (0, 0).
    assert abs(feats[0]) < 0.01
    assert abs(feats[1]) < 0.01
    # Wall distances at center should all be ~0.5.
    for i in range(2, 6):
        assert abs(feats[i] - 0.5) < 0.01


def test_spatial_features_player_at_corner():
    """Player at top-left corner -> normalized pos (-1, -1)."""
    raw = {
        "player": {"x": 80.0, "y": 160.0},
        "room_bounds": {"tl_x": 80.0, "tl_y": 160.0, "br_x": 600.0, "br_y": 440.0},
        "doors": [],
    }
    feats = _compute_spatial(raw)
    assert abs(feats[0] - (-1.0)) < 0.01
    assert abs(feats[1] - (-1.0)) < 0.01
    # Left and up walls at 0, right and down walls at 1.
    assert abs(feats[2] - 0.0) < 0.01   # dl
    assert abs(feats[3] - 0.0) < 0.01   # du
    assert abs(feats[4] - 1.0) < 0.01   # dr
    assert abs(feats[5] - 1.0) < 0.01   # dd


def test_spatial_features_door_direction():
    """Player at center, only RIGHT door open -> door direction points right."""
    raw = {
        "player": {"x": 340.0, "y": 300.0},
        "room_bounds": {"tl_x": 80.0, "tl_y": 160.0, "br_x": 600.0, "br_y": 440.0},
        "doors": [
            [0, 0, 0, 0, 0, 0],   # LEFT: doesn't exist
            [0, 0, 0, 0, 0, 0],   # UP: doesn't exist
            [1, 1, 0, 0, 0, 0],   # RIGHT: open
            [0, 0, 0, 0, 0, 0],   # DOWN: doesn't exist
        ],
    }
    feats = _compute_spatial(raw)
    # Door direction should be (+1, 0): unit vector pointing right.
    assert feats[6] > 0.9, f"expected +1 for right door, got {feats[6]}"
    assert abs(feats[7]) < 0.1, f"expected 0 for right door, got {feats[7]}"


def test_spatial_features_no_open_doors():
    """No open doors -> door direction is (0, 0)."""
    raw = {
        "player": {"x": 340.0, "y": 300.0},
        "room_bounds": {"tl_x": 80.0, "tl_y": 160.0, "br_x": 600.0, "br_y": 440.0},
        "doors": [
            [1, 0, 0, 0, 0, 0],   # LEFT: exists but not open
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    }
    feats = _compute_spatial(raw)
    assert abs(feats[6]) < 0.01
    assert abs(feats[7]) < 0.01


# ---- Model integration -----------------------------------------------------


def _random_batch(B: int = 2) -> dict[str, torch.Tensor]:
    return {
        "player":            torch.randn(B, PLAYER_DIM),
        "global":            torch.randn(B, GLOBAL_DIM),
        "passives":          torch.zeros(B, PASSIVES_K),
        "last_action":       torch.zeros(B, 2),
        "enemies_feats":     torch.randn(B, MAX_ENEMIES, ENEMY_FEATS),
        "enemies_mask":      torch.zeros(B, MAX_ENEMIES),
        "projectiles_feats": torch.randn(B, MAX_PROJECTILES, PROJ_FEATS),
        "projectiles_mask":  torch.zeros(B, MAX_PROJECTILES),
        "pickups_feats":     torch.randn(B, MAX_PICKUPS, PICKUP_FEATS),
        "pickups_mask":      torch.zeros(B, MAX_PICKUPS),
        "room_grid":         torch.zeros(B, 4, ROOM_H, ROOM_W),
        "doors":             torch.zeros(B, 4, 6),
        "spatial":           torch.zeros(B, SPATIAL_DIM),
        "player_history":    torch.zeros(B, PLAYER_HISTORY_DIM),
    }


def test_model_accepts_spatial_features():
    """Regression: model must accept the new 'spatial' obs key."""
    policy = IsaacPolicy()
    batch = _random_batch(B=3)
    hidden = policy.initial_hidden(3, torch.device("cpu"))
    logits, value, new_h = policy.step(batch, hidden)
    assert len(logits) == 2   # move + shoot heads
    assert value.shape == (3,)
    assert new_h.shape == (3, policy.cfg.gru_dim)


def test_policy_heads_near_uniform_at_init():
    """Orthogonal init with gain=0.01 for policy heads should give
    near-uniform action distribution at init (critical for BC preservation)."""
    policy = IsaacPolicy()
    batch = _random_batch(B=8)
    hidden = policy.initial_hidden(8, torch.device("cpu"))
    with torch.no_grad():
        logits, _, _ = policy.step(batch, hidden)
    for head_logits in logits:
        # Standard deviation of logits should be small at init because of
        # gain=0.01. If gain was default (sqrt(2)), std would be much larger.
        std = float(head_logits.std())
        assert std < 0.5, f"policy logits std={std:.3f} too high; check gain"


def test_layer_norm_present():
    """Regression: LayerNorm was added to trunk MLPs. Verify by module count."""
    policy = IsaacPolicy()
    ln_count = sum(1 for m in policy.modules() if isinstance(m, nn.LayerNorm))
    # Expect LayerNorm on: player, global, passives, last_action, enemy_encoder,
    # proj_encoder, pickup_encoder, doors, spatial, trunk (10 sub-MLPs) plus one
    # for the post-GRU normalization. Some MLPs have 2 LN layers (2-hidden).
    assert ln_count >= 10, f"only found {ln_count} LayerNorm modules"


# ---- Kickstarting KL computation ------------------------------------------


def test_kickstart_kl_zero_between_identical_policies():
    """KL divergence between a policy and itself must be zero."""
    student = IsaacPolicy()
    teacher = IsaacPolicy()
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    batch = _random_batch(B=4)
    hidden = student.initial_hidden(4, torch.device("cpu"))
    with torch.no_grad():
        s_logits, _, _ = student.step(batch, hidden)
        t_logits, _, _ = teacher.step(batch, hidden)

    kl_total = torch.zeros(4)
    for sl, tl in zip(s_logits, t_logits):
        sp_lp = torch.log_softmax(sl, dim=-1)
        tp_lp = torch.log_softmax(tl, dim=-1)
        sp = sp_lp.exp()
        kl = (sp * (sp_lp - tp_lp)).sum(-1)
        kl_total = kl_total + kl
    assert float(kl_total.mean()) < 1e-5


def test_kickstart_kl_positive_for_different_policies():
    """KL must be positive when policies differ."""
    student = IsaacPolicy()
    teacher = IsaacPolicy()
    # Randomize teacher differently.
    for p in teacher.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    teacher.eval()

    batch = _random_batch(B=4)
    hidden = student.initial_hidden(4, torch.device("cpu"))
    with torch.no_grad():
        s_logits, _, _ = student.step(batch, hidden)
        t_logits, _, _ = teacher.step(batch, hidden)

    kl_total = torch.zeros(4)
    for sl, tl in zip(s_logits, t_logits):
        sp_lp = torch.log_softmax(sl, dim=-1)
        tp_lp = torch.log_softmax(tl, dim=-1)
        sp = sp_lp.exp()
        kl = (sp * (sp_lp - tp_lp)).sum(-1)
        kl_total = kl_total + kl
    # KL >= 0 always; for different distributions it should be > 0.
    assert float(kl_total.mean()) > 0
