"""Tests for RND intrinsic curiosity module.

Verifies:
  * Target network has no gradient (frozen at init).
  * Predictor loss decreases when trained on a fixed batch.
  * Intrinsic reward is higher on novel states than on repeated ones.
  * EMA normalization keeps intrinsic reward at unit scale over training.
"""
from __future__ import annotations

import torch

from isaac_rl.dreamer.intrinsic import RND


def test_target_network_has_no_gradient():
    rnd = RND(feat_dim=64, embed_dim=32, hidden=64, target_hidden=32)
    for p in rnd.target.parameters():
        assert not p.requires_grad


def test_predictor_loss_decreases_on_fixed_batch():
    torch.manual_seed(0)
    rnd = RND(feat_dim=32, embed_dim=16, hidden=64, target_hidden=32)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    batch = torch.randn(64, 32)
    losses = []
    for _ in range(30):
        loss, _ = rnd.update(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    # After training, predictor loss on THE SAME batch should be much lower.
    assert losses[-1] < 0.5 * losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_intrinsic_reward_higher_on_novel_than_familiar():
    """After training the predictor on batch A, the intrinsic reward on
    batch A should be lower than on a fresh batch B."""
    torch.manual_seed(0)
    rnd = RND(feat_dim=32, embed_dim=16, hidden=64, target_hidden=32)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    familiar = torch.randn(64, 32)
    novel = torch.randn(64, 32)  # different seed
    # Train on 'familiar' many times.
    for _ in range(50):
        loss, _ = rnd.update(familiar)
        opt.zero_grad()
        loss.backward()
        opt.step()
    r_familiar = rnd.intrinsic_reward(familiar).mean()
    r_novel = rnd.intrinsic_reward(novel).mean()
    assert r_novel > r_familiar, f"novelty signal broken: novel={r_novel:.4f} familiar={r_familiar:.4f}"


def test_ema_normalization_stabilizes_magnitude():
    """After the predictor converges, the EMA-normalized intrinsic reward
    should have magnitude near 1 (unit-normalized by design)."""
    torch.manual_seed(0)
    rnd = RND(feat_dim=32, embed_dim=16, hidden=64, target_hidden=32)
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    for _ in range(100):
        batch = torch.randn(32, 32)
        loss, _ = rnd.update(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # After 100 updates the EMA std should stabilize. Intrinsic reward on
    # a fresh batch should be O(1) in magnitude, not exploding or vanishing.
    r = rnd.intrinsic_reward(torch.randn(64, 32)).mean()
    assert 0.01 < r < 100.0, f"intrinsic reward magnitude broken: {r}"


def test_output_shape_preserves_batch_dims():
    rnd = RND(feat_dim=16, embed_dim=8, hidden=32, target_hidden=16)
    # Feed a 3D tensor (H, B, feat): should return 2D (H, B).
    x = torch.randn(5, 8, 16)
    r = rnd.intrinsic_reward(x)
    assert r.shape == (5, 8)
