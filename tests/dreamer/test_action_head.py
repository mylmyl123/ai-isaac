"""MultiDiscreteActionHead: sampling, log-prob, entropy, indices <-> one-hot."""
import math

import torch

from isaac_rl.dreamer.action import (
    MultiDiscreteActionHead,
    indices_to_onehot,
    onehot_to_indices,
)


def test_action_head_output_shapes():
    head = MultiDiscreteActionHead(feat_size=256, factors=(9, 5), unimix_ratio=0.01)
    feat = torch.randn(3, 5, 256)
    dist = head(feat)

    a = dist.sample()
    assert a.shape == (3, 5, 14)

    lp = dist.log_prob(a)
    assert lp.shape == (3, 5)
    assert torch.isfinite(lp).all()

    ent = dist.entropy()
    assert ent.shape == (3, 5)
    assert torch.isfinite(ent).all()


def test_action_head_uniform_entropy_at_init():
    """With small-init logits + unimix=0.01, entropy should be near max = ln(9)+ln(5)."""
    head = MultiDiscreteActionHead(feat_size=256, factors=(9, 5), unimix_ratio=0.01)
    feat = torch.zeros(8, 256)
    dist = head(feat)
    ent = dist.entropy().mean().item()
    max_ent = math.log(9) + math.log(5)
    assert abs(ent - max_ent) < 0.05, f"entropy {ent} not close to max {max_ent}"


def test_indices_onehot_roundtrip():
    factors = (9, 5)
    indices = torch.tensor([[[0, 0], [4, 2], [8, 4]], [[3, 1], [7, 0], [1, 3]]])   # [2, 3, 2]
    oh = indices_to_onehot(indices, factors)
    assert oh.shape == (2, 3, 14)
    # Rows must sum to 2 (one 1 per factor).
    assert torch.all(oh.sum(dim=-1) == 2.0)
    rec = onehot_to_indices(oh, factors)
    assert torch.equal(rec, indices)


def test_action_gradient_flows():
    head = MultiDiscreteActionHead(feat_size=256, factors=(9, 5), hidden=64, layers=1)
    feat = torch.randn(4, 256, requires_grad=False)
    dist = head(feat)
    a = dist.sample()
    loss = -dist.log_prob(a).mean() - 0.01 * dist.entropy().mean()
    loss.backward()
    n_grad = sum(1 for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "no gradient reached actor params"


def test_action_head_sample_is_valid_multihot():
    """Each sampled action should be a concatenation of one-hots (one 1 per factor)."""
    head = MultiDiscreteActionHead(feat_size=128, factors=(9, 5), unimix_ratio=0.0)
    feat = torch.randn(100, 128)
    dist = head(feat)
    a = dist.sample()
    # First 9 dims should have exactly one 1; last 5 dims should have exactly one 1.
    assert torch.all(a[:, :9].sum(dim=-1).round() == 1.0)
    assert torch.all(a[:, 9:].sum(dim=-1).round() == 1.0)
