"""Regression test for the 2026-07-11 v_min/v_max wiring fix.

Before this fix, `DreamerConfig.value_v_min/value_v_max` were dead code:
the vendored `networks.MLP` constructed `DiscDist` without passing `low`/
`high`, so bin range was hardcoded at [-20, 20]. This test ensures a
config-level change to those values actually reaches the DiscDist buckets.
"""
from __future__ import annotations

import torch

from isaac_rl.dreamer.vendor import networks


def test_mlp_symlog_disc_uses_config_low_high():
    """MLP built with disc_low/disc_high should produce a DiscDist whose
    bucket range matches (in symlog space)."""
    mlp = networks.MLP(
        inp_dim=8,
        shape=(255,),
        layers=1,
        units=16,
        dist="symlog_disc",
        outscale=0.0,
        device="cpu",
        name="test",
        disc_low=-100.0,
        disc_high=100.0,
    )
    x = torch.zeros(1, 8)
    dist = mlp(x)
    # DiscDist stores buckets in symlog-transformed space [low, high]; bucket[0]
    # should be exactly disc_low and buckets[-1] should be disc_high.
    assert float(dist.buckets[0].item()) == -100.0
    assert float(dist.buckets[-1].item()) == 100.0
    assert dist.buckets.shape[0] == 255


def test_mlp_symlog_disc_default_range_unchanged():
    """Regression guard: absent disc_low/disc_high the default range is still
    [-20, 20] (matches DreamerV3 paper default)."""
    mlp = networks.MLP(
        inp_dim=8,
        shape=(255,),
        layers=1,
        units=16,
        dist="symlog_disc",
        outscale=0.0,
        device="cpu",
        name="test_default",
    )
    x = torch.zeros(1, 8)
    dist = mlp(x)
    assert float(dist.buckets[0].item()) == -20.0
    assert float(dist.buckets[-1].item()) == 20.0
