"""Random Network Distillation (Burda et al., 2018) intrinsic reward.

- Target network: fixed random init, never trained.
- Predictor network: trained to minimize MSE against target on observed states.
- Intrinsic reward: normalized MSE per state → high on novel states.

Observation input is the same trunk-encoded state the policy sees, detached.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        b_mean = float(x.mean().item())
        b_var = float(x.var(unbiased=False).item()) if x.numel() > 1 else 0.0
        b_count = x.numel()
        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        M2 = m_a + m_b + (delta ** 2) * self.count * b_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    def std(self) -> float:
        return max(self.var ** 0.5, 1e-6)


class RND(nn.Module):
    """RND module. `feat_dim` = size of the trunk features from IsaacPolicy.encode()."""

    def __init__(self, feat_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        def mlp():
            return nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, out_dim),
            )
        self.target = mlp()
        self.predictor = mlp()
        for p in self.target.parameters():
            p.requires_grad = False
        self.reward_rms = _RunningMeanStd()

    def intrinsic_reward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [B, feat_dim], detached. Returns [B] intrinsic reward."""
        with torch.no_grad():
            tgt = self.target(feats)
            pred = self.predictor(feats)
            err = ((pred - tgt) ** 2).mean(-1)
            self.reward_rms.update(err.detach())
            return err / self.reward_rms.std()

    def loss(self, feats: torch.Tensor) -> torch.Tensor:
        """MSE between predictor and (frozen) target."""
        with torch.no_grad():
            tgt = self.target(feats)
        pred = self.predictor(feats)
        return F.mse_loss(pred, tgt)
