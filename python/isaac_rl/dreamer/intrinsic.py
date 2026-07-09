"""Random Network Distillation (RND) for intrinsic curiosity in Dreamer.

Reference: Burda et al., "Exploration by Random Network Distillation" (ICLR 2019).
Applied to hard-exploration Atari games like Montezuma's Revenge.

Isaac has the same structural challenge: long chains of specific actions
(buy from shop -> bomb secret door -> use item -> reroll with D6) are
almost impossible to discover via random exploration. RND provides a
dense intrinsic reward for visiting novel states, which lets the agent
explore beyond what hand-crafted rewards prescribe.

DESIGN:
  * Target network `f_tgt`: fixed random-init MLP, no gradient.
  * Predictor `f_pred`: trainable MLP, learns to match `f_tgt` on visited states.
  * Intrinsic reward: ||f_tgt(feat) - f_pred(feat)||^2, normalized by EMA std.
  * Naturally decays: predictor becomes accurate on visited states,
    remains high on novel ones.

INTEGRATION POINT:
  The 'feat' input is the RSSM feature (deter + stoch concatenation) that
  Dreamer already computes for the critic and reward head. This means RND
  operates on the WM's LATENT representation of the state -- which is
  much cleaner than raw obs (avoids high-dim noise, low-dim semantic).

  Intrinsic reward is added to extrinsic reward during behavior training:

      total_reward = extrinsic_reward + rnd_intrinsic_scale * intrinsic

  So the critic learns Q-values that include curiosity, and the actor
  is trained to seek novel states through imagination. No changes to
  env-side reward flow required.

DECAY:
  Predictor loss serves double duty as a training signal AND a novelty
  measure. Early in training: predictor loss is high everywhere -> huge
  intrinsic reward -> agent explores wildly. Later: predictor fits common
  states -> intrinsic reward -> 0 there -> agent focuses on remaining
  novel states. Eventually: predictor fits nearly everywhere -> intrinsic
  reward vanishes -> extrinsic reward dominates.

  This is exactly the exploration schedule you'd want.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RND(nn.Module):
    """Random Network Distillation for intrinsic curiosity.

    Args:
        feat_dim: Dimension of the WM feature vector (deter + stoch*discrete
            for Dreamer). Both target and predictor take this as input.
        embed_dim: Dimension of the target/predictor output embedding.
            The intrinsic reward is the MSE between the two outputs, so
            larger embed_dim gives more granular novelty signal at the
            cost of more parameters. 128 is a well-tested default.
        hidden: Hidden layer width for both target and predictor MLPs.
            Predictor is slightly deeper (3 layers) so it can approximate
            the target (2 layers). This "predictor capacity > target
            capacity" is critical -- if they were equal, the predictor
            could trivially copy the target's weights on visited states.
        target_hidden: Target network hidden width. Keep smaller than
            predictor so the predictor has excess capacity to fit the
            target on visited states.
    """

    def __init__(
        self,
        feat_dim: int,
        embed_dim: int = 128,
        hidden: int = 256,
        target_hidden: int = 128,
    ):
        super().__init__()
        # Target: 2-layer MLP, random-init, FROZEN.
        self.target = nn.Sequential(
            nn.Linear(feat_dim, target_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(target_hidden, embed_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor: 3-layer MLP, trainable. Deeper than target so it has
        # capacity to approximate the target on visited states.
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
        )

        # Running statistics for intrinsic reward normalization. Without
        # normalization, RND's magnitude drifts over training and can
        # dominate or vanish w.r.t. extrinsic. Standardize per-sample MSE
        # using EMA mean/std -> stable reward magnitude across training.
        self.register_buffer("_ema_mean", torch.zeros(1))
        self.register_buffer("_ema_var", torch.ones(1))
        self._ema_decay = 0.99

    def intrinsic_reward(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward for a batch of features.

        Args:
            feat: Feature tensor of shape [..., feat_dim].

        Returns:
            Per-sample intrinsic reward of shape [...] (no trailing dim).
            Normalized by running std -> unit-scale.
        """
        with torch.no_grad():
            t = self.target(feat)
        p = self.predictor(feat)
        err = ((t - p) ** 2).mean(dim=-1)   # [...]
        # Normalize by EMA std so magnitude is stable across training.
        return err / (self._ema_var.sqrt() + 1e-8)

    def update(self, feat_batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Train predictor to match target on a batch of features.

        Args:
            feat_batch: Feature tensor of shape [B, ..., feat_dim].
                Typically the WM's posterior features from a training batch.

        Returns:
            (loss, metrics): loss is the mean predictor MSE for backward();
                metrics dict has 'rnd/predictor_loss' and
                'rnd/intrinsic_mean' (pre-normalization).
        """
        # Target is frozen -- no gradient path through it.
        with torch.no_grad():
            t = self.target(feat_batch)
        p = self.predictor(feat_batch)
        per_sample = ((t - p) ** 2).mean(dim=-1)    # [B, ...]
        loss = per_sample.mean()

        # Update EMA statistics (used by intrinsic_reward()).
        with torch.no_grad():
            flat = per_sample.detach().flatten()
            batch_mean = flat.mean()
            batch_var = flat.var()
            self._ema_mean.mul_(self._ema_decay).add_(batch_mean * (1 - self._ema_decay))
            self._ema_var.mul_(self._ema_decay).add_(batch_var * (1 - self._ema_decay))

        metrics = {
            "rnd/predictor_loss": float(loss.item()),
            "rnd/intrinsic_mean_raw": float(per_sample.mean().item()),
            "rnd/intrinsic_ema_mean": float(self._ema_mean.item()),
            "rnd/intrinsic_ema_std": float(self._ema_var.sqrt().item()),
        }
        return loss, metrics
