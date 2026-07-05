"""MultiDiscrete action head for Dreamer's actor.

Dreamer's canonical actor is a single ``OneHotDist`` over a flat categorical
action. Isaac has factored actions ``MultiDiscrete([9, 5])``: 9 move directions
and 5 shoot directions, independent. Cartesian product (45) would work but
throws away the factor structure — for the same total parameter budget you'd
learn 45 independent action logits instead of 9+5.

This module produces N parallel ``OneHotDist``s, one per factor, sampled and
scored independently. The joint log-prob and entropy factorise across
factors, which matches how PPO's ``IsaacPolicy`` treats the same action space
(model.py:429-459).

Concatenated one-hot (dim ``sum(factors) = 14``) is what feeds the RSSM's
action input — matches the vendor RSSM's ``num_actions`` scalar.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendor.tools import OneHotDist


class MultiDiscreteActionHead(nn.Module):
    """Actor head: RSSM feature -> N independent OneHotDist heads.

    Args:
      feat_size: RSSM feat dim (stoch*discrete + deter).
      factors: e.g. ``(9, 5)`` for Isaac's ``MultiDiscrete([9, 5])``.
      hidden, layers: MLP trunk before the per-factor linear projections.
      unimix_ratio: DreamerV3 default 0.01 — small uniform mixture into the
        categorical to prevent zero-probability actions.
    """

    def __init__(
        self,
        feat_size: int,
        factors: Sequence[int],
        hidden: int = 512,
        layers: int = 2,
        unimix_ratio: float = 0.01,
        act: str = "SiLU",
    ):
        super().__init__()
        self.factors = tuple(int(f) for f in factors)
        self.unimix_ratio = float(unimix_ratio)

        act_cls = getattr(nn, act)
        trunk: list[nn.Module] = []
        d_in = feat_size
        for _ in range(layers):
            trunk.append(nn.Linear(d_in, hidden, bias=False))
            trunk.append(nn.LayerNorm(hidden, eps=1e-3))
            trunk.append(act_cls())
            d_in = hidden
        self.trunk = nn.Sequential(*trunk)

        # One projection head per factor. Small init so early policy is close to
        # uniform (with unimix already helping) — matches PPO's gain=0.01 heads.
        self.heads = nn.ModuleList([nn.Linear(hidden, f) for f in self.factors])
        for h in self.heads:
            nn.init.uniform_(h.weight, -1e-3, 1e-3)
            nn.init.zeros_(h.bias)

    @property
    def onehot_dim(self) -> int:
        """Total one-hot-concat action dim = sum of factors. Used as RSSM's num_actions."""
        return int(sum(self.factors))

    def forward(self, feat: torch.Tensor) -> "MultiDiscreteDist":
        """Return a joint distribution over the MultiDiscrete action.

        Consumers of the joint dist call ``.sample()``, ``.log_prob(action)``,
        ``.entropy()``, ``.mode()`` — semantics match a single OneHotDist but
        over the concatenated one-hot representation.
        """
        h = self.trunk(feat)
        per_factor = [head(h) for head in self.heads]
        return MultiDiscreteDist(per_factor, unimix_ratio=self.unimix_ratio)


class MultiDiscreteDist:
    """Joint distribution over ``MultiDiscrete(factors)`` as concatenated one-hots.

    Underneath: N independent ``OneHotDist``s. All exposed methods operate on
    concatenated one-hot representation of shape ``[..., sum(factors)]``.

    - ``sample()`` -> [..., sum(factors)] one-hot concat, with straight-through
      gradient (inherited from OneHotDist).
    - ``mode()`` -> argmax one-hot concat, straight-through.
    - ``log_prob(action)`` -> [...] scalar. ``action`` shape must be
      [..., sum(factors)] one-hot concat; internal split by factor sizes.
    - ``entropy()`` -> [...] sum of per-factor entropies.
    - ``sample_indices()`` -> [..., N] int64 indices (env-facing). Use in the
      rollout loop; convert via ``indices_to_onehot`` for the RSSM.
    """

    def __init__(self, per_factor_logits: list[torch.Tensor], unimix_ratio: float):
        self._factors = [t.shape[-1] for t in per_factor_logits]
        self._dists = [OneHotDist(logits=l, unimix_ratio=unimix_ratio) for l in per_factor_logits]

    def _split(self, action: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(action, self._factors, dim=-1))

    def sample(self, sample_shape=()) -> torch.Tensor:
        parts = [d.sample(sample_shape) for d in self._dists]
        return torch.cat(parts, dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.cat([d.mode() for d in self._dists], dim=-1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        parts = self._split(action)
        return sum(d.log_prob(a) for d, a in zip(self._dists, parts))

    def entropy(self) -> torch.Tensor:
        return sum(d.entropy() for d in self._dists)


def indices_to_onehot(indices: torch.Tensor, factors: Sequence[int]) -> torch.Tensor:
    """Convert ``[..., N]`` int64 factor indices to ``[..., sum(factors)]`` one-hot concat.

    Used at the boundary between env-facing int actions and RSSM-facing one-hot
    action vectors.
    """
    parts = []
    for i, f in enumerate(factors):
        parts.append(F.one_hot(indices[..., i].long(), num_classes=int(f)).to(indices.dtype if indices.dtype.is_floating_point else torch.float32))
    return torch.cat(parts, dim=-1)


def onehot_to_indices(onehot: torch.Tensor, factors: Sequence[int]) -> torch.Tensor:
    """Inverse of ``indices_to_onehot``: ``[..., sum(factors)]`` -> ``[..., N]`` int64."""
    parts = torch.split(onehot, list(factors), dim=-1)
    return torch.stack([p.argmax(dim=-1) for p in parts], dim=-1)
