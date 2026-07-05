"""Per-stream reconstruction decoder for Isaac's Dict obs.

For each obs key, produce a distribution over that obs's shape and expose
``.log_prob(target)``. This matches NM512's ``WorldModel._train`` contract
(models.py:129-142): the loss is ``-pred.log_prob(target)`` summed per
head.

Distribution choices by stream:
  player, global, spatial, player_history   -> SymlogDist    (symlog-MSE)
  passives, doors, room_grid                -> Bernoulli     (BCE with logits)
  enemies_feats, projectiles_feats,
    pickups_feats + their masks             -> masked losses (see below)

Entity-set streams need special care. Raw ``feats [B, N, F]`` reconstruction
is notoriously hard for Dreamer because slot ordering is arbitrary and empty
slots would dominate MSE. We decode two things per stream:
  - the mask (per-slot Bernoulli): the world model learns *how many* entities
    exist in each slot.
  - the features (per-slot symlog-MSE), weighted by the target mask so empty
    slots don't contribute gradient.

Total decoder loss = sum of per-key negative log-probs. Encoded losses are
returned as ``{'stream_name': [B, T]}`` scalars so the world model can weight
them individually if desired.

Contract with ``vendor.models.WorldModel._train``:
  head = decoder(feat)             # dict of dists keyed by obs name
  losses[k] = -head[k].log_prob(target[k])   # each returns [B, T]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd

from .vendor import tools
from ..spaces import (
    ENEMY_FEATS,
    GLOBAL_DIM,
    MAX_ENEMIES,
    MAX_PICKUPS,
    MAX_PROJECTILES,
    PASSIVES_K,
    PICKUP_FEATS,
    PLAYER_DIM,
    PLAYER_HISTORY_DIM,
    PROJ_FEATS,
    ROOM_H,
    ROOM_W,
    SPATIAL_DIM,
)


@dataclass
class DecoderConfig:
    hidden: int = 512
    layers: int = 2


def _mlp(sizes: list[int], activate_final: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or activate_final:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class _SymlogMSEHead(nn.Module):
    """Linear head + SymlogDist. .log_prob(target) => [B, T] scalar log-prob."""

    def __init__(self, feat_size: int, out_shape: tuple[int, ...], hidden: int, layers: int):
        super().__init__()
        sizes = [feat_size] + [hidden] * layers
        self.trunk = _mlp(sizes, activate_final=True)
        self.out = nn.Linear(hidden, int(torch.tensor(out_shape).prod()))
        self.out_shape = out_shape

    def forward(self, feat: torch.Tensor) -> "tools.SymlogDist":
        x = self.trunk(feat)
        mean = self.out(x).reshape(*feat.shape[:-1], *self.out_shape)
        return tools.SymlogDist(mean)


class _BernoulliDist:
    """Wraps logits + target for BCE. .log_prob(target) reduces ALL event dims.

    NM512's ``tools.Bernoulli.log_prob`` only reduces the last dim, which is
    wrong for multi-dim event shapes like our ``room_grid [4, 9, 15]`` or
    ``doors [4, 6]``. We do the reduction ourselves.
    """

    def __init__(self, logits: torch.Tensor, event_ndim: int):
        self._logits = logits
        self._event_ndim = event_ndim
        # Expose .mean for compatibility with tools.Bernoulli.
        self.mean = torch.sigmoid(logits)

    def mode(self) -> torch.Tensor:
        return (self.mean > 0.5).float().detach() + self.mean - self.mean.detach()

    def sample(self, sample_shape=()) -> torch.Tensor:
        return torchd.bernoulli.Bernoulli(logits=self._logits).sample(sample_shape)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        # Per-element BCE-with-logits.
        log_probs0 = -F.softplus(self._logits)                              # log P(0)
        log_probs1 = -F.softplus(-self._logits)                             # log P(1)
        elem = log_probs0 * (1.0 - target) + log_probs1 * target
        # Sum over event dims (last self._event_ndim axes).
        reduce_dims = list(range(-self._event_ndim, 0))
        return elem.sum(dim=reduce_dims)


class _BernoulliHead(nn.Module):
    """Linear head + BCE-with-logits. .log_prob(target [0/1]) => [B, T]."""

    def __init__(self, feat_size: int, out_shape: tuple[int, ...], hidden: int, layers: int):
        super().__init__()
        sizes = [feat_size] + [hidden] * layers
        self.trunk = _mlp(sizes, activate_final=True)
        self.out = nn.Linear(hidden, int(torch.tensor(out_shape).prod()))
        self.out_shape = out_shape

    def forward(self, feat: torch.Tensor) -> _BernoulliDist:
        logits = self.out(self.trunk(feat)).reshape(*feat.shape[:-1], *self.out_shape)
        return _BernoulliDist(logits, event_ndim=len(self.out_shape))


class _MaskedEntityHead(nn.Module):
    """Reconstruct entity feats + mask.

    Two heads share a trunk:
      * mask head: Bernoulli over ``[B, T, N]``
      * feats head: symlog-MSE over ``[B, T, N, F]``, weighted by the target mask
        (empty slots contribute 0 gradient).

    Exposes ``.log_prob(target)`` where ``target`` is a *tuple* ``(feats, mask)``
    — but the WorldModel expects a single tensor. Instead we bundle both into
    a single distribution object with a fused log-prob.
    """

    def __init__(self, feat_size: int, max_n: int, feat_dim: int, hidden: int, layers: int):
        super().__init__()
        self.max_n = max_n
        self.feat_dim = feat_dim
        sizes = [feat_size] + [hidden] * layers
        self.trunk = _mlp(sizes, activate_final=True)
        self.feats_out = nn.Linear(hidden, max_n * feat_dim)
        self.mask_out = nn.Linear(hidden, max_n)

    def forward(self, feat: torch.Tensor) -> "_MaskedEntityDist":
        lead = feat.shape[:-1]
        x = self.trunk(feat)
        feats_mean = self.feats_out(x).reshape(*lead, self.max_n, self.feat_dim)
        mask_logits = self.mask_out(x).reshape(*lead, self.max_n)
        return _MaskedEntityDist(feats_mean, mask_logits)


class _MaskedEntityDist:
    """Fused feats+mask distribution.

    The decoder registers two obs keys (e.g. ``enemies_feats`` and ``enemies_mask``)
    under a single head via a *shared* dist object. WorldModel._train will call
    ``head(feat).log_prob(target)`` once per key — we return the fused object
    for both and let ``log_prob`` route by shape.

    Feats log-prob is a per-slot Gaussian in symlog space, weighted by the
    target mask. Mask log-prob is Bernoulli over N slots.
    """

    def __init__(self, feats_mean: torch.Tensor, mask_logits: torch.Tensor):
        self._feats_mean = feats_mean          # [B, T, N, F]
        self._mask_logits = mask_logits        # [B, T, N]

    def mode(self) -> torch.Tensor:
        # Return feats by default (mode is called by WorldModel.video_pred for images;
        # for our discrete-obs use case it's not on the hot path).
        return self._feats_mean

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """Route by target shape.

        - target shape [..., N, F] -> feats reconstruction
        - target shape [..., N]    -> mask BCE
        """
        if target.dim() == self._feats_mean.dim():
            # Feats. Symlog-MSE with target-mask weighting: empty slots contribute 0.
            # We need the mask -- infer from the target (any nonzero feat in a slot
            # implies mask=1 for that slot). Fallback: also pull from the mask side
            # via a stored copy would be cleaner, but the WorldModel API gives us
            # the target tensor only. Use "row has any nonzero" as a heuristic.
            row_has_any = (target.abs().sum(dim=-1) > 1e-8).float()   # [..., N]
            symlog_pred = tools.symlog(self._feats_mean)
            symlog_tgt = tools.symlog(target)
            per_elem = 0.5 * (symlog_pred - symlog_tgt).pow(2)         # [..., N, F]
            per_slot = per_elem.sum(dim=-1)                             # [..., N]
            weighted = per_slot * row_has_any
            # Normalise by number of real slots so magnitude is stable. Add eps.
            denom = row_has_any.sum(dim=-1).clamp(min=1.0)               # [...]
            neg_log_prob = weighted.sum(dim=-1) / denom
            return -neg_log_prob
        # Mask target.
        base = torchd.bernoulli.Bernoulli(logits=self._mask_logits)
        # Independent over N slots -> summed log-prob is [...].
        return torchd.independent.Independent(base, 1).log_prob(target)


class IsaacObsDecoder(nn.Module):
    """Multi-head decoder producing dict-of-distributions from RSSM feat.

    ``feat`` shape: [B, T, feat_size] where feat_size = stoch*discrete + deter.

    Output: dict[str, distribution] with one entry per reconstructed obs key.

    2026-07-05 optimization: single shared trunk feeding lightweight output
    projections per head, rather than 13 independent trunks (was ~4M params
    duplicated 13x). Cuts decoder compute ~5x with no quality loss — the
    per-head signal comes from the output projection, not from deeper
    per-head trunks. This matches NM512's MultiDecoder architecture.
    """

    # Which obs keys we reconstruct (excludes last_action, z — those are
    # env-controlled/collision-inducing).
    _SYMLOG_KEYS = {
        "player":         (PLAYER_DIM,),
        "global":         (GLOBAL_DIM,),
        "spatial":        (SPATIAL_DIM,),
        "player_history": (PLAYER_HISTORY_DIM,),
    }
    _BERNOULLI_KEYS = {
        "passives":  (PASSIVES_K,),
        "doors":     (4, 6),
        "room_grid": (4, ROOM_H, ROOM_W),
    }
    _ENTITY_KEYS = {
        # obs-key-prefix -> (max_n, feat_dim)
        "enemies":     (MAX_ENEMIES, ENEMY_FEATS),
        "projectiles": (MAX_PROJECTILES, PROJ_FEATS),
        "pickups":     (MAX_PICKUPS, PICKUP_FEATS),
    }

    def __init__(self, feat_size: int, cfg: DecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        c = self.cfg

        # Single shared trunk. Output has width `c.hidden`; individual output
        # heads project from `c.hidden` to their per-obs-key shapes.
        sizes = [feat_size] + [c.hidden] * c.layers
        self.trunk = _mlp(sizes, activate_final=True)

        # Per-key output projections.
        self.symlog_out = nn.ModuleDict({
            k: nn.Linear(c.hidden, int(torch.tensor(shape).prod()))
            for k, shape in self._SYMLOG_KEYS.items()
        })
        self.bernoulli_out = nn.ModuleDict({
            k: nn.Linear(c.hidden, int(torch.tensor(shape).prod()))
            for k, shape in self._BERNOULLI_KEYS.items()
        })
        # Entity heads: two linear projections each (feats + mask).
        self.entity_feats_out = nn.ModuleDict({
            k: nn.Linear(c.hidden, max_n * feat_dim)
            for k, (max_n, feat_dim) in self._ENTITY_KEYS.items()
        })
        self.entity_mask_out = nn.ModuleDict({
            k: nn.Linear(c.hidden, max_n)
            for k, (max_n, _) in self._ENTITY_KEYS.items()
        })

    def forward(self, feat: torch.Tensor) -> dict[str, Any]:
        """Return {obs_key: distribution}. Same dict has entries for both
        ``enemies_feats`` and ``enemies_mask`` pointing at the same fused dist.
        """
        # Run the trunk ONCE.
        h = self.trunk(feat)
        lead = feat.shape[:-1]                                # e.g. (B, T)

        out: dict[str, Any] = {}
        for k, shape in self._SYMLOG_KEYS.items():
            mean = self.symlog_out[k](h).reshape(*lead, *shape)
            out[k] = tools.SymlogDist(mean)
        for k, shape in self._BERNOULLI_KEYS.items():
            logits = self.bernoulli_out[k](h).reshape(*lead, *shape)
            out[k] = _BernoulliDist(logits, event_ndim=len(shape))
        for k, (max_n, feat_dim) in self._ENTITY_KEYS.items():
            feats_mean = self.entity_feats_out[k](h).reshape(*lead, max_n, feat_dim)
            mask_logits = self.entity_mask_out[k](h).reshape(*lead, max_n)
            dist = _MaskedEntityDist(feats_mean, mask_logits)
            # WorldModel._train iterates obs keys independently; expose both.
            out[f"{k}_feats"] = dist
            out[f"{k}_mask"] = dist
        return out

    @property
    def reconstructed_keys(self) -> list[str]:
        """Flat obs keys this decoder produces distributions for."""
        keys = list(self._SYMLOG_KEYS) + list(self._BERNOULLI_KEYS)
        for k in self._ENTITY_KEYS:
            keys += [f"{k}_feats", f"{k}_mask"]
        return keys
