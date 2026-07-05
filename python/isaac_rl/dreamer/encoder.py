"""Standalone Dict-obs encoder for Isaac.

Lifted from ``isaac_rl.model.IsaacPolicy.encode`` (see model.py:283-314) as a
standalone ``nn.Module`` so DreamerV3's world model can use it. Same encoder
architecture as the PPO baseline — entity attention + per-stream MLPs + room
CNN + trunk MLP — so ablations can compare Dreamer vs PPO with the encoder
held constant.

The z-latent branch (per-episode Gaussian, model.py:308-312) is dropped:
Dreamer's own stochastic RSSM state is unrelated and would collide.
The last_action stream is also dropped — Dreamer feeds the previous action
into the RSSM directly, and doubling it up would just teach the encoder to
memorize an env-side field.

Input: flat obs dict per ``spaces.flatten_dict_obs``. Callable on either
[B, ...] tensors or [T, B, ...] (Dreamer needs the latter for sequence
training). We take whatever leading dims and process the trailing shape.

Output: [..., embed_dim] where embed_dim is the trunk width.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

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
class EncoderConfig:
    """Sized for a single 8-16 GB GPU. Total encoder params ~5M.

    These defaults mirror the PPO baseline (model.py:57-88) with the z-latent
    and last_action streams removed. If you push to bigger hardware, scale
    ``entity_dim``, ``trunk_dim``, and ``embed_dim`` proportionally.
    """
    entity_dim: int = 192
    proj_dim: int = 192
    pickup_dim: int = 96
    grid_channels: tuple = (48, 96)
    trunk_dim: int = 768
    n_attn_heads: int = 4
    n_enemy_attn_layers: int = 2
    n_proj_attn_layers: int = 1
    # Final linear projection to Dreamer's embed dim. Set equal to trunk_dim
    # to skip the extra projection.
    embed_dim: int = 1024


def _mlp(sizes: list[int], activate_final: bool = False, layer_norm: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or activate_final:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MaskedSelfAttention(nn.Module):
    """Batched masked self-attention over a set of entity tokens.

    Copied verbatim from model.py:129-170. Handles zero-real-token rows
    (turns off padding mask on those rows so softmax doesn't NaN, then
    zeros the pooled output).
    """
    def __init__(self, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 2,
                dropout=0.0,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D], mask: [B, N] with 1 for real tokens. Return [B, D] masked mean."""
        pad_mask = mask < 0.5
        row_has_any = mask.sum(-1) > 0.5
        safe_pad_mask = pad_mask.clone()
        safe_pad_mask[~row_has_any] = False
        h = x
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=safe_pad_mask)
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        pooled = pooled * row_has_any.unsqueeze(-1).float()
        return pooled


class IsaacObsEncoder(nn.Module):
    """Dict-obs encoder producing a fixed [B, embed_dim] latent.

    Consumes the flat obs dict from ``spaces.flatten_dict_obs``. Ignores
    ``last_action`` (Dreamer feeds action into the RSSM directly) and ``z``
    (naming collision with Dreamer's stochastic state).

    Callable on [B, ...] or [T, B, ...] — we flatten leading dims, encode,
    then restore the leading dims.
    """

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        c = self.cfg

        self.player_mlp = _mlp([PLAYER_DIM, 192, 192], activate_final=True)
        self.global_mlp = _mlp([GLOBAL_DIM, 96, 96], activate_final=True)
        self.passives_mlp = _mlp([PASSIVES_K, 96], activate_final=True)

        self.enemy_encoder = _mlp([ENEMY_FEATS, c.entity_dim, c.entity_dim], activate_final=True)
        self.enemy_attn = MaskedSelfAttention(c.entity_dim, c.n_attn_heads, c.n_enemy_attn_layers)

        self.proj_encoder = _mlp([PROJ_FEATS, c.proj_dim, c.proj_dim], activate_final=True)
        self.proj_attn = MaskedSelfAttention(c.proj_dim, c.n_attn_heads, c.n_proj_attn_layers)

        self.pickup_encoder = _mlp([PICKUP_FEATS, c.pickup_dim, c.pickup_dim], activate_final=True)

        gc1, gc2 = c.grid_channels
        self.grid_conv = nn.Sequential(
            nn.Conv2d(4, gc1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(gc1, gc2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.doors_mlp = _mlp([4 * 6, 48], activate_final=True)
        self.spatial_mlp = _mlp([SPATIAL_DIM, 48, 48], activate_final=True)
        self.player_history_mlp = _mlp([PLAYER_HISTORY_DIM, 48, 48], activate_final=True)

        # Sum of per-stream output dims.
        static_dims = 192 + 96 + 96 + 48 + 48 + 48   # player, global, passives, doors, spatial, history
        trunk_in = static_dims + c.entity_dim + c.proj_dim + c.pickup_dim + gc2
        self.trunk = _mlp([trunk_in, c.trunk_dim, c.trunk_dim], activate_final=True)

        # Final projection to Dreamer's embed dim. Identity when equal.
        if c.embed_dim == c.trunk_dim:
            self.embed_proj = nn.Identity()
        else:
            self.embed_proj = nn.Linear(c.trunk_dim, c.embed_dim)

        _orthogonal_init(self, gain=math.sqrt(2))

    @property
    def outdim(self) -> int:
        """Match the NM512 MultiEncoder API (``outdim`` attr) for compatibility."""
        return self.cfg.embed_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a flat obs dict. Accepts arbitrary leading batch dims.

        Every obs tensor must share the same leading batch shape. For [T, B, ...]
        we flatten to [T*B, ...], encode, then reshape back.
        """
        # Peel leading dims off the reference key. All obs tensors share these.
        ref = obs["player"]
        lead = ref.shape[:-1]                                  # e.g. (T, B) or (B,)

        def _flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, *x.shape[len(lead):])

        player   = self.player_mlp(_flat(obs["player"]))
        global_  = self.global_mlp(_flat(obs["global"]))
        passives = self.passives_mlp(_flat(obs["passives"]))

        e_tokens = self.enemy_encoder(_flat(obs["enemies_feats"]))
        enemies  = self.enemy_attn(e_tokens, _flat(obs["enemies_mask"]))

        p_tokens = self.proj_encoder(_flat(obs["projectiles_feats"]))
        projs    = self.proj_attn(p_tokens, _flat(obs["projectiles_mask"]))

        k_tokens = self.pickup_encoder(_flat(obs["pickups_feats"]))
        m = _flat(obs["pickups_mask"]).unsqueeze(-1)
        pickups  = (k_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        grid = self.grid_conv(_flat(obs["room_grid"]))        # [N, gc2, H, W]
        grid = grid.mean(dim=(-1, -2))                         # GAP -> [N, gc2]

        doors = self.doors_mlp(_flat(obs["doors"]).flatten(1))
        spatial = self.spatial_mlp(_flat(obs["spatial"]))
        player_hist = self.player_history_mlp(_flat(obs["player_history"]))

        x = torch.cat(
            [player, global_, passives, enemies, projs, pickups, grid, doors, spatial, player_hist],
            dim=-1,
        )
        embed = self.embed_proj(self.trunk(x))
        # Restore original leading dims.
        return embed.reshape(*lead, embed.shape[-1])
