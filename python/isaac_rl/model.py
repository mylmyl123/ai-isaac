"""Actor-critic policy network for Isaac RL.

Architecture (plan §6):

  player, global, passives, last_action, spatial  ─► MLPs ─┐
  enemies      ─► per-row MLP → masked self-attn → mean pool ─┤
  projectiles  ─► per-row MLP → masked self-attn → mean pool  ├─► concat → MLP → GRU ─► factored heads + value
  pickups      ─► per-row MLP → mean pool                     │
  room_grid    ─► Conv → GAP                                  │
  doors        ─► MLP                                         ┘

Factored MultiDiscrete action space: [9, 5].

Implementation details (2026-07-02, following Andrychowicz 2020, Engstrom 2020,
OpenAI Five, AlphaStar):

1. Orthogonal weight initialization with specific gain scaling:
   - Hidden layers: gain=sqrt(2)  (Kaiming-normal-equivalent for ReLU)
   - Policy heads: gain=0.01      (near-uniform action distribution at init;
                                    critical for preserving BC weights during
                                    early PPO updates)
   - Value head:   gain=1.0        (standard scaling for regression heads)

2. LayerNorm on trunk and after GRU:
   Stabilizes activations across long rollouts. Without it, recurrent PPO can
   drift into activation-magnitude runaway, causing gradient explosion or
   policy collapse. Standard in AlphaStar / DreamerV3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spaces import (
    ACTION_FACTORS,
    ENEMY_FEATS,
    GLOBAL_DIM,
    MAX_ENEMIES,
    MAX_PICKUPS,
    MAX_PROJECTILES,
    PASSIVES_K,
    PICKUP_FEATS,
    PLAYER_DIM,
    PROJ_FEATS,
    ROOM_H,
    ROOM_W,
    SPATIAL_DIM,
)


@dataclass
class PolicyConfig:
    entity_dim: int = 128
    proj_dim: int = 128
    pickup_dim: int = 64
    grid_channels: tuple = (32, 64)
    trunk_dim: int = 512
    gru_dim: int = 512
    n_attn_heads: int = 2
    n_enemy_attn_layers: int = 2
    n_proj_attn_layers: int = 1


def _mlp(sizes: list[int], activate_final: bool = False, layer_norm: bool = False) -> nn.Sequential:
    """MLP with optional LayerNorm before each non-final ReLU.

    LayerNorm placement follows the 'pre-norm' convention: LN -> Linear -> ReLU.
    We apply LN before the FIRST hidden layer and between subsequent layers,
    but not on the output layer (so the caller can attach heads without
    duplicating normalization).
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or activate_final:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    """Apply orthogonal init to every Linear/Conv2d in `module` with given gain.

    Biases zeroed. Skips modules with no weight (LayerNorm, ReLU, etc.).
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRUCell):
            # PyTorch's default GRU init is uniform; orthogonal weight_ih and
            # weight_hh works much better in practice for on-policy RL.
            nn.init.orthogonal_(m.weight_ih, gain=gain)
            nn.init.orthogonal_(m.weight_hh, gain=gain)
            if m.bias_ih is not None:
                nn.init.zeros_(m.bias_ih)
            if m.bias_hh is not None:
                nn.init.zeros_(m.bias_hh)


class MaskedSelfAttention(nn.Module):
    """Batched masked self-attention over a set of entity tokens."""

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
        """x: [B, N, D], mask: [B, N] with 1 for real tokens.

        Return pooled [B, D] (masked mean).
        """
        # TransformerEncoderLayer expects a boolean src_key_padding_mask
        # where True = "ignore this position".
        pad_mask = mask < 0.5   # [B, N]

        # If a batch element has zero real tokens, self-attn softmax turns into NaNs.
        # Fix by unsetting the padding mask on such rows (their pooled output will still be 0).
        row_has_any = mask.sum(-1) > 0.5
        safe_pad_mask = pad_mask.clone()
        safe_pad_mask[~row_has_any] = False

        h = x
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=safe_pad_mask)

        # Masked mean pool.
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        pooled = pooled * row_has_any.unsqueeze(-1).float()   # zero out empty rows
        return pooled


class IsaacPolicy(nn.Module):
    """Actor-critic module with per-env recurrent GRU state.

    step_forward()  — one step, keeps hidden state (rollout).
    forward()       — full sequence for PPO minibatch update.
    """

    action_factors = torch.as_tensor(ACTION_FACTORS.tolist(), dtype=torch.long)

    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        c = self.cfg

        self.player_mlp   = _mlp([PLAYER_DIM, 128, 128], activate_final=True, layer_norm=True)
        self.global_mlp   = _mlp([GLOBAL_DIM, 64, 64], activate_final=True, layer_norm=True)
        self.passives_mlp = _mlp([PASSIVES_K, 64], activate_final=True, layer_norm=True)
        self.last_action_mlp = _mlp([len(ACTION_FACTORS), 32], activate_final=True, layer_norm=True)

        self.enemy_encoder = _mlp([ENEMY_FEATS, c.entity_dim, c.entity_dim], activate_final=True, layer_norm=True)
        self.enemy_attn = MaskedSelfAttention(c.entity_dim, c.n_attn_heads, c.n_enemy_attn_layers)

        self.proj_encoder = _mlp([PROJ_FEATS, c.proj_dim, c.proj_dim], activate_final=True, layer_norm=True)
        self.proj_attn = MaskedSelfAttention(c.proj_dim, c.n_attn_heads, c.n_proj_attn_layers)

        self.pickup_encoder = _mlp([PICKUP_FEATS, c.pickup_dim, c.pickup_dim], activate_final=True, layer_norm=True)

        gc1, gc2 = c.grid_channels
        self.grid_conv = nn.Sequential(
            nn.Conv2d(4, gc1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(gc1, gc2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.doors_mlp = _mlp([4 * 6, 32], activate_final=True, layer_norm=True)
        # Spatial features MLP (schema v2). Small dedicated pathway for
        # preprocessed room-position/wall-distance/door-direction features.
        # Feeding these as first-class inputs is far more sample-efficient
        # than making the network learn spatial reasoning from raw pixels.
        self.spatial_mlp = _mlp([SPATIAL_DIM, 32, 32], activate_final=True, layer_norm=True)

        trunk_in = 128 + 64 + 64 + 32 + c.entity_dim + c.proj_dim + c.pickup_dim + gc2 + 32 + 32
        self.trunk = _mlp([trunk_in, c.trunk_dim, c.trunk_dim], activate_final=True, layer_norm=True)

        self.gru = nn.GRUCell(c.trunk_dim, c.gru_dim)
        # Post-GRU LayerNorm: stabilises hidden state across rollout ticks.
        # Crucial for long-horizon recurrent PPO — without it, GRU hidden states
        # can drift in magnitude across the 256-tick rollouts, causing gradient
        # explosion during backprop.
        self.gru_ln = nn.LayerNorm(c.gru_dim)

        # Factored action heads.
        self.heads = nn.ModuleList([nn.Linear(c.gru_dim, n) for n in ACTION_FACTORS.tolist()])
        self.value_head = nn.Linear(c.gru_dim, 1)

        # Auxiliary regression heads (see aux_labels_from_obs in ppo.py). Each
        # forces the trunk to encode a summary statistic of the current obs as
        # a first-class feature. Style: UNREAL (Jaderberg et al. 2017) but with
        # obs-derived targets instead of pixel-control.
        # Targets (3 scalars):
        #   [0] nearest-enemy normalised distance
        #   [1] enemy count normalised by MAX_ENEMIES
        #   [2] nearest-projectile normalised distance
        self.aux_head = nn.Linear(c.gru_dim, 3)

        # ---- Weight initialization (see module docstring) ------------------
        # Hidden layers (everything except heads): orthogonal, gain=sqrt(2).
        # This is the "Kaiming" equivalent for ReLU under orthogonal init and
        # is the standard for PPO (Engstrom 2020, Andrychowicz 2020).
        _orthogonal_init(self, gain=math.sqrt(2))
        # Then override the heads with proper scales:
        # - Policy heads: gain=0.01 makes the initial action distribution
        #   nearly uniform, which is critical when starting from BC-pretrained
        #   weights (large policy logits would immediately drift the policy).
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        # - Value head: gain=1.0 keeps standard scaling for regression.
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        # - Aux head: gain=1.0 like value head (also regression).
        nn.init.orthogonal_(self.aux_head.weight, gain=1.0)
        nn.init.zeros_(self.aux_head.bias)

    # -- encoders -----------------------------------------------------------

    def encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a flat batch of observations into a trunk feature vector [B, trunk_dim]."""
        player   = self.player_mlp(obs["player"])
        global_  = self.global_mlp(obs["global"])
        passives = self.passives_mlp(obs["passives"])
        last_act = self.last_action_mlp(obs["last_action"])

        e_tokens = self.enemy_encoder(obs["enemies_feats"])
        enemies  = self.enemy_attn(e_tokens, obs["enemies_mask"])

        p_tokens = self.proj_encoder(obs["projectiles_feats"])
        projs    = self.proj_attn(p_tokens, obs["projectiles_mask"])

        k_tokens = self.pickup_encoder(obs["pickups_feats"])
        m = obs["pickups_mask"].unsqueeze(-1)
        pickups  = (k_tokens * m).sum(1) / m.sum(1).clamp(min=1.0)

        grid = self.grid_conv(obs["room_grid"])        # [B, gc2, H, W]
        grid = grid.mean(dim=(-1, -2))                  # GAP -> [B, gc2]

        doors = self.doors_mlp(obs["doors"].flatten(1))
        spatial = self.spatial_mlp(obs["spatial"])

        x = torch.cat([player, global_, passives, last_act, enemies, projs, pickups, grid, doors, spatial], dim=-1)
        return self.trunk(x)

    # -- rollout step -------------------------------------------------------

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.cfg.gru_dim, device=device)

    def step(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        done_mask: torch.Tensor | None = None,
    ):
        """One rollout step. Return (logits_list, value, new_hidden).

        `done_mask` [B]: 1.0 where the previous step ended an episode (reset hidden).
        """
        if done_mask is not None:
            hidden = hidden * (1.0 - done_mask.unsqueeze(-1))
        h = self.encode(obs)
        new_hidden = self.gru(h, hidden)
        new_hidden_ln = self.gru_ln(new_hidden)
        logits = [head(new_hidden_ln) for head in self.heads]
        value = self.value_head(new_hidden_ln).squeeze(-1)
        # Return the un-normalized hidden state so the GRU sees consistent
        # state across steps (LayerNorm is applied only for the readouts).
        return logits, value, new_hidden

    def aux_predict(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return auxiliary regression predictions from a hidden state.

        Shape: [B, 3] — (nearest_enemy_dist, enemy_count, nearest_proj_dist),
        all normalised. See aux_labels_from_obs() in ppo.py for the targets.
        """
        return self.aux_head(self.gru_ln(hidden))

    # -- sequence forward (PPO update) --------------------------------------

    def sequence_forward(
        self,
        seq_obs: dict[str, torch.Tensor],
        seq_dones: torch.Tensor,
        init_hidden: torch.Tensor,
    ):
        """Roll the policy over a T×B sequence.

        seq_obs values have shape [T, B, ...].
        seq_dones: [T, B] float, 1.0 where the episode ended at that step (mask BEFORE step).
        init_hidden: [B, gru_dim] hidden state at t=0.
        Return: (logits_list [T*B, n_i], values [T*B], entropies_list [T*B, n_i])  — flattened.
        """
        T = seq_dones.shape[0]
        B = seq_dones.shape[1]
        h = init_hidden

        # Encode each timestep separately because GRUCell is per-step and
        # we need to inject done-masks between steps.
        step_hiddens = []
        for t in range(T):
            step_obs = {k: v[t] for k, v in seq_obs.items()}
            enc = self.encode(step_obs)
            h = h * (1.0 - seq_dones[t].unsqueeze(-1))
            h = self.gru(enc, h)
            step_hiddens.append(h)
        hs = torch.stack(step_hiddens, dim=0)            # [T, B, D]
        flat = hs.view(T * B, -1)
        flat_ln = self.gru_ln(flat)

        logits = [head(flat_ln) for head in self.heads]
        values = self.value_head(flat_ln).squeeze(-1)
        aux = self.aux_head(flat_ln)
        return logits, values, aux

    # -- action distribution helpers ----------------------------------------

    @staticmethod
    def log_prob_from_logits(logits_list, actions: torch.Tensor) -> torch.Tensor:
        """Sum of per-factor log-probs. actions: [B, n_factors]."""
        total = torch.zeros(actions.shape[0], device=actions.device)
        for i, logits in enumerate(logits_list):
            lp = F.log_softmax(logits, dim=-1)
            total = total + lp.gather(1, actions[:, i:i + 1]).squeeze(-1)
        return total

    @staticmethod
    def entropy_from_logits(logits_list) -> torch.Tensor:
        """Sum of per-factor entropies."""
        total = torch.zeros(logits_list[0].shape[0], device=logits_list[0].device)
        for logits in logits_list:
            p = F.softmax(logits, dim=-1)
            lp = F.log_softmax(logits, dim=-1)
            total = total - (p * lp).sum(-1)
        return total

    @staticmethod
    def sample_from_logits(logits_list, greedy: bool = False) -> torch.Tensor:
        """Return [B, n_factors] sampled action."""
        parts = []
        for logits in logits_list:
            if greedy:
                a = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                a = torch.multinomial(probs, 1).squeeze(-1)
            parts.append(a)
        return torch.stack(parts, dim=-1)
