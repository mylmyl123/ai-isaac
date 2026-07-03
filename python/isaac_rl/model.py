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
    PLAYER_HISTORY_DIM,
    PROJ_FEATS,
    ROOM_H,
    ROOM_W,
    SPATIAL_DIM,
)


@dataclass
class PolicyConfig:
    # Network capacity scaled up (2026-07-02) to take advantage of GPU compute.
    # Previous values were tuned for a compute-limited baseline; on any modern
    # GPU (RTX 3060 Ti or better) these larger dimensions run at similar
    # wall-clock speed but represent the input better. Total params: ~8M (was ~2M).
    entity_dim: int = 192          # was 128; more capacity for enemy encoding
    proj_dim: int = 192            # was 128; more capacity for projectile encoding
    pickup_dim: int = 96           # was 64
    grid_channels: tuple = (48, 96)   # was (32, 64); ~2.25x conv capacity
    trunk_dim: int = 768           # was 512
    gru_dim: int = 1024            # was 512; recurrent memory is the most
                                   # important width for our recurrent PPO
    n_attn_heads: int = 4          # was 2; more attention heads
    n_enemy_attn_layers: int = 2
    n_proj_attn_layers: int = 1
    # B3: Predict-future-rewards aux task horizon (number of future ticks
    # to predict). N=8 covers ~130ms of game time at 60Hz — enough for
    # short-term consequence modeling.
    reward_pred_horizon: int = 8
    # B4: Latent variable conditioning. Per-episode z ~ N(0, I) is sampled
    # at reset and fed to the policy as additional obs. Encourages strategic
    # diversity (aggressive/defensive/exploratory play styles emerge across
    # episodes). Set z_dim=0 to disable.
    z_dim: int = 16
    # B1: Distributional value function (DreamerV3-style twohot categorical).
    # Instead of predicting a scalar E[V(s)], predict a categorical distribution
    # over N atoms. Reduces value function variance dramatically.
    # Set value_atoms=1 to fall back to scalar value.
    value_atoms: int = 51        # 51 atoms = C51-standard
    value_v_min: float = -20.0   # min return (in symlog space if using symlog_rewards)
    value_v_max: float = 20.0    # max return


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

        self.player_mlp   = _mlp([PLAYER_DIM, 192, 192], activate_final=True, layer_norm=True)
        self.global_mlp   = _mlp([GLOBAL_DIM, 96, 96], activate_final=True, layer_norm=True)
        self.passives_mlp = _mlp([PASSIVES_K, 96], activate_final=True, layer_norm=True)
        self.last_action_mlp = _mlp([len(ACTION_FACTORS), 48], activate_final=True, layer_norm=True)

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
        self.doors_mlp = _mlp([4 * 6, 48], activate_final=True, layer_norm=True)
        # Spatial features MLP (schema v2). Small dedicated pathway for
        # preprocessed room-position/wall-distance/door-direction features.
        # Feeding these as first-class inputs is far more sample-efficient
        # than making the network learn spatial reasoning from raw pixels.
        self.spatial_mlp = _mlp([SPATIAL_DIM, 48, 48], activate_final=True, layer_norm=True)
        # Player history MLP (frame stacking, 2026-07-02). Consumes the last
        # N frames of (nx, ny, vx, vy) to give the network explicit access
        # to short-term dynamics. Redundant with GRU but helps early BC.
        self.player_history_mlp = _mlp([PLAYER_HISTORY_DIM, 48, 48], activate_final=True, layer_norm=True)
        # B4: Latent variable z MLP. Small dedicated pathway for the
        # per-episode strategic latent. Zero-dim -> disabled.
        self.z_mlp = _mlp([c.z_dim, 32, 32], activate_final=True, layer_norm=True) if c.z_dim > 0 else None
        z_out_dim = 32 if c.z_dim > 0 else 0

        # trunk_in is computed from the actual per-stream output sizes rather
        # than hardcoded constants so future capacity changes stay consistent.
        static_dims = 192 + 96 + 96 + 48 + 48 + 48 + 48 + z_out_dim  # +z_out_dim if z enabled
        trunk_in = static_dims + c.entity_dim + c.proj_dim + c.pickup_dim + gc2
        self.trunk = _mlp([trunk_in, c.trunk_dim, c.trunk_dim], activate_final=True, layer_norm=True)

        self.gru = nn.GRUCell(c.trunk_dim, c.gru_dim)
        # Post-GRU LayerNorm: stabilises hidden state across rollout ticks.
        # Crucial for long-horizon recurrent PPO — without it, GRU hidden states
        # can drift in magnitude across the 256-tick rollouts, causing gradient
        # explosion during backprop.
        self.gru_ln = nn.LayerNorm(c.gru_dim)

        # Factored action heads.
        self.heads = nn.ModuleList([nn.Linear(c.gru_dim, n) for n in ACTION_FACTORS.tolist()])
        # B1: Distributional value head. Outputs `value_atoms` logits;
        # softmax -> categorical distribution over returns.
        # value_atoms=1 falls back to scalar value (backward-compat).
        self.value_head = nn.Linear(c.gru_dim, c.value_atoms)
        # Support atoms (registered as buffer so .to(device) transfers them).
        support = torch.linspace(c.value_v_min, c.value_v_max, c.value_atoms)
        self.register_buffer("value_support", support, persistent=False)

        # Auxiliary regression heads (see aux_labels_from_obs in ppo.py). Each
        # forces the trunk to encode a summary statistic of the current obs as
        # a first-class feature. Style: UNREAL (Jaderberg et al. 2017) but with
        # obs-derived targets instead of pixel-control.
        # Targets (3 scalars):
        #   [0] nearest-enemy normalised distance
        #   [1] enemy count normalised by MAX_ENEMIES
        #   [2] nearest-projectile normalised distance
        self.aux_head = nn.Linear(c.gru_dim, 3)

        # B3: Predict-future-rewards aux head (UNREAL-style, Jaderberg 2017).
        # Predicts the next N tick rewards from the current hidden state.
        # Forces the trunk to be PREDICTIVE (not just descriptive), which
        # accelerates value function convergence.
        # N = c.reward_pred_horizon (default 8). Output shape [B, N].
        self.reward_pred_head = nn.Linear(c.gru_dim, c.reward_pred_horizon)

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
        # Also init the new heads.
        nn.init.orthogonal_(self.aux_head.weight, gain=1.0)
        nn.init.zeros_(self.aux_head.bias)
        nn.init.orthogonal_(self.reward_pred_head.weight, gain=1.0)
        nn.init.zeros_(self.reward_pred_head.bias)

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
        player_hist = self.player_history_mlp(obs["player_history"])

        streams = [player, global_, passives, last_act, enemies, projs, pickups, grid, doors, spatial, player_hist]
        if self.z_mlp is not None:
            z = obs.get("z")
            if z is None:
                z = torch.zeros(player.shape[0], self.cfg.z_dim, device=player.device)
            streams.append(self.z_mlp(z))
        x = torch.cat(streams, dim=-1)
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
        value = self._value_from_head(self.value_head(new_hidden_ln))
        # Return the un-normalized hidden state so the GRU sees consistent
        # state across steps (LayerNorm is applied only for the readouts).
        return logits, value, new_hidden

    # B1: distributional value helpers -----------------------------------

    def _value_from_head(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert value_head output to a scalar E[V].

        If value_atoms == 1: pass-through (scalar mode, backward-compat).
        Else: softmax + expected value over the support atoms.
        """
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.value_support).sum(-1)

    def value_twohot_target(self, returns: torch.Tensor) -> torch.Tensor:
        """DreamerV3-style twohot encoding of scalar returns.

        Given returns of arbitrary shape [...] produce target distribution
        [..., value_atoms] with prob mass on the two atoms nearest to the
        return, split proportionally to distance.
        """
        support = self.value_support
        n_atoms = support.shape[0]
        if n_atoms == 1:
            return returns.unsqueeze(-1)
        r = returns.clamp(support[0], support[-1])
        delta = (support[-1] - support[0]) / (n_atoms - 1)
        idx_float = (r - support[0]) / delta
        idx_lo = idx_float.floor().long().clamp(0, n_atoms - 1)
        idx_hi = (idx_lo + 1).clamp(0, n_atoms - 1)
        frac_hi = idx_float - idx_lo.float()
        frac_lo = 1.0 - frac_hi
        target = torch.zeros(*r.shape, n_atoms, device=r.device)
        target.scatter_add_(-1, idx_lo.unsqueeze(-1), frac_lo.unsqueeze(-1))
        target.scatter_add_(-1, idx_hi.unsqueeze(-1), frac_hi.unsqueeze(-1))
        return target

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
        # B1: distributional value. Returns SCALAR expected value for GAE;
        # for the loss, the trainer accesses .value_head_logits() separately.
        value_logits = self.value_head(flat_ln)
        values = self._value_from_head(value_logits)
        aux = self.aux_head(flat_ln)
        reward_pred = self.reward_pred_head(flat_ln)
        return logits, values, aux, reward_pred, value_logits

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
