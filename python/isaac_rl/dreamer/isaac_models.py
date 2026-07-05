"""Isaac-specific DreamerV3 world model and imagination behavior.

Composed from vendored building blocks (RSSM, MLP, DiscDist, OneHotDist,
lambda_return, static_scan, Optimizer) + our own encoder/decoder/action head.

Not a subclass of the vendor ``WorldModel`` — the vendor version is
pixel-first (asserts ``obs["image"]``, does /255.0 in preprocess, uses
``MultiEncoder``/``MultiDecoder`` which route by shape). Cleaner to write
our own world model that shares the same structure but speaks our schema.

Public API:
  IsaacWorldModel(cfg)
    .train_step(batch)       -> (post, context, metrics)
      batch: dict of numpy arrays from replay.sample()
    .encode(obs)             -> embed tensor for env-side rollouts
    .obs_step(...)           -> RSSM observe one step (env-side rollouts)
    .img_step(...)           -> RSSM imagination step

  IsaacImagBehavior(cfg, world_model)
    .train_step(start_state) -> metrics
      start_state: batched RSSM state dict from world model
    .actor                   -> MultiDiscreteActionHead
    .critic                  -> value network (MLP with symlog_disc dist)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import MultiDiscreteActionHead
from .config import DreamerConfig
from .decoder import DecoderConfig, IsaacObsDecoder
from .encoder import EncoderConfig, IsaacObsEncoder
from .vendor import networks, tools


def _to_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class IsaacWorldModel(nn.Module):
    """Encoder + RSSM + Decoder + reward head + continue head.

    Loss = -log_prob(reconstruction) + -log_prob(reward) + -log_prob(cont) + KL_loss.

    KL loss combines dynamics-KL (train prior to match posterior, dyn_scale
    weight) and representation-KL (train posterior to match prior, rep_scale
    weight), each clipped at ``kl_free_bits``. Standard DreamerV3 formula.
    """

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self._device = device

        # ---- Encoder ---------------------------------------------------
        enc_cfg = EncoderConfig(
            embed_dim=cfg.encoder_embed_dim,
            trunk_dim=cfg.encoder_trunk_dim,
        )
        self.encoder = IsaacObsEncoder(enc_cfg)

        # ---- RSSM ------------------------------------------------------
        # num_actions is the concat-one-hot dim. Isaac: 9 + 5 = 14.
        # We fix action factors here for the RSSM's shape; the actor uses the
        # same factors from cfg.
        from ..spaces import ACTION_FACTORS
        self._action_factors = tuple(int(x) for x in ACTION_FACTORS.tolist())
        self._num_actions = int(sum(self._action_factors))

        self.dynamics = networks.RSSM(
            stoch=cfg.rssm_stoch,
            deter=cfg.rssm_deter,
            hidden=cfg.rssm_hidden,
            rec_depth=cfg.rssm_rec_depth,
            discrete=cfg.rssm_discrete,
            act=cfg.rssm_act,
            norm=cfg.rssm_norm,
            mean_act=cfg.rssm_mean_act,
            std_act=cfg.rssm_std_act,
            min_std=cfg.rssm_min_std,
            unimix_ratio=cfg.rssm_unimix_ratio,
            initial=cfg.rssm_initial,
            num_actions=self._num_actions,
            embed=self.encoder.outdim,
            device=str(device),
        )

        feat_size = cfg.rssm_stoch * cfg.rssm_discrete + cfg.rssm_deter

        # ---- Decoder ---------------------------------------------------
        dec_cfg = DecoderConfig(hidden=cfg.decoder_hidden, layers=cfg.decoder_layers)
        self.decoder = IsaacObsDecoder(feat_size, dec_cfg)

        # ---- Reward + continue heads (vendor MLP with the right dist) ---
        # symlog_disc = 255-bin twohot over symlog space [-20, 20].
        self.reward_head = networks.MLP(
            feat_size,
            (255,),
            cfg.reward_head_layers,
            cfg.decoder_hidden,
            cfg.rssm_act,
            cfg.rssm_norm,
            dist="symlog_disc",
            outscale=0.0,
            device=str(device),
            name="Reward",
        )
        # Binary continue flag. NM512 uses dist="binary" which needs `shape=()`.
        self.cont_head = networks.MLP(
            feat_size,
            (),
            cfg.cont_head_layers,
            cfg.decoder_hidden,
            cfg.rssm_act,
            cfg.rssm_norm,
            dist="binary",
            outscale=1.0,
            device=str(device),
            name="Cont",
        )

        # ---- Optimizer (vendor Optimizer wraps AMP + grad clip) --------
        # On CPU, torch.cuda.amp.GradScaler is a no-op; safe to instantiate.
        self._opt = tools.Optimizer(
            "model",
            list(self.parameters()),
            cfg.world_model_lr,
            cfg.world_model_eps,
            cfg.world_model_grad_clip,
            cfg.weight_decay,
            opt="adam",
            use_amp=False,
        )

        self.to(device)

    # ------------------------------------------------------------------
    # env-side rollout helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(obs)

    @torch.no_grad()
    def initial_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        return self.dynamics.initial(batch_size)

    @torch.no_grad()
    def obs_step(
        self,
        prev_state: dict[str, torch.Tensor] | None,
        prev_action: torch.Tensor,          # [B, num_actions] one-hot concat
        embed: torch.Tensor,                # [B, embed_dim]
        is_first: torch.Tensor,             # [B] float (0/1)
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        post, prior = self.dynamics.obs_step(prev_state, prev_action, embed, is_first)
        return post, prior

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, np.ndarray]) -> tuple[dict, dict, dict[str, float]]:
        """One WM gradient update on a [B, T] batch from replay.

        Returns (post, context, metrics). ``post`` is the RSSM posterior state
        (batched, with grad detached) — used by ImagBehavior as start states
        for imagination rollouts. ``context`` carries embed/feat/kl for logging.
        """
        cfg = self.cfg
        device = self._device
        # numpy -> torch on device
        obs_t = {k: _to_tensor(v, device) for k, v in batch.items()
                 if k not in ("action", "reward", "is_first", "is_terminal")}
        action = _to_tensor(batch["action"], device)         # [B, T, num_actions]
        reward = _to_tensor(batch["reward"], device)         # [B, T]
        is_first = _to_tensor(batch["is_first"], device)     # [B, T]
        is_terminal = _to_tensor(batch["is_terminal"], device)

        # ---- encode -----------------------------------------------------
        embed = self.encoder(obs_t)                         # [B, T, embed_dim]

        # ---- rssm observe ----------------------------------------------
        post, prior = self.dynamics.observe(embed, action, is_first)

        # ---- kl loss ---------------------------------------------------
        kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
            post, prior, cfg.kl_free_bits, cfg.kl_dyn_scale, cfg.kl_rep_scale,
        )
        # kl_loss shape [B, T].

        # ---- decoder + reward + cont ------------------------------------
        feat = self.dynamics.get_feat(post)                  # [B, T, feat_size]
        recon_dists = self.decoder(feat)
        reward_dist = self.reward_head(feat)
        cont_dist = self.cont_head(feat)

        # Reconstruction losses per key.
        losses: dict[str, torch.Tensor] = {}
        for key, dist in recon_dists.items():
            target = obs_t[key]
            losses[key] = -dist.log_prob(target)             # [B, T]

        # Reward: symlog-disc log_prob expects target shape (..., 1). Reward
        # from replay is [B, T] scalar; unsqueeze.
        losses["reward"] = -reward_dist.log_prob(reward.unsqueeze(-1)) * cfg.reward_loss_scale
        # Cont: target is (1 - is_terminal). vendor dist=binary expects [B, T, 1].
        cont_target = (1.0 - is_terminal).unsqueeze(-1)
        losses["cont"] = -cont_dist.log_prob(cont_target) * cfg.cont_loss_scale

        # ---- total loss ------------------------------------------------
        model_loss = sum(losses.values()) + kl_loss           # [B, T]
        total = model_loss.mean()

        # ---- optimize --------------------------------------------------
        metrics = self._opt(total, list(self.parameters()))

        # ---- log breakdown --------------------------------------------
        metrics.update({f"loss/{k}": float(v.mean().item()) for k, v in losses.items()})
        metrics["loss/kl"] = float(kl_value.mean().item())
        metrics["loss/kl_dyn"] = float(dyn_loss.mean().item())
        metrics["loss/kl_rep"] = float(rep_loss.mean().item())
        metrics["loss/total"] = float(total.item())

        # Detach posterior for behavior training.
        post_detached = {k: v.detach() for k, v in post.items()}
        context = {"embed": embed.detach(), "feat": feat.detach()}
        return post_detached, context, metrics


class IsaacImagBehavior(nn.Module):
    """Actor + critic trained on imagined RSSM rollouts.

    Actor: MultiDiscreteActionHead. Critic: 255-bin twohot value MLP.
    Loss: reinforce actor + entropy bonus, λ-return critic (twohot CE).
    Slow-critic EMA + optional reward EMA follow DreamerV3.
    """

    def __init__(self, cfg: DreamerConfig, world_model: IsaacWorldModel):
        super().__init__()
        self.cfg = cfg
        self.world_model = world_model
        device = world_model._device
        self._device = device

        feat_size = cfg.rssm_stoch * cfg.rssm_discrete + cfg.rssm_deter

        # ---- Actor -----------------------------------------------------
        from ..spaces import ACTION_FACTORS
        self.actor = MultiDiscreteActionHead(
            feat_size=feat_size,
            factors=tuple(int(x) for x in ACTION_FACTORS.tolist()),
            hidden=cfg.actor_hidden,
            layers=cfg.actor_layers,
            unimix_ratio=cfg.unimix_ratio,
            act=cfg.rssm_act,
        )

        # ---- Critic (255-bin twohot) -----------------------------------
        self.critic = networks.MLP(
            feat_size,
            (255,),
            cfg.critic_layers,
            cfg.critic_hidden,
            cfg.rssm_act,
            cfg.rssm_norm,
            dist="symlog_disc",
            outscale=0.0,
            device=str(device),
            name="Critic",
        )
        if cfg.slow_target:
            self._slow_critic = copy.deepcopy(self.critic)
            for p in self._slow_critic.parameters():
                p.requires_grad_(False)
            self._slow_updates = 0

        # ---- Optimizers ------------------------------------------------
        self._actor_opt = tools.Optimizer(
            "actor",
            list(self.actor.parameters()),
            cfg.actor_lr,
            cfg.actor_eps,
            cfg.actor_grad_clip,
            cfg.weight_decay,
            opt="adam",
            use_amp=False,
        )
        self._critic_opt = tools.Optimizer(
            "critic",
            list(self.critic.parameters()),
            cfg.critic_lr,
            cfg.critic_eps,
            cfg.critic_grad_clip,
            cfg.weight_decay,
            opt="adam",
            use_amp=False,
        )

        # Reward EMA for advantage normalization (DreamerV3).
        if cfg.reward_ema:
            self.register_buffer("_reward_ema", torch.zeros(2, device=device))

        self.to(device)

    # ------------------------------------------------------------------
    # imagination rollout
    # ------------------------------------------------------------------

    def _imagine(self, start_state: dict[str, torch.Tensor], horizon: int) -> tuple[
        torch.Tensor, dict[str, torch.Tensor], torch.Tensor
    ]:
        """Roll ``horizon`` imagined steps forward from batched start states.

        Returns (feats, states, actions) each of shape [H+1, B, ...].
        The first entry is the start state's feat (with no action taken).
        """
        dyn = self.world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start_state.items()}

        feats = [dyn.get_feat(start)]
        states = {k: [v] for k, v in start.items()}
        actions: list[torch.Tensor] = []
        state = start

        for _ in range(horizon):
            feat = dyn.get_feat(state)
            action = self.actor(feat.detach()).sample()          # [B, num_actions]
            next_state = dyn.img_step(state, action)
            feats.append(dyn.get_feat(next_state))
            for k, v in next_state.items():
                states[k].append(v)
            actions.append(action)
            state = next_state

        feats_t = torch.stack(feats, dim=0)                       # [H+1, B, feat]
        states_t = {k: torch.stack(v, dim=0) for k, v in states.items()}
        actions_t = torch.stack(actions, dim=0)                   # [H, B, num_actions]
        return feats_t, states_t, actions_t

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def _compute_targets(
        self,
        feats: torch.Tensor,                # [H+1, B, feat]
        states: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """λ-returns for imagined rollouts. Returns (target, weights, base).

        Keep the trailing dim on reward/value/discount ([H, B, 1] not [H, B]) —
        the vendored ``lambda_return`` reshapes assuming a 3D input.
        """
        cfg = self.cfg
        wm = self.world_model

        # Reward and continue prediction on the imagined feats.
        reward = wm.reward_head(feats).mode()                # [H+1, B, 1]
        cont = wm.cont_head(feats).mean                       # [H+1, B, 1]
        # Cont dist is Bernoulli through Independent(..., 1); its .mean is shape
        # [..., 1]. Reward's twohot .mode() also keeps the last dim.
        if cont.dim() == feats.dim() - 1:                    # [H+1, B] -> [H+1, B, 1]
            cont = cont.unsqueeze(-1)
        discount = cfg.gamma * cont

        value = self.critic(feats).mode()                    # [H+1, B, 1]
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=cfg.gae_lambda,
            axis=0,
        )
        # lambda_return returns a list of per-timestep [B, 1] tensors.
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], dim=0), dim=0
        ).detach()
        base = value[:-1]
        return target, weights, base

    def _actor_loss(
        self,
        feats: torch.Tensor,
        actions: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        # Actor produces distributions from feats[0..H-1] — the steps where an
        # action was actually taken. feats[H] is the terminal imagined state.
        policy = self.actor(feats[:-1].detach())                  # dist over [H, B, ...]
        # ``lambda_return`` returns a tuple/list of B elements, each [H, 1].
        # Stack on dim=1 to reconstitute [H, B, 1] (matches vendor convention).
        if isinstance(target, (list, tuple)):
            target = torch.stack(list(target), dim=1)             # [H, B, 1]

        # Reward EMA normalization (5-95 percentile scale).
        if cfg.reward_ema:
            flat = target.detach().flatten()
            q = torch.quantile(flat, torch.tensor([0.05, 0.95], device=flat.device))
            with torch.no_grad():
                self._reward_ema[:] = 0.01 * q + 0.99 * self._reward_ema
            scale = (self._reward_ema[1] - self._reward_ema[0]).clip(min=1.0).detach()
            offset = self._reward_ema[0].detach()
            adv = (target - offset) / scale - (base - offset) / scale
        else:
            adv = target - base

        # log_prob at t=0..H-1 (matches actions); unsqueeze to [H, B, 1] to match target.
        log_prob = policy.log_prob(actions).unsqueeze(-1)         # [H, B, 1]
        ent = policy.entropy()                                     # [H, B]

        # Reinforce: log_prob * detached advantage. weights[:-1] is [H, B, 1].
        actor_target = log_prob * adv.detach()
        actor_loss = -(weights[:-1] * actor_target).mean()
        # Entropy bonus across imagined trajectory.
        actor_loss = actor_loss - cfg.actor_entropy * ent.mean()

        metrics = {
            "loss/actor": float(actor_loss.item()),
            "loss/actor_entropy": float(ent.mean().item()),
            "loss/actor_adv_mean": float(adv.mean().item()),
        }
        return actor_loss, metrics

    def _critic_loss(
        self,
        feats: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        value_input = feats.detach()
        value_dist = self.critic(value_input[:-1])
        if isinstance(target, (list, tuple)):
            target = torch.stack(list(target), dim=1)             # [H, B, 1]

        # symlog_disc's log_prob expects a target with trailing dim of size 1
        # matching self.buckets (255) via the twohot machinery. target already
        # has shape [H, B, 1] — pass as-is.
        loss = -value_dist.log_prob(target.detach())
        if cfg.slow_target:
            slow_dist = self._slow_critic(value_input[:-1])
            loss = loss - value_dist.log_prob(slow_dist.mode().detach())
        loss = (weights[:-1].squeeze(-1) * loss).mean()
        return loss, {"loss/critic": float(loss.item())}

    def _update_slow(self):
        if not self.cfg.slow_target:
            return
        if self._slow_updates % self.cfg.slow_target_update == 0:
            frac = self.cfg.slow_target_fraction
            for s, d in zip(self.critic.parameters(), self._slow_critic.parameters()):
                d.data = frac * s.data + (1 - frac) * d.data
        self._slow_updates += 1

    def train_step(self, start_state: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        self._update_slow()

        feats, states, actions = self._imagine(start_state, cfg.imag_horizon)
        target, weights, base = self._compute_targets(feats, states)

        actor_loss, actor_metrics = self._actor_loss(feats, actions, target, weights, base)
        # Actor step
        self._actor_opt(actor_loss, list(self.actor.parameters()))

        critic_loss, critic_metrics = self._critic_loss(feats, target, weights)
        self._critic_opt(critic_loss, list(self.critic.parameters()))

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        return metrics
