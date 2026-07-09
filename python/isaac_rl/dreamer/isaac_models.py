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
import time
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


class _NullContext:
    """no-op context manager used when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *args): return False


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

        # Optional: torch.compile the RSSM step methods. These are called
        # seq_len times per WM update (32 -> 16 in the XS config), and each
        # call has Python-dispatch overhead across nn.Sequential layers.
        # torch.compile fuses the graph, cutting ~1.5-2x off wm_rssm_observe.
        #
        # Requires Triton, which is broken/missing on Windows in many PyTorch
        # builds. If the compile succeeds at registration time but throws at
        # FIRST CALL time (Triton missing), we catch that at the outer
        # train_step boundary and revert to eager for the rest of the run.
        # Config default is False (off) — enable only if Triton is verified.
        compile_ok = getattr(cfg, "compile_rssm", False) and hasattr(torch, "compile") and device.type == "cuda"
        self._rssm_compiled = False
        self._eager_obs_step = self.dynamics.obs_step
        self._eager_img_step = self.dynamics.img_step
        if compile_ok:
            try:
                self.dynamics.obs_step = torch.compile(
                    self.dynamics.obs_step, mode="reduce-overhead", fullgraph=False,
                )
                self.dynamics.img_step = torch.compile(
                    self.dynamics.img_step, mode="reduce-overhead", fullgraph=False,
                )
                self._rssm_compiled = True
            except Exception as e:  # pragma: no cover
                import logging
                logging.getLogger("dreamer").warning(
                    "torch.compile on RSSM failed at init (%s); falling back to eager.", e,
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

        # AMP autocast dtype (set once at init based on cfg.amp_dtype). Wraps
        # forward passes to run bf16/fp16 matmul + attention. On Ampere+ we
        # default to bf16 — same throughput as fp16 with better numerics, no
        # GradScaler needed (bf16 has fp32-equivalent dynamic range).
        amp_str = getattr(cfg, "amp_dtype", "off")
        if amp_str == "bf16":
            self._amp_dtype = torch.bfloat16
        elif amp_str == "fp16":
            self._amp_dtype = torch.float16
        else:
            self._amp_dtype = None

        # ---- Optimizer (vendor Optimizer wraps AMP + grad clip) --------
        # We handle autocast ourselves (see train_step); the vendor Optimizer's
        # GradScaler is only needed for fp16. With bf16 it's a no-op, so
        # keep use_amp=False and the GradScaler stays disabled.
        self._opt = tools.Optimizer(
            "model",
            list(self.parameters()),
            cfg.world_model_lr,
            cfg.world_model_eps,
            cfg.world_model_grad_clip,
            cfg.weight_decay,
            opt="adam",
            use_amp=(amp_str == "fp16"),
        )

        self.to(device)

    def _revert_to_eager(self, reason: str) -> None:
        """Restore unmodified obs_step / img_step so subsequent calls run in
        eager mode. Called when torch.compile throws at runtime (Triton
        missing on Windows, unsupported graph break, etc.).
        """
        if not self._rssm_compiled:
            return
        import logging
        logging.getLogger("dreamer").warning(
            "torch.compile RSSM failed at runtime (%s); reverting to eager. "
            "Set compile_rssm=false in the YAML to silence.", reason,
        )
        self.dynamics.obs_step = self._eager_obs_step
        self.dynamics.img_step = self._eager_img_step
        self._rssm_compiled = False

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

        If torch.compile on the RSSM throws at runtime (Triton missing, graph
        break, etc.), we catch the error, revert to eager mode, and retry
        once. Subsequent calls run in eager for the rest of the run.
        """
        try:
            return self._train_step_inner(batch)
        except Exception as e:
            msg = str(e).lower()
            if self._rssm_compiled and any(k in msg for k in ("triton", "torchdynamo", "compile", "torch._dynamo")):
                self._revert_to_eager(str(e))
                return self._train_step_inner(batch)
            raise

    def _train_step_inner(self, batch: dict[str, np.ndarray]) -> tuple[dict, dict, dict[str, float]]:
        """Inner WM training step (see ``train_step`` for the retry wrapper).

        Populates ``self.last_step_times`` with per-section wall-clock in ms
        (CUDA-synced), so the trainer can propagate them to TB. Sections:
        obs_to_gpu, encode, rssm_observe, kl, decode, losses, backward.
        """
        cfg = self.cfg
        device = self._device
        is_cuda = device.type == "cuda"
        times: dict[str, float] = {}

        def _tic():
            if is_cuda:
                torch.cuda.synchronize()
            return time.perf_counter()

        def _toc(name: str, t0: float):
            if is_cuda:
                torch.cuda.synchronize()
            times[name] = 1000.0 * (time.perf_counter() - t0)

        # ---- obs marshaling (numpy -> torch on device) --------------------
        t = _tic()
        obs_t = {k: _to_tensor(v, device) for k, v in batch.items()
                 if k not in ("action", "reward", "is_first", "is_terminal")}
        action = _to_tensor(batch["action"], device)         # [B, T, num_actions]
        reward = _to_tensor(batch["reward"], device)         # [B, T]
        is_first = _to_tensor(batch["is_first"], device)     # [B, T]
        is_terminal = _to_tensor(batch["is_terminal"], device)
        _toc("wm_obs_to_gpu", t)

        # AMP: wrap the whole forward + loss compute in autocast so matmul /
        # attention / decoder heads run in bf16/fp16. The final `.backward()`
        # is handled by the vendor Optimizer with (or without) GradScaler.
        amp_enabled = self._amp_dtype is not None and is_cuda
        _amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype)
            if amp_enabled
            else _NullContext()
        )
        with _amp_ctx:
            # ---- encode -----------------------------------------------------
            t = _tic()
            embed = self.encoder(obs_t)                         # [B, T, embed_dim]
            _toc("wm_encode", t)

            # ---- rssm observe ----------------------------------------------
            t = _tic()
            post, prior = self.dynamics.observe(embed, action, is_first)
            _toc("wm_rssm_observe", t)

            # ---- kl loss ---------------------------------------------------
            t = _tic()
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior, cfg.kl_free_bits, cfg.kl_dyn_scale, cfg.kl_rep_scale,
            )
            # Free-bits diagnostic: fraction of KL-elements above the free
            # threshold. If this is ~1.0 the clip isn't binding (KL is
            # unconstrained, so raising kl_free_bits is pointless). If it's
            # ~0.0 the KL is fully clipped and free_bits should be lowered.
            with torch.no_grad():
                kl_free_bits_frac = float((kl_value > cfg.kl_free_bits).float().mean().item())
            _toc("wm_kl", t)

            # ---- decoder + reward + cont ------------------------------------
            t = _tic()
            feat = self.dynamics.get_feat(post)                  # [B, T, feat_size]
            recon_dists = self.decoder(feat)
            reward_dist = self.reward_head(feat)
            cont_dist = self.cont_head(feat)
            _toc("wm_decode", t)

            # Reconstruction losses per key.
            t = _tic()
            losses: dict[str, torch.Tensor] = {}
            for key, dist in recon_dists.items():
                target = obs_t[key]
                losses[key] = -dist.log_prob(target)             # [B, T]

            # Reward: symlog-disc log_prob expects target shape (..., 1).
            losses["reward"] = -reward_dist.log_prob(reward.unsqueeze(-1)) * cfg.reward_loss_scale
            cont_target = (1.0 - is_terminal).unsqueeze(-1)
            losses["cont"] = -cont_dist.log_prob(cont_target) * cfg.cont_loss_scale

            model_loss = sum(losses.values()) + kl_loss
            total = model_loss.mean()
            _toc("wm_losses", t)

        # ---- optimize (fwd+bwd+step, backward runs outside autocast) -----
        t = _tic()
        metrics = self._opt(total, list(self.parameters()))
        _toc("wm_backward", t)

        # ---- log breakdown --------------------------------------------
        metrics.update({f"loss/{k}": float(v.mean().item()) for k, v in losses.items()})
        metrics["loss/kl"] = float(kl_value.mean().item())
        metrics["loss/kl_dyn"] = float(dyn_loss.mean().item())
        metrics["loss/kl_rep"] = float(rep_loss.mean().item())
        metrics["loss/kl_free_bits_frac"] = kl_free_bits_frac
        metrics["loss/total"] = float(total.item())

        # Expose timings for the trainer to log.
        self.last_step_times = times

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

        # ---- RND intrinsic curiosity (2026-07-09 v2) --------------------
        # Optional — gated by cfg.rnd_enabled. Provides intrinsic reward for
        # visiting novel states, enabling emergent multi-step behavior
        # discovery without hand-scripting chains. See
        # dreamer/intrinsic.py for full rationale.
        if getattr(cfg, "rnd_enabled", False):
            from .intrinsic import RND
            self.rnd = RND(
                feat_dim=feat_size,
                embed_dim=getattr(cfg, "rnd_embed_dim", 128),
                hidden=getattr(cfg, "rnd_hidden", 256),
                target_hidden=getattr(cfg, "rnd_target_hidden", 128),
            ).to(device)
            # Separate optimizer for the predictor. Target is frozen.
            self._rnd_opt = torch.optim.Adam(
                self.rnd.predictor.parameters(),
                lr=getattr(cfg, "rnd_lr", 1e-4),
            )
            self._rnd_scale = float(getattr(cfg, "rnd_intrinsic_scale", 0.1))
        else:
            self.rnd = None
            self._rnd_opt = None
            self._rnd_scale = 0.0

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

        # AMP dtype mirrors WM's setting.
        amp_str = getattr(cfg, "amp_dtype", "off")
        if amp_str == "bf16":
            self._amp_dtype = torch.bfloat16
        elif amp_str == "fp16":
            self._amp_dtype = torch.float16
        else:
            self._amp_dtype = None

        # ---- Optimizers ------------------------------------------------
        self._actor_opt = tools.Optimizer(
            "actor",
            list(self.actor.parameters()),
            cfg.actor_lr,
            cfg.actor_eps,
            cfg.actor_grad_clip,
            cfg.weight_decay,
            opt="adam",
            use_amp=(amp_str == "fp16"),
        )
        self._critic_opt = tools.Optimizer(
            "critic",
            list(self.critic.parameters()),
            cfg.critic_lr,
            cfg.critic_eps,
            cfg.critic_grad_clip,
            cfg.weight_decay,
            opt="adam",
            use_amp=(amp_str == "fp16"),
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
        # RND intrinsic reward (2026-07-09 v2): add curiosity bonus to
        # imagined reward so the critic learns Q-values that include
        # exploration incentive, and the actor is trained to seek novel
        # states through imagination. Detached feats so RND gradient
        # doesn't flow back into the WM.
        if self.rnd is not None and self._rnd_scale > 0.0:
            with torch.no_grad():
                intrinsic = self.rnd.intrinsic_reward(feats.detach())  # [H+1, B]
            reward = reward + self._rnd_scale * intrinsic.unsqueeze(-1)
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
        entropy_bonus = cfg.actor_entropy * ent.mean()
        actor_loss = actor_loss - entropy_bonus

        # ---- diagnostics -------------------------------------------------
        # Track the two competing loss magnitudes so we can see when entropy
        # bonus is swamping the reinforce signal (the 2026-07-06 XS pathology).
        reinforce_mag = (weights[:-1] * actor_target).abs().mean()
        metrics = {
            "loss/actor": float(actor_loss.item()),
            "loss/actor_entropy": float(ent.mean().item()),
            "loss/actor_adv_mean": float(adv.mean().item()),
            "loss/actor_adv_std": float(adv.std().item()),
            "loss/actor_adv_abs_mean": float(adv.abs().mean().item()),
            "loss/actor_target_mean": float(target.mean().item()),
            "loss/actor_target_std": float(target.std().item()),
            "loss/actor_logprob_mean": float(log_prob.mean().item()),
            "loss/actor_reinforce_mag": float(reinforce_mag.item()),
            "loss/actor_entropy_bonus_mag": float(entropy_bonus.item()),
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
        """One imagination-based actor+critic gradient update.

        Uses the world model's RSSM to imagine forward for ``imag_horizon``
        steps from each start state. If torch.compile on the RSSM throws at
        runtime, catch it, revert to eager, and retry once.
        """
        try:
            return self._train_step_inner(start_state)
        except Exception as e:
            msg = str(e).lower()
            wm = self.world_model
            if getattr(wm, "_rssm_compiled", False) and any(
                k in msg for k in ("triton", "torchdynamo", "compile", "torch._dynamo")
            ):
                wm._revert_to_eager(str(e))
                return self._train_step_inner(start_state)
            raise

    def _train_step_inner(self, start_state: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        self._update_slow()

        amp_enabled = self._amp_dtype is not None and self._device.type == "cuda"
        _amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype)
            if amp_enabled
            else _NullContext()
        )
        with _amp_ctx:
            feats, states, actions = self._imagine(start_state, cfg.imag_horizon)
            target, weights, base = self._compute_targets(feats, states)

            actor_loss, actor_metrics = self._actor_loss(feats, actions, target, weights, base)
        self._actor_opt(actor_loss, list(self.actor.parameters()))

        with _amp_ctx:
            critic_loss, critic_metrics = self._critic_loss(feats, target, weights)
        self._critic_opt(critic_loss, list(self.critic.parameters()))

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)

        # ---- RND predictor training (2026-07-09 v2) -------------------
        # Train the RND predictor on the same imagined features that fed
        # the actor/critic. Using imagined feats (rather than replay feats)
        # so the predictor sees the same state distribution that the
        # critic evaluates — keeps intrinsic-reward magnitude calibrated
        # to what the agent actually plans over. Detached feats so RND
        # doesn't affect WM gradients.
        if self.rnd is not None:
            with _amp_ctx:
                rnd_loss, rnd_metrics = self.rnd.update(feats.detach())
            self._rnd_opt.zero_grad(set_to_none=True)
            rnd_loss.backward()
            self._rnd_opt.step()
            metrics.update(rnd_metrics)

        return metrics
