"""Behavior cloning: demo collection + supervised pretraining.

Two entry points:

  collect_demos(env, policy, n_steps, save_path):
      Run a policy (typically HeuristicPolicy) in the vec env for n_steps and
      save (obs, action) trajectories to a .npz file. No RL training.

  bc_pretrain(policy_net, demos_path, epochs, ...):
      Supervised pretraining: load (obs, action) pairs, train the policy net
      to predict the demo action from the obs via cross-entropy on each of
      the 5 MultiDiscrete action heads.

Design notes:

The saved .npz contains one entry per observation field (batched over all
recorded ticks). This keeps disk format aligned with what the vec env yields
so we can feed it straight into the policy net without any encoding step.

Action distribution is imbalanced (heuristic mostly doesn't fire pill/bomb/
item). The BC loss weights the shoot/move heads more heavily than the three
binary heads. Otherwise the model wastes capacity confidently predicting 0
on the boring heads and under-fitting movement/shoot.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .spaces import flatten_dict_obs

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- collect


def collect_demos(
    env,
    policy,
    n_steps: int,
    save_path: str | Path,
    log_every: int = 500,
) -> Path:
    """Run `policy` in `env` for n_steps and save (obs, action) trajectories.

    Args:
        env: SyncVecEnv-like object. env.reset() returns (list[obs_dict], list[info_dict]).
             env.step(actions_arr) returns (list[obs_dict], rewards, terms, truncs, list[info_dict]).
        policy: object with .act(raw_obs) -> action ndarray of shape (5,).
        n_steps: total demonstration steps to collect across all envs. Real
                 wall-clock is roughly n_steps / (n_envs * sps).
        save_path: destination .npz file. Parent dirs are created.
        log_every: log a progress line every N env steps (across the fleet).

    Returns the resolved save_path.
    """
    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_envs = env.n
    action_dim = len(env.action_space.nvec) if hasattr(env.action_space, "nvec") else 2
    log.info("collecting %d heuristic demo steps into %s (%d envs, action_dim=%d)",
             n_steps, save_path, n_envs, action_dim)

    obs_list, infos = env.reset()

    # Buffers accumulate across ticks. Each obs is flattened via
    # flatten_dict_obs() to expose enemies_feats / enemies_mask / etc as flat
    # keys — same layout the policy network expects. We accumulate per-key
    # numpy arrays and stack at the end.
    per_field_stacks: dict[str, list[np.ndarray]] = {}
    action_stack: list[np.ndarray] = []

    def _accumulate_obs(o: dict[str, Any]) -> None:
        flat = flatten_dict_obs(o)
        for k, v in flat.items():
            per_field_stacks.setdefault(k, []).append(np.asarray(v))

    total_steps = 0
    t_start = time.time()
    # Import lazily so bc.py doesn't hard-require the human_override module.
    try:
        from isaac_rl.human_override import get_instance as _get_override
    except ImportError:
        _get_override = lambda: None   # type: ignore
    while total_steps < n_steps:
        # One action per env from the heuristic (needs raw obs from info).
        actions = np.zeros((n_envs, action_dim), dtype=np.int64)
        for i in range(n_envs):
            raw = infos[i].get("raw") or {}
            act = policy.act(raw)
            # Pad or truncate to match env's action dim (heuristic may return
            # 2-dim under the new action space).
            actions[i, :min(len(act), action_dim)] = act[:action_dim]

        # Apply human override BEFORE recording: if the user is manually
        # steering the bot, we want BC to learn from the HUMAN's action, not
        # the heuristic's. This makes demo collection double as DAgger-style
        # correction gathering.
        override = _get_override()
        if override is not None:
            move, shoot = override.get_action()
            if move is not None or shoot is not None:
                # Apply to all envs (single keyboard, single override).
                for i in range(n_envs):
                    if move is not None and action_dim >= 1:
                        actions[i, 0] = move
                    if shoot is not None and action_dim >= 2:
                        actions[i, 1] = shoot

        # Record (obs_before_action, action).
        for i in range(n_envs):
            _accumulate_obs(obs_list[i])
            action_stack.append(actions[i])

        obs_list, _rewards, _terms, _truncs, infos = env.step(actions)
        total_steps += n_envs

        if total_steps % max(1, log_every) < n_envs:
            elapsed = time.time() - t_start
            sps = total_steps / max(1e-6, elapsed)
            log.info("[demos] step %s/%s  sps=%.0f", f"{total_steps:,}", f"{n_steps:,}", sps)

    # Stack all buffers into arrays. Save as .npz with flat keys.
    log.info("[demos] stacking %d transitions...", len(action_stack))
    save_dict: dict[str, np.ndarray] = {
        "actions": np.stack(action_stack),
        "n_transitions": np.array([len(action_stack)], dtype=np.int64),
    }
    for k, buf in per_field_stacks.items():
        save_dict[f"obs__{k}"] = np.stack(buf)

    np.savez_compressed(save_path, **save_dict)
    log.info(
        "[demos] wrote %s (%d transitions, %.1f MB, %.1f min elapsed)",
        save_path, len(action_stack), save_path.stat().st_size / 1e6, (time.time() - t_start) / 60,
    )
    return save_path


# ---------------------------------------------------------------- pretraining


def _load_demos_to_tensors(demos_path: str | Path, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Load a .npz demo file and return (obs_dict, actions) as tensors on `device`.

    Handles both:
      * Flat keys: obs__enemies_feats, obs__enemies_mask, obs__player, ...
        (current format, matches flatten_dict_obs output)
      * Nested keys: obs__enemies__feats, obs__enemies__mask, ...
        (legacy format from older commits — auto-converted to flat)
    """
    log.info("loading demos from %s", demos_path)
    data = np.load(demos_path)
    keys = list(data.keys())

    actions = torch.as_tensor(np.asarray(data["actions"]), dtype=torch.long, device=device)
    obs: dict[str, torch.Tensor] = {}

    for key in keys:
        if key in ("actions", "n_transitions") or not key.startswith("obs__"):
            continue
        parts = key.split("__")
        arr = np.asarray(data[key])
        # Cast mask fields to float32 (encode_obs uses int8 but model expects float32).
        dtype = torch.float32
        if len(parts) == 2:
            flat_key = parts[1]
        elif len(parts) == 3:
            # Legacy nested layout: obs__enemies__feats -> enemies_feats
            flat_key = f"{parts[1]}_{parts[2]}"
        else:
            log.warning("skipping unrecognized demo key: %s", key)
            continue
        obs[flat_key] = torch.as_tensor(arr, dtype=dtype, device=device)

    log.info("loaded %d transitions, obs keys: %s", actions.shape[0], sorted(obs.keys()))
    return obs, actions


def _slice_obs(obs: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    """Index into every tensor in a flat obs dict."""
    return {k: v[idx] for k, v in obs.items()}


def bc_pretrain(
    policy_net,
    demos_path: str | Path,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3.0e-4,
    device: torch.device | None = None,
    action_head_weights: tuple[float, ...] | None = None,
    log_every: int = 20,
) -> None:
    """Supervised-learning pretraining. Mutates policy_net in place.

    Trains a 5-head classifier: the policy's forward pass produces 5 logits
    tensors (move, shoot, use_active, drop_bomb, pill_card). We use
    cross-entropy per head against the demo action indices, weighted by
    action_head_weights so movement + shooting dominate the loss.

    Recurrent hidden state is NOT propagated across the shuffled batch (each
    minibatch is a random slice, so the GRU state is reset per minibatch —
    that's fine for BC, we're only teaching action prediction).
    """
    if device is None:
        device = next(policy_net.parameters()).device

    obs, actions = _load_demos_to_tensors(demos_path, device)
    n = actions.shape[0]
    if n == 0:
        log.warning("bc_pretrain: no transitions loaded, skipping")
        return

    # Auto-derive head weights from action dim if not provided. All heads get
    # equal weight of 1.0. Old default (2.0, 2.0, 0.5, 0.5, 0.5) was for the
    # legacy 5-head action space; now that the action space is 2 dims, uniform
    # weights are cleaner.
    action_dim = actions.shape[1]
    if action_head_weights is None:
        action_head_weights = (1.0,) * action_dim
    if len(action_head_weights) != action_dim:
        log.warning("action_head_weights length %d != action_dim %d; using uniform weights",
                    len(action_head_weights), action_dim)
        action_head_weights = (1.0,) * action_dim

    optim = torch.optim.Adam(policy_net.parameters(), lr=lr)
    policy_net.train()

    log.info("BC: %d transitions, %d epochs, batch_size=%d, lr=%.1e, action_dim=%d, head_weights=%s",
             n, epochs, batch_size, lr, action_dim, action_head_weights)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        epoch_correct = np.zeros(action_dim, dtype=np.int64)
        epoch_total = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            b_obs = _slice_obs(obs, idx)
            b_act = actions[idx]                                # [B, 5]
            B = b_act.shape[0]

            # Initial hidden state per minibatch. Policy expects [B, gru_dim] init.
            hidden = policy_net.initial_hidden(B, device)
            done_mask = torch.zeros(B, device=device)
            logits, _value, _hidden_out = policy_net.step(b_obs, hidden, done_mask=done_mask)
            # logits: list of 5 tensors [B, K_i]
            loss = 0.0
            for head_idx, head_logits in enumerate(logits):
                weight = action_head_weights[head_idx]
                loss = loss + weight * F.cross_entropy(head_logits, b_act[:, head_idx])
                epoch_correct[head_idx] += (head_logits.argmax(dim=-1) == b_act[:, head_idx]).sum().item()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optim.step()

            epoch_loss += float(loss.item())
            epoch_total += B
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        acc = epoch_correct / max(1, epoch_total)
        # Format accuracy per action head dynamically (2 or 5 heads).
        acc_str = " ".join(f"h{i}={a:.2f}" for i, a in enumerate(acc[:action_dim]))
        log.info("[BC] epoch %d/%d  loss=%.3f  acc: %s", epoch + 1, epochs, avg_loss, acc_str)

    policy_net.eval()
    log.info("[BC] pretraining complete")
