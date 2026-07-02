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
    log.info("collecting %d heuristic demo steps into %s (%d envs)", n_steps, save_path, n_envs)

    obs_list, infos = env.reset()

    # Buffers accumulate across ticks. Each obs is a dict of ndarrays. We stack
    # by field at the end.
    per_field_stacks: dict[str, list[np.ndarray]] = {}
    per_field_nested: dict[str, dict[str, list[np.ndarray]]] = {}
    action_stack: list[np.ndarray] = []

    def _accumulate_obs(o: dict[str, Any]) -> None:
        for k, v in o.items():
            if isinstance(v, dict):
                per_field_nested.setdefault(k, {})
                for kk, vv in v.items():
                    per_field_nested[k].setdefault(kk, []).append(np.asarray(vv))
            else:
                per_field_stacks.setdefault(k, []).append(np.asarray(v))

    total_steps = 0
    t_start = time.time()
    while total_steps < n_steps:
        # One action per env from the heuristic (needs raw obs from info).
        actions = np.zeros((n_envs, 5), dtype=np.int64)
        for i in range(n_envs):
            raw = infos[i].get("raw") or {}
            actions[i] = policy.act(raw)

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

    # Stack all buffers into arrays. Save as .npz with flattened keys.
    log.info("[demos] stacking %d transitions...", len(action_stack))
    save_dict: dict[str, np.ndarray] = {
        "actions": np.stack(action_stack),
        "n_transitions": np.array([len(action_stack)], dtype=np.int64),
    }
    for k, buf in per_field_stacks.items():
        save_dict[f"obs__{k}"] = np.stack(buf)
    for k, nested in per_field_nested.items():
        for kk, buf in nested.items():
            save_dict[f"obs__{k}__{kk}"] = np.stack(buf)

    np.savez_compressed(save_path, **save_dict)
    log.info(
        "[demos] wrote %s (%d transitions, %.1f MB, %.1f min elapsed)",
        save_path, len(action_stack), save_path.stat().st_size / 1e6, (time.time() - t_start) / 60,
    )
    return save_path


# ---------------------------------------------------------------- pretraining


def _load_demos_to_tensors(demos_path: str | Path, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Load a .npz demo file and return (obs_dict, actions) as tensors on `device`.

    The obs dict is reconstructed to match the encode_obs output shape:
    top-level keys carry non-dict values; keys with __ separators reconstruct
    the nested {"enemies": {"feats": ..., "mask": ...}, ...} structure.
    """
    log.info("loading demos from %s", demos_path)
    data = np.load(demos_path)
    keys = list(data.keys())

    actions = torch.as_tensor(np.asarray(data["actions"]), dtype=torch.long, device=device)
    obs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}

    for key in keys:
        if key in ("actions", "n_transitions"):
            continue
        if not key.startswith("obs__"):
            continue
        parts = key.split("__")
        arr = np.asarray(data[key])
        t = torch.as_tensor(arr, dtype=torch.float32, device=device)
        if len(parts) == 2:
            # obs__<field>
            obs[parts[1]] = t
        elif len(parts) == 3:
            # obs__<field>__<subfield>
            sub = obs.setdefault(parts[1], {})
            assert isinstance(sub, dict)
            sub[parts[2]] = t

    log.info("loaded %d transitions, obs keys: %s", actions.shape[0], list(obs.keys()))
    return obs, actions


def _slice_obs(obs: dict, idx: torch.Tensor) -> dict:
    """Index into every leaf tensor in a (possibly nested) obs dict."""
    out: dict = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            out[k] = {kk: vv[idx] for kk, vv in v.items()}
        else:
            out[k] = v[idx]
    return out


def bc_pretrain(
    policy_net,
    demos_path: str | Path,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3.0e-4,
    device: torch.device | None = None,
    action_head_weights: tuple[float, ...] = (2.0, 2.0, 0.5, 0.5, 0.5),
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

    optim = torch.optim.Adam(policy_net.parameters(), lr=lr)
    policy_net.train()

    log.info("BC: %d transitions, %d epochs, batch_size=%d, lr=%.1e", n, epochs, batch_size, lr)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        epoch_correct = np.zeros(5, dtype=np.int64)
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
        log.info(
            "[BC] epoch %d/%d  loss=%.3f  acc: move=%.2f shoot=%.2f act=%.2f bomb=%.2f pill=%.2f",
            epoch + 1, epochs, avg_loss, *acc,
        )

    policy_net.eval()
    log.info("[BC] pretraining complete")
