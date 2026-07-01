"""Smoke test for the policy network: run one forward pass on a batch of zeros."""
from __future__ import annotations

import numpy as np
import torch

from isaac_rl.model import IsaacPolicy, PolicyConfig
from isaac_rl.spaces import ACTION_FACTORS, flatten_dict_obs, zero_obs


def _fake_batch(batch_size: int, device="cpu"):
    """Build a [B, ...] batched obs dict from B zero-obs samples."""
    obs_list = [zero_obs() for _ in range(batch_size)]
    flats = [flatten_dict_obs(o) for o in obs_list]
    out = {}
    for k in flats[0].keys():
        arr = np.stack([f[k] for f in flats], axis=0)
        out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


def test_policy_step_shapes():
    policy = IsaacPolicy(PolicyConfig(entity_dim=32, proj_dim=32, pickup_dim=16, trunk_dim=64, gru_dim=64))
    B = 3
    obs = _fake_batch(B)
    hidden = policy.initial_hidden(B, torch.device("cpu"))
    logits, value, new_hidden = policy.step(obs, hidden)
    assert len(logits) == len(ACTION_FACTORS)
    for i, l in enumerate(logits):
        assert l.shape == (B, ACTION_FACTORS[i])
    assert value.shape == (B,)
    assert new_hidden.shape == hidden.shape


def test_sample_and_logprob_are_finite():
    policy = IsaacPolicy(PolicyConfig(entity_dim=32, proj_dim=32, pickup_dim=16, trunk_dim=64, gru_dim=64))
    B = 4
    obs = _fake_batch(B)
    hidden = policy.initial_hidden(B, torch.device("cpu"))
    logits, _, _ = policy.step(obs, hidden)
    a = policy.sample_from_logits(logits)
    assert a.shape == (B, len(ACTION_FACTORS))
    lp = policy.log_prob_from_logits(logits, a)
    ent = policy.entropy_from_logits(logits)
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ent).all()


def test_sequence_forward_shape():
    policy = IsaacPolicy(PolicyConfig(entity_dim=32, proj_dim=32, pickup_dim=16, trunk_dim=64, gru_dim=64))
    T, B = 5, 2
    # Build a [T, B, ...] sequence by stacking two batched obs.
    seq = {}
    for t in range(T):
        b = _fake_batch(B)
        for k, v in b.items():
            seq.setdefault(k, []).append(v)
    seq = {k: torch.stack(vs, dim=0) for k, vs in seq.items()}
    dones = torch.zeros(T, B)
    init = policy.initial_hidden(B, torch.device("cpu"))
    logits, values = policy.sequence_forward(seq, dones, init)
    assert values.shape == (T * B,)
    assert len(logits) == len(ACTION_FACTORS)
