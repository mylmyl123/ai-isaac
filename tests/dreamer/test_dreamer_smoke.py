"""End-to-end smoke: 100-step prefill + a few WM+behavior updates on a fake env."""
import numpy as np
import pytest
import torch

from isaac_rl.dreamer.config import DreamerConfig
from isaac_rl.dreamer.isaac_models import IsaacImagBehavior, IsaacWorldModel
from isaac_rl.dreamer.replay import OBS_SCHEMA, SequenceReplay


def _fake_batch(B: int, T: int, onehot_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random-ish transitions in the exact replay schema."""
    batch = {}
    for k, (shape, dtype) in OBS_SCHEMA.items():
        if k in ("passives", "room_grid", "doors", "enemies_mask", "projectiles_mask", "pickups_mask"):
            batch[k] = (rng.uniform(size=(B, T) + shape) > 0.7).astype(dtype)
        else:
            batch[k] = rng.standard_normal(size=(B, T) + shape).astype(dtype) * 0.3
    action = np.zeros((B, T, onehot_dim), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            action[b, t, rng.integers(0, 9)] = 1.0
            action[b, t, 9 + rng.integers(0, 5)] = 1.0
    batch["action"] = action
    batch["reward"] = (rng.standard_normal((B, T)) * 0.1).astype(np.float32)
    batch["is_first"] = np.zeros((B, T), dtype=np.float32)
    batch["is_first"][:, 0] = 1.0
    batch["is_terminal"] = np.zeros((B, T), dtype=np.float32)
    return batch


def test_world_model_train_step_runs():
    cfg = DreamerConfig(device="cpu", batch_size=2, seq_len=8, imag_horizon=4)
    wm = IsaacWorldModel(cfg)
    rng = np.random.default_rng(0)
    batch = _fake_batch(cfg.batch_size, cfg.seq_len, 14, rng)
    post, ctx, metrics = wm.train_step(batch)
    assert "loss/total" in metrics
    assert np.isfinite(metrics["loss/total"])
    assert post["deter"].shape == (cfg.batch_size, cfg.seq_len, cfg.rssm_deter)


def test_behavior_train_step_runs():
    cfg = DreamerConfig(device="cpu", batch_size=2, seq_len=8, imag_horizon=4)
    wm = IsaacWorldModel(cfg)
    beh = IsaacImagBehavior(cfg, wm)
    rng = np.random.default_rng(1)
    batch = _fake_batch(cfg.batch_size, cfg.seq_len, 14, rng)
    post, _, _ = wm.train_step(batch)
    metrics = beh.train_step(post)
    assert "loss/actor" in metrics
    assert "loss/critic" in metrics
    assert np.isfinite(metrics["loss/actor"])
    assert np.isfinite(metrics["loss/critic"])


def test_world_model_loss_decreases_over_updates():
    """Sanity: on the same batch, WM total loss should trend down over 5 updates."""
    cfg = DreamerConfig(device="cpu", batch_size=2, seq_len=8, imag_horizon=4)
    wm = IsaacWorldModel(cfg)
    rng = np.random.default_rng(2)
    batch = _fake_batch(cfg.batch_size, cfg.seq_len, 14, rng)
    losses = []
    for _ in range(5):
        _, _, m = wm.train_step(batch)
        losses.append(m["loss/total"])
    assert losses[-1] < losses[0], f"WM loss did not decrease: {losses}"


def test_replay_to_wm_end_to_end():
    """Fill a replay buffer, sample, run one WM + behavior update."""
    cfg = DreamerConfig(device="cpu", batch_size=2, seq_len=8, imag_horizon=4)
    wm = IsaacWorldModel(cfg)
    beh = IsaacImagBehavior(cfg, wm)
    rng = np.random.default_rng(3)

    replay = SequenceReplay(capacity=200, onehot_dim=14)
    for i in range(100):
        flat = {}
        for k, (shape, dtype) in OBS_SCHEMA.items():
            flat[k] = np.zeros(shape, dtype=dtype)
        action = np.zeros(14, dtype=np.float32)
        action[rng.integers(0, 9)] = 1.0
        action[9 + rng.integers(0, 5)] = 1.0
        replay.add(flat, action, float(rng.standard_normal()) * 0.1,
                   is_first=(i == 0), is_terminal=False, is_last=False)

    batch = replay.sample(cfg.batch_size, cfg.seq_len, rng=rng)
    post, _, m_wm = wm.train_step(batch)
    m_beh = beh.train_step(post)
    assert np.isfinite(m_wm["loss/total"])
    assert np.isfinite(m_beh["loss/actor"])
    assert np.isfinite(m_beh["loss/critic"])
