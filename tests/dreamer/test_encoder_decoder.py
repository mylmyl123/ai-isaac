"""Encoder/decoder round-trip + shape correctness."""
import numpy as np
import pytest
import torch

from isaac_rl.dreamer.encoder import EncoderConfig, IsaacObsEncoder
from isaac_rl.dreamer.decoder import DecoderConfig, IsaacObsDecoder
from isaac_rl.spaces import flatten_dict_obs, zero_obs


def _random_flat_obs(rng: np.random.Generator, dtype=np.float32) -> dict:
    """Random obs matching the flat schema, roughly matching real magnitudes."""
    z = zero_obs()
    flat = flatten_dict_obs(z)
    out = {}
    for k, v in flat.items():
        if k in ("passives", "room_grid", "doors", "enemies_mask", "projectiles_mask", "pickups_mask"):
            # 0/1 targets
            out[k] = (rng.uniform(size=v.shape) > 0.7).astype(dtype)
        else:
            out[k] = rng.standard_normal(size=v.shape).astype(dtype) * 0.5
    return out


def test_encoder_output_shape_batch():
    enc = IsaacObsEncoder(EncoderConfig(embed_dim=512))
    rng = np.random.default_rng(0)
    flat = _random_flat_obs(rng)
    # [B=3]
    batch = {k: torch.from_numpy(np.stack([v for _ in range(3)])) for k, v in flat.items()}
    out = enc(batch)
    assert out.shape == (3, 512), f"got {out.shape}"


def test_encoder_output_shape_seq_batch():
    enc = IsaacObsEncoder(EncoderConfig(embed_dim=1024))
    rng = np.random.default_rng(1)
    flat = _random_flat_obs(rng)
    # [T=4, B=2]
    batch = {k: torch.from_numpy(np.stack([np.stack([v for _ in range(2)]) for _ in range(4)])) for k, v in flat.items()}
    out = enc(batch)
    assert out.shape == (4, 2, 1024)


def test_decoder_log_probs_return_batch_time_shape():
    """Every reconstructed key produces a [B, T] log-prob."""
    feat_size = 32 * 32 + 512
    dec = IsaacObsDecoder(feat_size, DecoderConfig(hidden=128, layers=1))
    feat = torch.randn(2, 4, feat_size)
    dists = dec(feat)

    rng = np.random.default_rng(2)
    flat = _random_flat_obs(rng)
    targets = {
        k: torch.from_numpy(np.stack([np.stack([v for _ in range(2)]) for _ in range(4)])).permute(1, 0, *range(2, 2 + v.ndim)).contiguous()
        for k, v in flat.items()
    }
    for key in dec.reconstructed_keys:
        lp = dists[key].log_prob(targets[key])
        assert lp.shape == (2, 4), f"{key}: got {tuple(lp.shape)}"
        assert torch.isfinite(lp).all(), f"{key}: non-finite log_prob"


def test_decoder_loss_is_backprop_able():
    feat_size = 32 * 32 + 512
    enc = IsaacObsEncoder(EncoderConfig(embed_dim=1024, trunk_dim=512))
    dec = IsaacObsDecoder(feat_size, DecoderConfig(hidden=128, layers=1))

    rng = np.random.default_rng(3)
    flat = _random_flat_obs(rng)
    batch = {k: torch.from_numpy(np.stack([np.stack([v for _ in range(2)]) for _ in range(4)])) for k, v in flat.items()}
    embed = enc(batch)   # [4, 2, 1024]
    # Fabricate a fake feat by projecting embed into feat_size.
    proj = torch.nn.Linear(1024, feat_size)
    feat = proj(embed)
    dists = dec(feat)

    targets = batch
    loss = sum(-dists[k].log_prob(targets[k]).mean() for k in dec.reconstructed_keys)
    assert torch.isfinite(loss).all()
    loss.backward()
    # Verify gradients exist on some enc/dec params.
    got_enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    got_dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())
    assert got_enc_grad
    assert got_dec_grad


def test_encoder_ignores_z_and_last_action():
    """Setting z / last_action to different values should not change the encoder output."""
    enc = IsaacObsEncoder(EncoderConfig(embed_dim=256))
    enc.eval()

    rng = np.random.default_rng(4)
    flat_a = _random_flat_obs(rng)
    flat_b = {k: v.copy() for k, v in flat_a.items()}
    # Perturb z and last_action wildly in flat_b.
    flat_b["z"] = flat_b["z"] + 100.0
    flat_b["last_action"] = np.array([0.9, 0.9], dtype=np.float32)

    a = {k: torch.from_numpy(v).unsqueeze(0) for k, v in flat_a.items()}
    b = {k: torch.from_numpy(v).unsqueeze(0) for k, v in flat_b.items()}
    with torch.no_grad():
        out_a = enc(a)
        out_b = enc(b)
    diff = (out_a - out_b).abs().max()
    assert diff < 1e-5, f"encoder should ignore z/last_action but diff={diff}"
