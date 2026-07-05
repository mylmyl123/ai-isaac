"""Sequence replay buffer: episode boundaries, terminal-obs, wrapping."""
import numpy as np
import pytest

from isaac_rl.dreamer.replay import OBS_SCHEMA, SequenceReplay
from isaac_rl.spaces import flatten_dict_obs, zero_obs


def _fake_obs(step: int, ep: int) -> dict[str, np.ndarray]:
    flat = flatten_dict_obs(zero_obs())
    # Perturb player so different eps/steps look different in the tests.
    flat = {k: v.copy() for k, v in flat.items()}
    flat["player"][0] = float(step)
    flat["player"][1] = float(ep)
    return flat


def test_add_and_len():
    buf = SequenceReplay(capacity=100, onehot_dim=14)
    assert len(buf) == 0
    for i in range(50):
        buf.add(_fake_obs(i, 0), np.zeros(14), 0.0, i == 0, False, False)
    assert len(buf) == 50


def test_sample_shapes():
    buf = SequenceReplay(capacity=200, onehot_dim=14)
    for i in range(100):
        buf.add(_fake_obs(i, 0), np.eye(14)[i % 14], float(i), i == 0, False, i == 99)
    batch = buf.sample(batch_size=4, seq_len=16)
    for key, (shape, _) in OBS_SCHEMA.items():
        assert batch[key].shape == (4, 16) + shape, f"{key} shape {batch[key].shape}"
    assert batch["action"].shape == (4, 16, 14)
    assert batch["reward"].shape == (4, 16)
    assert batch["is_first"].shape == (4, 16)


def test_capacity_wraps():
    buf = SequenceReplay(capacity=50, onehot_dim=14)
    for i in range(100):
        buf.add(_fake_obs(i, 0), np.zeros(14), 0.0, False, False, False)
    assert len(buf) == 50
    # Should still sample cleanly.
    batch = buf.sample(4, 8)
    assert batch["player"].shape == (4, 8, 40)


def test_raises_when_seq_too_long():
    buf = SequenceReplay(capacity=100, onehot_dim=14)
    for i in range(10):
        buf.add(_fake_obs(i, 0), np.zeros(14), 0.0, i == 0, False, False)
    with pytest.raises(ValueError):
        buf.sample(1, 16)


def test_is_first_marks_episode_boundaries():
    """Transitions written with is_first=True must round-trip through sampling."""
    buf = SequenceReplay(capacity=1000, onehot_dim=14)
    firsts = []
    for i in range(500):
        is_first = (i % 25 == 0)   # 20 episodes of length 25
        firsts.append(i if is_first else -1)
        buf.add(_fake_obs(i, i // 25), np.zeros(14), 0.0, is_first, False, i % 25 == 24)
    # Total is_first flags = 500 // 25 = 20.
    total_first = int(buf._is_first.sum())
    assert total_first == 20

    # Sample enough sequences to have some hit an episode start.
    rng = np.random.default_rng(7)
    total_sample_firsts = 0
    for _ in range(30):
        batch = buf.sample(8, 16, rng=rng)
        total_sample_firsts += int(batch["is_first"].sum())
    # We can't assert an exact number (random), but with 240 * 16 samples out
    # of a 500-transition buffer where 20/500 have is_first, we expect ~150 hits.
    assert total_sample_firsts > 20, f"is_first flags too rare in samples: {total_sample_firsts}"
