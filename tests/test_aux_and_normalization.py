"""Tests for auxiliary supervised losses in ppo.py."""
from __future__ import annotations

import numpy as np
import torch

from isaac_rl.ppo import aux_labels_from_obs, RunningMeanStd
from isaac_rl.spaces import MAX_ENEMIES, MAX_PROJECTILES, ENEMY_FEATS, PROJ_FEATS


def _empty_obs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "enemies_feats": torch.zeros(batch, MAX_ENEMIES, ENEMY_FEATS),
        "enemies_mask": torch.zeros(batch, MAX_ENEMIES),
        "projectiles_feats": torch.zeros(batch, MAX_PROJECTILES, PROJ_FEATS),
        "projectiles_mask": torch.zeros(batch, MAX_PROJECTILES),
    }


def test_aux_labels_no_entities():
    """No enemies + no projectiles -> nearest distances at cap (2.0), count 0."""
    obs = _empty_obs(batch=1)
    labels = aux_labels_from_obs(obs)
    assert labels.shape == (1, 3)
    assert labels[0, 0].item() == 2.0    # nearest enemy dist capped
    assert labels[0, 1].item() == 0.0    # enemy count
    assert labels[0, 2].item() == 2.0    # nearest projectile dist capped


def test_aux_labels_one_enemy_dead_center():
    """Enemy at dx=dy=0: normalized dist = 0."""
    obs = _empty_obs(batch=1)
    obs["enemies_mask"][0, 0] = 1.0
    # feats[..., 2] = dx/480, feats[..., 3] = dy/270. Set both to 0.
    labels = aux_labels_from_obs(obs)
    assert labels[0, 0].item() < 0.01     # very close
    assert 0.04 < labels[0, 1].item() < 0.05   # 1/24 ~= 0.0417


def test_aux_labels_enemy_far():
    """Enemy at (dx=240, dy=135) -> norm dist = sqrt(0.5^2 + 0.5^2) = 0.707."""
    obs = _empty_obs(batch=1)
    obs["enemies_mask"][0, 0] = 1.0
    obs["enemies_feats"][0, 0, 2] = 0.5   # dx/480
    obs["enemies_feats"][0, 0, 3] = 0.5   # dy/270
    labels = aux_labels_from_obs(obs)
    assert abs(labels[0, 0].item() - 0.7071) < 0.01


def test_aux_labels_picks_nearest():
    """Multiple enemies — labels reflect the closest one."""
    obs = _empty_obs(batch=1)
    # Enemy 0: far
    obs["enemies_mask"][0, 0] = 1.0
    obs["enemies_feats"][0, 0, 2] = 1.0
    obs["enemies_feats"][0, 0, 3] = 0.0
    # Enemy 1: close
    obs["enemies_mask"][0, 1] = 1.0
    obs["enemies_feats"][0, 1, 2] = 0.1
    obs["enemies_feats"][0, 1, 3] = 0.0
    labels = aux_labels_from_obs(obs)
    assert abs(labels[0, 0].item() - 0.1) < 0.01
    assert 0.08 < labels[0, 1].item() < 0.09    # 2/24 ~= 0.0833


def test_aux_labels_batched():
    """Sanity: batching works."""
    obs = _empty_obs(batch=4)
    obs["enemies_mask"][0, 0] = 1.0    # env 0 has an enemy
    obs["enemies_feats"][0, 0, 2] = 0.5
    labels = aux_labels_from_obs(obs)
    assert labels.shape == (4, 3)
    assert labels[0, 0].item() < 2.0        # env 0 has visible enemy
    assert labels[1, 0].item() == 2.0       # env 1 has none


# ---- RunningMeanStd ---------------------------------------------------------


def test_running_mean_std_matches_numpy():
    """Basic sanity: after enough updates, running stats approximate ground truth."""
    rms = RunningMeanStd()
    data = np.random.randn(10000).astype(np.float32) * 5.0 + 3.0
    for i in range(0, 10000, 100):
        rms.update(data[i:i + 100])
    assert abs(rms.mean - 3.0) < 0.5
    assert abs(rms.std - 5.0) < 0.5


def test_running_mean_std_handles_torch_input():
    rms = RunningMeanStd()
    rms.update(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert abs(rms.mean - 3.0) < 0.01
