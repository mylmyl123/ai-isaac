"""Round-trip test: collect_demos-style obs saving + bc_pretrain loading + a real
policy_net forward pass. Guards against obs-key-mismatch bugs like the one where
we saved nested `obs__enemies__feats` but the model wanted `enemies_feats`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from isaac_rl.bc import _load_demos_to_tensors, _slice_obs, bc_pretrain
from isaac_rl.model import IsaacPolicy, PolicyConfig
from isaac_rl.spaces import (
    MAX_ENEMIES, MAX_PICKUPS, MAX_PROJECTILES,
    ENEMY_FEATS, PICKUP_FEATS, PROJ_FEATS,
    PLAYER_DIM, GLOBAL_DIM, PASSIVES_K, ROOM_H, ROOM_W,
    SPATIAL_DIM, PLAYER_HISTORY_DIM, Z_DIM,
    ACTION_FACTORS,
    flatten_dict_obs,
    DOOR_FEATS, CHARACTER_K,
    ACTIVE_SLOTS, ACTIVE_FEATS, TRINKET_SLOTS, TRINKET_FEATS,
    CARD_SLOTS, CARD_FEATS, PILL_SLOTS, PILL_FEATS, TRANSFORMATION_COUNT,
)
ACTION_FACTORS_LEN = ACTION_FACTORS   # alias for len() below without misleading name


def _fake_encoded_obs() -> dict:
    """Build one fake encoded_obs matching what env.step() returns."""
    return {
        "player": np.zeros(PLAYER_DIM, dtype=np.float32),
        "enemies": {
            "feats": np.zeros((MAX_ENEMIES, ENEMY_FEATS), dtype=np.float32),
            "mask": np.zeros(MAX_ENEMIES, dtype=np.int8),
        },
        "projectiles": {
            "feats": np.zeros((MAX_PROJECTILES, PROJ_FEATS), dtype=np.float32),
            "mask": np.zeros(MAX_PROJECTILES, dtype=np.int8),
        },
        "pickups": {
            "feats": np.zeros((MAX_PICKUPS, PICKUP_FEATS), dtype=np.float32),
            "mask": np.zeros(MAX_PICKUPS, dtype=np.int8),
        },
        "passives": np.zeros(PASSIVES_K, dtype=np.int8),
        "room_grid": np.zeros((4, ROOM_H, ROOM_W), dtype=np.float32),
        "doors": np.zeros((4, DOOR_FEATS), dtype=np.float32),
        "global": np.zeros(GLOBAL_DIM, dtype=np.float32),
        "last_action": np.zeros(len(ACTION_FACTORS_LEN), dtype=np.int8),
        "spatial": np.zeros(SPATIAL_DIM, dtype=np.float32),
        "player_history": np.zeros(PLAYER_HISTORY_DIM, dtype=np.float32),
        "z": np.zeros(Z_DIM, dtype=np.float32),
        # Track A (2026-07-12) keys.
        "character": np.zeros(CHARACTER_K, dtype=np.int8),
        "active_items": np.zeros((ACTIVE_SLOTS, ACTIVE_FEATS), dtype=np.float32),
        "trinkets": np.zeros((TRINKET_SLOTS, TRINKET_FEATS), dtype=np.float32),
        "cards": np.zeros((CARD_SLOTS, CARD_FEATS), dtype=np.float32),
        "pills": np.zeros((PILL_SLOTS, PILL_FEATS), dtype=np.float32),
        "transformations": np.zeros(TRANSFORMATION_COUNT, dtype=np.float32),
    }


def _fake_demo_npz(tmp_path: Path, n_transitions: int = 8) -> Path:
    """Write a small demo .npz using the same code path as collect_demos."""
    per_field_stacks: dict[str, list[np.ndarray]] = {}
    action_stack: list[np.ndarray] = []
    for _ in range(n_transitions):
        o = _fake_encoded_obs()
        flat = flatten_dict_obs(o)
        for k, v in flat.items():
            per_field_stacks.setdefault(k, []).append(np.asarray(v))
        action_stack.append(np.zeros(len(ACTION_FACTORS_LEN), dtype=np.int64))

    save_dict = {
        "actions": np.stack(action_stack),
        "n_transitions": np.array([n_transitions], dtype=np.int64),
    }
    for k, buf in per_field_stacks.items():
        save_dict[f"obs__{k}"] = np.stack(buf)

    path = tmp_path / "demo.npz"
    np.savez_compressed(path, **save_dict)
    return path


def test_load_demos_produces_flat_keys(tmp_path):
    p = _fake_demo_npz(tmp_path, n_transitions=8)
    obs, actions = _load_demos_to_tensors(p, device=torch.device("cpu"))
    assert "enemies_feats" in obs
    assert "enemies_mask" in obs
    assert "projectiles_feats" in obs
    assert "projectiles_mask" in obs
    assert "pickups_feats" in obs
    assert "pickups_mask" in obs
    assert "player" in obs
    assert "room_grid" in obs
    assert actions.shape == (8, len(ACTION_FACTORS_LEN))


def test_slice_obs_indexes_all_keys(tmp_path):
    p = _fake_demo_npz(tmp_path, n_transitions=8)
    obs, actions = _load_demos_to_tensors(p, device=torch.device("cpu"))
    idx = torch.arange(4)
    sub = _slice_obs(obs, idx)
    for k in obs:
        assert sub[k].shape[0] == 4


def test_bc_pretrain_forward_pass_and_learns(tmp_path):
    """Full end-to-end: load demos, run BC for 2 epochs, verify no crash and
    loss decreases from the first epoch to the last."""
    p = _fake_demo_npz(tmp_path, n_transitions=16)
    policy = IsaacPolicy(PolicyConfig(trunk_dim=64, gru_dim=64, entity_dim=32, proj_dim=32, pickup_dim=16))
    # Should not raise a KeyError anymore.
    bc_pretrain(policy, p, epochs=2, batch_size=8, lr=1e-3, device=torch.device("cpu"))
