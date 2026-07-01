"""Offline sanity checks — do not need a live Isaac.

Run:
    PYTHONPATH=python pytest tests/
"""
from __future__ import annotations

import socket

import numpy as np
import pytest

from isaac_rl.protocol import recv_frame, send_frame
from isaac_rl.spaces import (
    ACTION_FACTORS,
    action_space,
    encode_action,
    encode_obs,
    observation_space,
    zero_obs,
    MAX_ENEMIES,
    ENEMY_FEATS,
    ROOM_H,
    ROOM_W,
)


def test_zero_obs_matches_space():
    space = observation_space()
    obs = zero_obs()
    assert space.contains(obs)


def test_encode_action_shape():
    a = np.array([3, 2, 1, 0, 1], dtype=np.int64)
    d = encode_action(a)
    assert d == {"move": 3, "shoot": 2, "use_active": 1, "drop_bomb": 0, "pill_card": 1}


def test_action_space_shape():
    space = action_space()
    assert tuple(space.nvec.tolist()) == tuple(ACTION_FACTORS.tolist())


def test_encode_obs_from_lua_payload():
    raw = {
        "schema": 1,
        "tick": 42,
        "player": {"x": 320.0, "y": 280.0, "hp_red": 6, "hp_soul": 2, "damage": 3.5},
        "global": {"stage": 1, "room_type": 1, "is_clear": False},
        "passives": [1, 5, 12],
        "doors": [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
    }
    obs = encode_obs(raw, last_action=np.array([1, 2, 0, 0, 0], dtype=np.int64))
    assert obs["player"][0] == pytest.approx(320.0)
    assert obs["player"][4] == pytest.approx(6.0)     # hp_red
    assert obs["global"][0] == pytest.approx(1.0)     # stage
    # Passives: indices 1, 5, 12 in Lua -> 0, 4, 11 in numpy (0-based).
    assert obs["passives"][0] == 1
    assert obs["passives"][4] == 1
    assert obs["passives"][11] == 1
    assert obs["passives"][3] == 0
    # last_action normalized: move=1/(9-1)=0.125, shoot=2/(5-1)=0.5
    assert obs["last_action"][0] == pytest.approx(0.125)
    assert obs["last_action"][1] == pytest.approx(0.5)
    # Doors decoded.
    assert obs["doors"][0, 0] == 1.0
    assert obs["doors"][2, 3] == 1.0


def test_encode_obs_entities_padded_and_masked():
    raw = {
        "player": {},
        "global": {},
        "enemies": {
            "feats": [
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 3, 0, 20, 1.0, 5, 0],
                [0.6, 0.6, 0.1, 0.1, 0.0, 0.0, 0.8, 0, 0, 0, 3, 0, 20, 1.0, 5, 0],
            ],
            "mask": [1, 1],
            "count": 2,
        },
    }
    obs = encode_obs(raw)
    assert obs["enemies"]["feats"].shape == (MAX_ENEMIES, ENEMY_FEATS)
    assert obs["enemies"]["mask"].shape == (MAX_ENEMIES,)
    assert obs["enemies"]["mask"][:2].sum() == 2
    assert obs["enemies"]["mask"][2:].sum() == 0
    # Enemy 0 features filled.
    assert obs["enemies"]["feats"][0, 0] == pytest.approx(0.5)
    # Padding is zero.
    assert obs["enemies"]["feats"][5, :].sum() == 0


def test_encode_obs_room_grid_flat_to_2d():
    walls = [0] * (ROOM_H * ROOM_W)
    walls[0] = 1
    walls[ROOM_H * ROOM_W - 1] = 1
    raw = {
        "player": {},
        "global": {},
        "room_grid": {"walls": walls, "rocks": [], "spikes": [], "poop": []},
    }
    obs = encode_obs(raw)
    assert obs["room_grid"].shape == (4, ROOM_H, ROOM_W)
    assert obs["room_grid"][0, 0, 0] == 1.0
    assert obs["room_grid"][0, ROOM_H - 1, ROOM_W - 1] == 1.0


def test_protocol_roundtrip_over_socketpair():
    a, b = socket.socketpair()
    try:
        payload = {"hello": True, "seed": 12345, "nested": {"x": 1.0, "y": [1, 2, 3]}}
        send_frame(a, payload)
        got = recv_frame(b)
        assert got == payload
    finally:
        a.close()
        b.close()


def test_protocol_roundtrip_multiple_frames():
    a, b = socket.socketpair()
    try:
        for i in range(5):
            send_frame(a, {"i": i, "action": [i % 9, i % 5, 0, 0, 0]})
        for i in range(5):
            got = recv_frame(b)
            assert got["i"] == i
    finally:
        a.close()
        b.close()
