"""Sanity checks that don't need a live Isaac.

Run:
    PYTHONPATH=python pytest tests/
"""
from __future__ import annotations

import io
import socket
import threading

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
)


def test_zero_obs_matches_space():
    space = observation_space()
    obs = zero_obs()
    assert space.contains(obs), "zero_obs() must satisfy observation_space()"


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
    }
    obs = encode_obs(raw, last_action=np.array([1, 2, 0, 0, 0], dtype=np.int64))
    assert obs["player"][0] == pytest.approx(320.0)
    assert obs["player"][4] == pytest.approx(6.0)  # hp_red
    assert obs["global"][0] == pytest.approx(1.0)  # stage
    # last_action normalized: move=1/(9-1)=0.125, shoot=2/(5-1)=0.5
    assert obs["last_action"][0] == pytest.approx(0.125)
    assert obs["last_action"][1] == pytest.approx(0.5)


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
