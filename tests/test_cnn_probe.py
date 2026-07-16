"""Tests for the full-room CNN architecture + its bearing probe (2026-07-15 v2).

The decisive test (test_cnn_can_learn_to_aim) is the architecture gate: the
full-room-tensor CNN must be CAPABLE of representing aim=f(enemy position) at
ANY distance (the flat MLP couldn't at all; the earlier egocentric crop lost
far enemies). Guards the whole rebuild.

NOTE: the supervised-fit gate is compute-heavy; kept at modest steps.

Run:
    PYTHONPATH=python pytest tests/test_cnn_probe.py -q
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from isaac_rl.cleanrl_ppo import CNNActorCritic
from isaac_rl.spaces import ROOM_TENSOR_C, ROOM_TENSOR_H, ROOM_TENSOR_W, SCALAR_DIM

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cnn_probe", _REPO / "tools" / "cnn_bearing_probe.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_probe_constants_match_lua():
    assert cp.RT_CELL_PX == 8.0
    assert cp.ROOM_W_PX == 480.0 and cp.ROOM_H_PX == 270.0


def test_correct_cardinal():
    assert cp.correct_cardinal(100, 0) == 2
    assert cp.correct_cardinal(-100, 0) == 4
    assert cp.correct_cardinal(0, 100) == 3
    assert cp.correct_cardinal(0, -100) == 1


def test_build_grid_scalar_shapes_and_placement():
    g, s, dx, dy = cp.build_grid_scalar(math.radians(0), dist_px=100)
    assert g.shape == (ROOM_TENSOR_C, ROOM_TENSOR_H, ROOM_TENSOR_W)
    assert s.shape == (SCALAR_DIM,)
    assert g[0].sum() == 1.0   # exactly one player cell
    assert g[1].sum() == 1.0   # exactly one enemy cell
    # enemy to the right of player (dx>0)
    py_row, px_col = np.nonzero(g[0])
    ey_row, ex_col = np.nonzero(g[1])
    assert ex_col[0] > px_col[0]


def test_cnn_forward_shapes():
    net = CNNActorCritic(hidden_dim=256, active_factors=2)
    g = torch.randn(4, ROOM_TENSOR_C, ROOM_TENSOR_H, ROOM_TENSOR_W)
    s = torch.randn(4, SCALAR_DIM)
    dists, v = net.forward(g, s)
    assert len(dists) == 5 and dists[1].logits.shape == (4, 5) and v.shape == (4,)


def test_fresh_cnn_is_not_aiming():
    torch.manual_seed(0)
    net = CNNActorCritic(hidden_dim=256, active_factors=2)
    correct, total, distinct, _ = cp._probe(net, n_bearings=12, dists=(120.0, 320.0))
    assert correct <= 0.5 * total, f"fresh net should be ~chance, got {correct}/{total}"


def test_cnn_can_learn_to_aim():
    """ARCHITECTURE GATE: the full-room CNN must represent aim=f(position) at
    multiple distances. Modest fit steps to keep the test runnable."""
    torch.manual_seed(0); np.random.seed(0)
    net = CNNActorCritic(hidden_dim=256, active_factors=2)
    cp.supervised_fit(net, steps=200)
    correct, total, distinct, _ = cp._probe(net, n_bearings=12, dists=(120.0, 200.0, 320.0))
    assert correct >= 0.7 * total, f"CNN must learn to aim (>=70%), got {correct}/{total}"
    assert len(distinct) >= 3
