"""Tests for the CNN architecture + its bearing probe (2026-07-15 rebuild).

The decisive test (test_cnn_can_learn_to_aim) is the architecture gate: the
egocentric-grid CNN must be CAPABLE of representing aim=f(enemy bearing), which
the flat MLP measurably could not (it maxed at chance even after fitting). This
guards the whole rebuild — if it regresses, the architecture lost its reason to
exist.

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
from isaac_rl.spaces import EGO_CHANNELS, EGO_GRID, SCALAR_DIM

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cnn_probe", _REPO / "tools" / "cnn_bearing_probe.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_probe_cell_math_matches_lua_constants():
    # The Python probe must rasterize with the SAME constants as the Lua mod.
    assert cp.EGO_CELL_PX == 16.0 and cp.EGO_CENTER == 10


def test_correct_cardinal():
    assert cp.correct_cardinal(100, 0) == 2   # right
    assert cp.correct_cardinal(-100, 0) == 4  # left
    assert cp.correct_cardinal(0, 100) == 3   # down (isaac +y)
    assert cp.correct_cardinal(0, -100) == 1  # up


def test_build_grid_scalar_shapes_and_placement():
    g, s, dx, dy = cp.build_grid_scalar(math.radians(0), dist_px=100)
    assert g.shape == (EGO_CHANNELS, EGO_GRID, EGO_GRID)
    assert s.shape == (SCALAR_DIM,)
    # player self at center; exactly one enemy-presence cell.
    assert g[0, cp.EGO_CENTER, cp.EGO_CENTER] == 1.0
    assert g[1].sum() == 1.0
    # enemy to the right (dx>0) should be right of center.
    ys, xs = np.nonzero(g[1])
    assert xs[0] > cp.EGO_CENTER


def test_cnn_forward_shapes():
    net = CNNActorCritic(hidden_dim=256, active_factors=2)
    g = torch.randn(4, EGO_CHANNELS, EGO_GRID, EGO_GRID)
    s = torch.randn(4, SCALAR_DIM)
    dists, v = net.forward(g, s)
    assert len(dists) == 5 and dists[1].logits.shape == (4, 5) and v.shape == (4,)


def test_fresh_cnn_is_not_aiming():
    torch.manual_seed(0)
    net = CNNActorCritic(hidden_dim=256, active_factors=2)
    correct, distinct, _ = cp._probe(net)
    assert correct <= 12, f"fresh net should be ~chance, got {correct}/24"


def test_cnn_can_learn_to_aim():
    """THE ARCHITECTURE GATE: the CNN must be able to represent aim=f(bearing).
    The flat MLP could not (chance even after fitting); the CNN grid can."""
    torch.manual_seed(0); np.random.seed(0)
    net = CNNActorCritic(hidden_dim=256, active_factors=2)
    cp.supervised_fit(net, steps=250)
    correct, distinct, _ = cp._probe(net)
    assert correct >= 18, f"CNN must learn to aim (>=18/24), got {correct}/24"
    assert len(distinct) >= 3, f"aiming uses multiple directions, got {distinct}"
