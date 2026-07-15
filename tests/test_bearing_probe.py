"""Tests for the bearing probe + the representability fix (2026-07-15).

The corner-camp was diagnosed as the shoot head being unable to represent
'aim = f(enemy bearing)'. These tests pin:
  * the probe's scoring logic (correct_cardinal)
  * that a FRESH net is blind (constant shoot across bearings) — the failure
  * that after supervised fitting the FIXED net (normalization + aim feature)
    CAN aim — i.e. the architecture is now capable, which it measurably wasn't.

Run:
    PYTHONPATH=python pytest tests/test_bearing_probe.py -q
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from isaac_rl.cleanrl_ppo import ActorCritic

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bearing_probe", _REPO / "tools" / "bearing_probe.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)


def test_correct_cardinal_scoring():
    # 1=up 2=right 3=down 4=left; Isaac +y is down.
    assert bp.correct_cardinal(100, 0) == 2
    assert bp.correct_cardinal(-100, 0) == 4
    assert bp.correct_cardinal(0, 100) == 3
    assert bp.correct_cardinal(0, -100) == 1
    assert bp.correct_cardinal(100, 10) == 2   # dominant axis = x
    assert bp.correct_cardinal(10, -100) == 1  # dominant axis = y


def test_build_obs_encodes_enemy_bearing():
    # Enemy at 45deg should read isotropically in the aim feature (spatial 8,9).
    flat, dx, dy = bp.build_obs(math.radians(45), dist_px=140)
    assert dx > 0 and dy > 0
    assert flat.shape[0] > 2600   # includes the 3 new aim dims


def _probe_net(net, n=24, dist=200.0):
    net.eval()
    correct = 0
    distinct = set()
    with torch.no_grad():
        for i in range(n):
            th = 2 * math.pi * i / n
            flat, dx, dy = bp.build_obs(th, dist)
            s = int(net.forward(torch.tensor(flat).unsqueeze(0))[0][1].logits.argmax())
            distinct.add(s)
            correct += (s == bp.correct_cardinal(dx, dy))
    return correct, distinct


def test_fresh_net_is_blind():
    """A fresh (untrained) net must be blind — constant shoot across bearings,
    ~chance accuracy. This is the failure signature the probe detects."""
    torch.manual_seed(0)
    sample, _, _ = bp.build_obs(0.0)
    net = ActorCritic(sample.shape[0], 256, 2, active_factors=2, normalize_obs=True)
    correct, distinct = _probe_net(net)
    assert correct <= 10, f"fresh net should be ~chance, got {correct}/24"


def test_fixed_net_can_learn_to_aim():
    """After the fix (normalization + explicit aim feature), the architecture
    must be CAPABLE of representing aim = f(bearing): supervised-fit it to the
    correct cardinal and the probe should pass. This is the representability
    guarantee the reward could never provide on the old (blind) obs."""
    torch.manual_seed(0); np.random.seed(0)
    sample, _, _ = bp.build_obs(0.0)
    obs_dim = sample.shape[0]
    net = ActorCritic(obs_dim, 256, 2, active_factors=2, normalize_obs=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X, Y = [], []
    for _ in range(1500):
        th = np.random.uniform(0, 2 * math.pi)
        flat, dx, dy = bp.build_obs(th, dist_px=np.random.uniform(120, 300))
        X.append(flat); Y.append(bp.correct_cardinal(dx, dy))
    X = torch.tensor(np.array(X)); Y = torch.tensor(Y)
    net.update_norm(X)
    for _ in range(250):
        dists, _ = net.forward(X)
        loss = torch.nn.functional.cross_entropy(dists[1].logits, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    correct, distinct = _probe_net(net)
    assert correct >= 18, f"fixed net should learn to aim (>=18/24), got {correct}/24"
    assert len(distinct) >= 3, f"aiming uses multiple directions, got {distinct}"
