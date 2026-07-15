"""CNN bearing probe — verify the NEW CNN architecture can AIM (2026-07-15).

Same idea as tools/bearing_probe.py but for the egocentric-grid CNN
(CNNActorCritic): rasterize ONE enemy into the 14x21x21 grid at N bearings
around the player, and check whether the shoot head tracks the enemy.

  * BLIND (the flat-MLP failure): argmax(shoot) constant across bearings -> ~chance.
  * AIMING (the goal): argmax(shoot) rotates with the enemy -> most bearings correct.

Two modes:
  python tools/cnn_bearing_probe.py                 # fresh net (should be ~chance)
  python tools/cnn_bearing_probe.py --fit           # supervised-fit then probe
                                                     #   -> proves the ARCHITECTURE
                                                     #      can represent aiming
  python tools/cnn_bearing_probe.py <checkpoint.pt> # probe a trained checkpoint

PASS: >= 18/24 bearings pick the correct cardinal. Exit 0 pass / 1 fail.
This is the pre-run GATE: the architecture must pass (--fit mode) before we
add recurrence or spend any multi-hour training run.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from isaac_rl.cleanrl_ppo import CNNActorCritic
from isaac_rl.spaces import EGO_CHANNELS, EGO_GRID, SCALAR_DIM

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"
SHOOT_NAME = {0: "none", 1: "up", 2: "right", 3: "down", 4: "left"}

# MUST match mods/isaac-rl-bridge/obs.lua build_ego_grid.
EGO_CELL_PX = 16.0
EGO_CENTER = 10


def correct_cardinal(dx: float, dy: float) -> int:
    """Best cardinal to hit enemy at pixel offset (dx,dy). Isaac +x right, +y down.
    1=up 2=right 3=down 4=left."""
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1


def _cell(dx: float, dy: float):
    """world offset (dx,dy) -> (cx,cy) 0-based, same math as the Lua rasterizer."""
    cx = math.floor(dx / EGO_CELL_PX + 0.5) + EGO_CENTER
    cy = math.floor(dy / EGO_CELL_PX + 0.5) + EGO_CENTER
    return cx, cy


def build_grid_scalar(bearing_rad: float, dist_px: float = 120.0):
    """One enemy at the given bearing rasterized into the ego grid. Returns
    (grid[14,21,21], scalar[161], dx, dy). dist 120px keeps the enemy inside
    the 336px crop so it lands on a real cell."""
    grid = np.zeros((EGO_CHANNELS, EGO_GRID, EGO_GRID), dtype=np.float32)
    grid[0, EGO_CENTER, EGO_CENTER] = 1.0                    # ch0 player_self
    dx = math.cos(bearing_rad) * dist_px
    dy = math.sin(bearing_rad) * dist_px
    cx, cy = _cell(dx, dy)
    cx = min(max(cx, 0), EGO_GRID - 1); cy = min(max(cy, 0), EGO_GRID - 1)
    grid[1, cy, cx] = 1.0                                    # ch1 enemy_presence
    grid[2, cy, cx] = 1.0                                    # ch2 enemy_hp_frac (full hp)
    scalar = np.zeros(SCALAR_DIM, dtype=np.float32)          # scalars irrelevant for aim
    return grid, scalar, dx, dy


def _probe(net, n=24, dist=120.0):
    net.eval()
    correct = 0
    distinct = set()
    rows = []
    with torch.no_grad():
        for i in range(n):
            th = 2 * math.pi * i / n
            g, s, dx, dy = build_grid_scalar(th, dist)
            gt = torch.from_numpy(g).unsqueeze(0)
            st = torch.from_numpy(s).unsqueeze(0)
            shoot = int(net.forward(gt, st)[0][1].logits.argmax().item())
            want = correct_cardinal(dx, dy)
            distinct.add(shoot); correct += (shoot == want)
            rows.append((math.degrees(th), shoot, want))
    return correct, distinct, rows


def supervised_fit(net, steps=300, dist_range=(80.0, 160.0)):
    """Fit the net to shoot the correct cardinal at random bearings. Tests
    whether the ARCHITECTURE can represent aim=f(bearing) — the thing the flat
    MLP could not."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    G, S, Y = [], [], []
    for _ in range(2000):
        th = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(*dist_range)
        g, s, dx, dy = build_grid_scalar(th, d)
        G.append(g); S.append(s); Y.append(correct_cardinal(dx, dy))
    G = torch.from_numpy(np.array(G)); S = torch.from_numpy(np.array(S)); Y = torch.tensor(Y)
    net.update_norm(S)
    net.train()
    for _ in range(steps):
        dists, _ = net.forward(G, S)
        loss = torch.nn.functional.cross_entropy(dists[1].logits, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--fit", action="store_true", help="supervised-fit before probing (architecture test)")
    ap.add_argument("--n-bearings", type=int, default=24)
    ap.add_argument("--dist", type=float, default=120.0)
    ap.add_argument("--pass-frac", type=float, default=0.75)
    args = ap.parse_args()

    net = CNNActorCritic(hidden_dim=256, active_factors=2, normalize_obs=True)
    src = "FRESH net"
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "net" not in ck:
            print(f"{RED}[FAIL]{RESET} no 'net' key in checkpoint"); return 1
        w = ck["net"].get("cnn.0.weight")
        if w is not None and w.shape[1] != EGO_CHANNELS:
            print(f"{RED}[FAIL]{RESET} checkpoint CNN in_channels {w.shape[1]} != {EGO_CHANNELS}"); return 1
        net.load_state_dict(ck["net"], strict=False)
        src = args.checkpoint
    if args.fit:
        loss = supervised_fit(net)
        src += f" (supervised-fit, final loss {loss:.3f})"

    print()
    print(f"==== CNN bearing probe: {src} ====")
    correct, distinct, rows = _probe(net, args.n_bearings, args.dist)
    for deg, shoot, want in rows:
        ok = shoot == want
        print(f"{deg:8.0f}  shoot={SHOOT_NAME.get(shoot,shoot):>6}  want={SHOOT_NAME.get(want,want):>6}  "
              f"{(GREEN+'ok'+RESET) if ok else (RED+'x'+RESET)}")
    frac = correct / args.n_bearings
    print(f"\ncorrect: {correct}/{args.n_bearings} ({100*frac:.0f}%)  "
          f"distinct shoot outputs: {[SHOOT_NAME.get(c,c) for c in sorted(distinct)]}")
    if len(distinct) == 1:
        print(f"{RED}BLIND{RESET}: shoot constant across bearings.")
    if frac >= args.pass_frac:
        print(f"{GREEN}PASS{RESET}: CNN shoot head tracks enemy bearing (>= {int(args.pass_frac*100)}%).")
        return 0
    print(f"{RED}FAIL{RESET}: does not track bearing (< {int(args.pass_frac*100)}%).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
