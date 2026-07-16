"""CNN bearing probe (full-room tensor, 2026-07-15 v2).

Verify the full-room CNN can AIM: rasterize ONE enemy at its TRUE room position
into the (14,34,60) room tensor, at N bearings AND at realistic spawn distances
(200-500px — not just close range, which was the blind spot that let the old
±168px-crop probe pass falsely). Check whether the shoot head tracks the enemy.

  python tools/cnn_bearing_probe.py            # fresh net (~chance)
  python tools/cnn_bearing_probe.py --fit      # supervised-fit then probe (arch gate)
  python tools/cnn_bearing_probe.py ckpt.pt    # probe a trained checkpoint

PASS: >= 75% of (bearing x distance) samples pick the correct cardinal.
Exit 0 pass / 1 fail. Pre-run GATE before recurrence / any real run.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from isaac_rl.cleanrl_ppo import CNNActorCritic
from isaac_rl.spaces import ROOM_TENSOR_C, ROOM_TENSOR_H, ROOM_TENSOR_W, SCALAR_DIM

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"
SHOOT_NAME = {0: "none", 1: "up", 2: "right", 3: "down", 4: "left"}

# MUST match mods/isaac-rl-bridge/obs.lua build_room_tensor.
ROOM_W_PX, ROOM_H_PX = 480.0, 270.0          # room interior world-px
RT_CELL_PX = 8.0
# Player fixed at room center for the probe; enemy placed at player+offset,
# clamped to stay inside the room so it always rasterizes to a real cell.
PLAYER_XY = (ROOM_W_PX / 2.0, ROOM_H_PX / 2.0)


def correct_cardinal(dx: float, dy: float) -> int:
    """Best cardinal for enemy at pixel offset (dx,dy). Isaac +x right, +y down.
    1=up 2=right 3=down 4=left."""
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1


def _cell(x: float, y: float):
    cx = int(math.floor(x / RT_CELL_PX)); cy = int(math.floor(y / RT_CELL_PX))
    cx = min(max(cx, 0), ROOM_TENSOR_W - 1); cy = min(max(cy, 0), ROOM_TENSOR_H - 1)
    return cx, cy


def build_grid_scalar(bearing_rad: float, dist_px: float):
    """Player at room center, one enemy at bearing+distance, rasterized into the
    full-room tensor. Returns (grid[14,34,60], scalar[39], dx, dy)."""
    grid = np.zeros((ROOM_TENSOR_C, ROOM_TENSOR_H, ROOM_TENSOR_W), dtype=np.float32)
    px, py = PLAYER_XY
    pcx, pcy = _cell(px, py)
    grid[0, pcy, pcx] = 1.0                                   # ch0 player_self
    dx = math.cos(bearing_rad) * dist_px
    dy = math.sin(bearing_rad) * dist_px
    ex = min(max(px + dx, 0.0), ROOM_W_PX)                    # keep enemy in-room
    ey = min(max(py + dy, 0.0), ROOM_H_PX)
    # recompute the ACTUAL offset after clamping (so the label matches the cell)
    dx, dy = ex - px, ey - py
    ecx, ecy = _cell(ex, ey)
    grid[1, ecy, ecx] = 1.0                                   # ch1 enemy_presence
    grid[2, ecy, ecx] = 1.0                                   # ch2 enemy_hp_frac
    scalar = np.zeros(SCALAR_DIM, dtype=np.float32)
    return grid, scalar, dx, dy


def _probe(net, n_bearings=24, dists=(120.0, 200.0, 320.0)):
    net.eval()
    correct = total = 0
    distinct = set()
    rows = []
    with torch.no_grad():
        for dist in dists:
            for i in range(n_bearings):
                th = 2 * math.pi * i / n_bearings
                g, s, dx, dy = build_grid_scalar(th, dist)
                shoot = int(net.forward(torch.from_numpy(g).unsqueeze(0),
                                        torch.from_numpy(s).unsqueeze(0))[0][1].logits.argmax().item())
                want = correct_cardinal(dx, dy)
                distinct.add(shoot); correct += (shoot == want); total += 1
                rows.append((dist, math.degrees(th), shoot, want))
    return correct, total, distinct, rows


def supervised_fit(net, steps=300, dist_range=(80.0, 420.0)):
    """Fit to shoot the correct cardinal at random bearings AND distances across
    the realistic spawn range. Tests whether the ARCHITECTURE can represent
    aim=f(enemy position) at any distance — the property the crop lacked."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    G, S, Y = [], [], []
    for _ in range(3000):
        th = rng.uniform(0, 2 * math.pi); d = rng.uniform(*dist_range)
        g, s, dx, dy = build_grid_scalar(th, d)
        G.append(g); S.append(s); Y.append(correct_cardinal(dx, dy))
    G = torch.from_numpy(np.array(G)); S = torch.from_numpy(np.array(S)); Y = torch.tensor(Y)
    net.update_norm(S); net.train()
    for _ in range(steps):
        dists, _ = net.forward(G, S)
        loss = torch.nn.functional.cross_entropy(dists[1].logits, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--n-bearings", type=int, default=24)
    ap.add_argument("--pass-frac", type=float, default=0.75)
    args = ap.parse_args()

    net = CNNActorCritic(hidden_dim=256, active_factors=2, normalize_obs=True)
    src = "FRESH net"
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "net" not in ck:
            print(f"{RED}[FAIL]{RESET} no 'net' key in checkpoint"); return 1
        w = ck["net"].get("cnn.0.weight")
        if w is not None and w.shape[1] != ROOM_TENSOR_C:
            print(f"{RED}[FAIL]{RESET} checkpoint CNN in_channels {w.shape[1]} != {ROOM_TENSOR_C}"); return 1
        net.load_state_dict(ck["net"], strict=False); src = args.checkpoint
    if args.fit:
        loss = supervised_fit(net); src += f" (supervised-fit, final loss {loss:.3f})"

    print(f"\n==== CNN bearing probe (full-room tensor): {src} ====")
    # Distances span close to far — the 320px case is the one the old crop failed.
    correct, total, distinct, rows = _probe(net, args.n_bearings, dists=(120.0, 200.0, 320.0))
    # per-distance breakdown
    per = {}
    for dist, deg, shoot, want in rows:
        per.setdefault(dist, [0, 0]); per[dist][1] += 1; per[dist][0] += (shoot == want)
    for dist, (c, t) in sorted(per.items()):
        print(f"  dist {dist:5.0f}px: {c}/{t} correct ({100*c/t:.0f}%)")
    frac = correct / total
    print(f"\ntotal: {correct}/{total} ({100*frac:.0f}%)  distinct shoot: {[SHOOT_NAME.get(c,c) for c in sorted(distinct)]}")
    if len(distinct) == 1:
        print(f"{RED}BLIND{RESET}: shoot constant across all bearings/distances.")
    if frac >= args.pass_frac:
        print(f"{GREEN}PASS{RESET}: CNN aims across bearings AND distances (>= {int(args.pass_frac*100)}%).")
        return 0
    print(f"{RED}FAIL{RESET}: does not aim (< {int(args.pass_frac*100)}%).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
