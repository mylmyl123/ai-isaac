"""Offline bearing probe — verify the shoot head AIMS, without a training run.

The corner-camping failure was diagnosed as the shoot head being blind to enemy
position (a raw frame counter saturated the Tanh trunk and swamped the enemy-
bearing signal). This probe measures that DIRECTLY on a checkpoint: it places a
single enemy at N bearings around the player, holds everything else fixed, and
records which cardinal the shoot head picks at each bearing.

  * BLIND policy (the failure): argmax(shoot) is the SAME direction at every
    bearing -> matches the correct cardinal ~1/4 of the time (chance).
  * AIMING policy (the goal): argmax(shoot) ROTATES with the enemy -> matches
    the correct cardinal most of the time.

Run this on a checkpoint BEFORE committing to a multi-hour training run:

    PYTHONPATH=python python tools/bearing_probe.py ckpts/latest_stage0_*.pt

PASS threshold: >= 18/24 bearings pick the correct cardinal (configurable).
Exit 0 = pass (shoot tracks enemy), 1 = fail (still blind).
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from isaac_rl.cleanrl_ppo import ActorCritic
from isaac_rl.spaces import ACTION_FACTORS, encode_obs, flatten_dict_obs

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"

# shoot factor: 0=none, 1=up, 2=right, 3=down, 4=left (mods/.../main.lua:437-440)
SHOOT_NAME = {0: "none", 1: "up", 2: "right", 3: "down", 4: "left"}


def _flat(o) -> np.ndarray:
    parts = []
    for k in sorted(flatten_dict_obs(o).keys()):
        parts.append(np.asarray(flatten_dict_obs(o)[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


def correct_cardinal(dx: float, dy: float) -> int:
    """Best single cardinal to hit an enemy at pixel offset (dx, dy).
    Isaac screen coords: +x right, +y down. 1=up 2=right 3=down 4=left."""
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 4
    return 3 if dy > 0 else 1


def build_obs(bearing_rad: float, dist_px: float = 200.0,
              room=(0.0, 0.0, 480.0, 270.0), player=(240.0, 135.0),
              frame_count: float = 3000.0) -> tuple[np.ndarray, float, float]:
    """A realistic Stage-0 obs with the enemy at the given bearing. Includes a
    large raw frame_count on purpose (the saturation source we're testing)."""
    tl_x, tl_y, br_x, br_y = room
    px, py = player
    dx = math.cos(bearing_rad) * dist_px
    dy = math.sin(bearing_rad) * dist_px
    ex, ey = px + dx, py + dy
    # enemies_feats: [nx, ny, rel_x/480, rel_y/270, ...16]
    efeat = [(ex - tl_x) / (br_x - tl_x), (ey - tl_y) / (br_y - tl_y),
             dx / 480.0, dy / 270.0, 0, 0, 1] + [0] * 9
    raw = {
        "player": {"x": px, "y": py, "hp_red": 3, "fire_delay": 10,
                   "fire_cooldown": 0, "frame_count": frame_count, "can_shoot": True},
        "global": {"frames_since_room": 500, "frames_since_hit": 500},
        "room_bounds": {"tl_x": tl_x, "tl_y": tl_y, "br_x": br_x, "br_y": br_y},
        "enemies": {"feats": [efeat], "mask": [1]},
    }
    return _flat(encode_obs(raw)), dx, dy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="Path to a .pt checkpoint. Omit to probe a FRESH (untrained) net.")
    ap.add_argument("--n-bearings", type=int, default=24)
    ap.add_argument("--dist", type=float, default=200.0, help="enemy distance in px")
    ap.add_argument("--pass-frac", type=float, default=0.75,
                    help="fraction of bearings that must pick the correct cardinal to PASS")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    args = ap.parse_args()

    # Determine obs_dim from a sample obs (must match current spaces.py).
    sample, _, _ = build_obs(0.0, args.dist)
    obs_dim = sample.shape[0]

    active_factors = 2  # stage 0/A/B: move + shoot
    normalize_obs = True
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg")
        hidden = getattr(cfg, "hidden_dim", args.hidden_dim)
        nlayers = getattr(cfg, "n_hidden_layers", args.n_layers)
        normalize_obs = getattr(cfg, "normalize_obs", True)
        net = ActorCritic(obs_dim, hidden, nlayers, active_factors=active_factors,
                          normalize_obs=normalize_obs)
        if "net" not in ckpt:
            print(f"{RED}[ FAIL ]{RESET} checkpoint has no 'net' key (keys: {list(ckpt.keys())}) "
                  f"— not a CleanRL-PPO checkpoint (old DreamerV3-era format?). Can't probe.")
            return 1
        sd = ckpt["net"]
        # Guard: an old checkpoint trained at a different obs_dim can't be probed
        # against the current schema. Report clearly instead of a cryptic error.
        w0 = sd.get("trunk.0.weight")
        if w0 is not None and w0.shape[1] != obs_dim:
            print(f"{RED}[ FAIL ]{RESET} checkpoint obs_dim {w0.shape[1]} != current schema {obs_dim} "
                  f"(schema changed since this checkpoint — retrain, can't probe old one)")
            return 1
        net.load_state_dict(sd, strict=False)
        src = args.checkpoint
    else:
        net = ActorCritic(obs_dim, args.hidden_dim, args.n_layers,
                          active_factors=active_factors, normalize_obs=normalize_obs)
        src = "FRESH (untrained) net"
    net.eval()

    print()
    print(f"==== bearing probe: {src} ====")
    print(f"obs_dim={obs_dim} normalize_obs={normalize_obs} n_bearings={args.n_bearings} dist={args.dist}px")
    print(f"{'bearing°':>8}  {'shoot(argmax)':>14}  {'correct':>8}  {'ok':>3}")

    correct = 0
    chosen = []
    with torch.no_grad():
        for i in range(args.n_bearings):
            theta = 2 * math.pi * i / args.n_bearings
            flat, dx, dy = build_obs(theta, args.dist)
            x = torch.from_numpy(flat).unsqueeze(0)
            dists, _ = net.forward(x)
            shoot = int(dists[1].logits.argmax(dim=-1).item())  # factor 1 = shoot
            want = correct_cardinal(dx, dy)
            ok = (shoot == want)
            correct += ok
            chosen.append(shoot)
            print(f"{math.degrees(theta):8.0f}  {SHOOT_NAME.get(shoot,shoot):>14}  "
                  f"{SHOOT_NAME.get(want,want):>8}  {(GREEN+'  ✓'+RESET) if ok else (RED+'  ✗'+RESET)}")

    frac = correct / args.n_bearings
    distinct = sorted(set(chosen))
    print()
    print(f"correct: {correct}/{args.n_bearings} ({100*frac:.0f}%)   "
          f"distinct shoot outputs across bearings: {[SHOOT_NAME.get(c,c) for c in distinct]}")
    if len(distinct) == 1:
        print(f"{RED}BLIND{RESET}: shoot is CONSTANT across all bearings — the policy ignores enemy position.")
    if frac >= args.pass_frac:
        print(f"{GREEN}PASS{RESET}: shoot head tracks enemy bearing (>= {int(args.pass_frac*100)}%). Representation OK.")
        return 0
    print(f"{RED}FAIL{RESET}: shoot head does not track enemy bearing (< {int(args.pass_frac*100)}%). Still blind.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
