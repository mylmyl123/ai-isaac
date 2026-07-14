"""Collect heuristic-policy demos overnight for BC pretraining.

Usage (Windows PowerShell):
    python tools\\collect_heuristic_demos.py `
        --isaac "C:\\Program Files (x86)\\Steam\\steamapps\\common\\The Binding of Isaac Rebirth\\isaac-ng.exe" `
        --n-envs 4 `
        --steps 2_000_000 `
        --out demos\\heuristic_2m.npz

Design notes:
    - Uses the same launch/vec_env stack as training, so if training works
      this works.
    - HeuristicPolicy v3 (see python/isaac_rl/heuristic.py) is <150 lines
      and covers combat + door-seeking + pickup-seeking. That's the "20th
      percentile scripted bot" corpus recommended in the 2026-07-13 review.
    - 2 million steps at 4 envs \u00d7 15 Hz is ~9.3 h of wall clock. Overnight.
    - Output .npz is BC-ready (matches the layout bc.bc_pretrain expects).

To run BC pretraining after the demos are collected, see
    python/isaac_rl/bc.py :: bc_pretrain
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add python/ to path so `import isaac_rl` works when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

from isaac_rl.bc import collect_demos                # noqa: E402
from isaac_rl.heuristic import HeuristicConfig, HeuristicPolicy  # noqa: E402
from isaac_rl.vec_env import build_vec_env           # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isaac", type=str, default=None,
                    help="Path to isaac-ng.exe. Omit to launch Isaac manually.")
    ap.add_argument("--n-envs", type=int, default=2,
                    help="Number of concurrent Isaac instances.")
    ap.add_argument("--base-port", type=int, default=9500)
    ap.add_argument("--steps", type=int, default=500_000,
                    help="Total demo steps to collect across all envs.")
    ap.add_argument("--out", type=str, default="demos/heuristic_demos.npz",
                    help="Output .npz path. Parent dirs created automatically.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Heuristic PRNG seed (affects door choice + sidestep direction).")
    ap.add_argument("--reset-stage", type=int, default=1,
                    help="Force Isaac to a given stage on episode reset. 1 = Basement 1.")
    ap.add_argument("--max-episode-steps", type=int, default=1800,
                    help="Truncate episodes at this many ticks (default ~2min at 15Hz).")
    args = ap.parse_args()

    launch_isaac = args.isaac is not None
    env = build_vec_env(
        n_envs=args.n_envs,
        base_port=args.base_port,
        reset_stage=args.reset_stage,
        max_episode_steps=args.max_episode_steps,
        isaac_binary=args.isaac,
        launch_isaac=launch_isaac,
    )

    policy = HeuristicPolicy(HeuristicConfig(seed=args.seed))
    try:
        out_path = collect_demos(env, policy, n_steps=args.steps, save_path=args.out)
        print(f"[demos] saved: {out_path}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
