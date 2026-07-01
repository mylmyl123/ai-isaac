"""Cross-platform launcher for Isaac with the RL bridge mod.

Steam route (all platforms with Steam installed):
    steam -applaunch 250900 --luadebug

Direct binary (Linux native or macOS):
    <install-dir>/isaac-ng --luadebug

We prefer the Steam URL scheme because it also loads workshop/Repentance DLC state.
Set ISAAC_RL_PORT before launching if you want a non-default port; the mod reads it.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


STEAM_APP_ID = 250900  # The Binding of Isaac: Rebirth (Repentance is a DLC on the same app)


def _find_steam() -> str | None:
    for candidate in ("steam", "steam.sh"):
        p = shutil.which(candidate)
        if p:
            return p
    if platform.system() == "Darwin":
        mac = Path("/Applications/Steam.app/Contents/MacOS/steam_osx")
        if mac.exists():
            return str(mac)
    if platform.system() == "Windows":
        for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
            if not root:
                continue
            p = Path(root) / "Steam" / "steam.exe"
            if p.exists():
                return str(p)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9500, help="Passed to the mod via ISAAC_RL_PORT")
    ap.add_argument("--binary", type=str, default=None,
                    help="Direct path to the Isaac binary; skips Steam.")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="Extra argument to append (repeatable).")
    args = ap.parse_args()

    env = os.environ.copy()
    env["ISAAC_RL_PORT"] = str(args.port)

    if args.binary:
        cmd = [args.binary, "--luadebug", *args.extra_arg]
    else:
        steam = _find_steam()
        if not steam:
            print("error: could not locate Steam. Pass --binary <path-to-isaac> instead.", file=sys.stderr)
            return 2
        cmd = [steam, "-applaunch", str(STEAM_APP_ID), "--luadebug", *args.extra_arg]

    print("launching:", " ".join(cmd))
    print(f"  ISAAC_RL_PORT={args.port}")
    proc = subprocess.Popen(cmd, env=env)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
