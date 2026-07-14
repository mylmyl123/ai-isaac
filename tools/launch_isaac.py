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
    ap.add_argument("--stage0", action="store_true",
                    help="Enable the Stage-0 curriculum in the mod "
                         "(sets ISAAC_RL_STAGE0=1). Every new room becomes "
                         "'one fly, no other enemies' — use with the "
                         "stage0_one_fly_xs.yaml training config.")
    args = ap.parse_args()

    env = os.environ.copy()
    env["ISAAC_RL_PORT"] = str(args.port)
    if args.stage0:
        env["ISAAC_RL_STAGE0"] = "1"
        print("  ISAAC_RL_STAGE0=1 (Stage-0 curriculum: one fly per room)")

    if args.binary:
        cmd = [args.binary, "--luadebug", *args.extra_arg]
        # Isaac reads its resources with paths relative to CWD (resources/
        # scripts/enums.lua, packed/*.a, etc). If we launch from the caller's
        # shell cwd, Isaac will fail with 'cannot open resources/scripts/
        # enums.lua: No such file or directory' and exit within a second.
        # Set cwd to the binary's parent so Isaac's asset lookup works.
        launch_cwd = str(Path(args.binary).resolve().parent)
    else:
        steam = _find_steam()
        if not steam:
            print("error: could not locate Steam. Pass --binary <path-to-isaac> instead.", file=sys.stderr)
            return 2
        cmd = [steam, "-applaunch", str(STEAM_APP_ID), "--luadebug", *args.extra_arg]
        launch_cwd = None   # Steam sets cwd correctly on its own

    print("launching:", " ".join(cmd))
    print(f"  ISAAC_RL_PORT={args.port}")
    if launch_cwd:
        print(f"  cwd={launch_cwd}")
    proc = subprocess.Popen(cmd, env=env, cwd=launch_cwd)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
