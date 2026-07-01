"""Standalone Isaac launch tester.

Purpose: figure out *why* Isaac closes right after opening, independent of any
of the training machinery.

Usage:
    python tools\test_launch.py --isaac "C:\\Program Files (x86)\\Steam\\steamapps\\common\\The Binding of Isaac Rebirth\\isaac-ng.exe"

The tool tries several launch strategies in sequence, waiting up to 10 seconds
each and reporting whether Isaac stays alive. Whichever strategy works, use
that pattern in train.py's IsaacFleet.spawn().

Prints Isaac's log.txt tail after each attempt so you can see WHY it exited.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def log_paths_for_strategy(strategy: str, isaac_dir: Path) -> list[Path]:
    """Return likely locations where Isaac may have written log.txt."""
    return [
        # Standard user documents path
        Path.home() / "Documents" / "My Games" / "Binding of Isaac Repentance" / "log.txt",
        # Per-instance cwd (strategies 2 and 3)
        Path.cwd() / ".isaac-test" / strategy / "log.txt",
        # Right next to the binary
        isaac_dir / "log.txt",
    ]


def print_recent_log(paths: list[Path], since: float) -> None:
    print("\n  --- checking log files ---")
    for p in paths:
        if not p.exists():
            print(f"  [absent]  {p}")
            continue
        mtime = p.stat().st_mtime
        if mtime < since:
            print(f"  [stale]   {p}  (mtime {mtime - since:+.1f}s vs launch)")
            continue
        print(f"  [FRESH]   {p}")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"           could not read: {e}")
            continue
        tail = "\n".join(text.splitlines()[-40:])
        print(f"           last 40 lines:\n{tail}\n")


def try_strategy(name: str, cmd: list[str], cwd: Path | None, env: dict, wait_s: float, isaac_dir: Path) -> bool:
    print(f"\n=== strategy: {name} ===")
    print(f"cwd  = {cwd}")
    print(f"cmd  = {' '.join(cmd)}")
    print(f"env  ISAAC_RL_PORT={env.get('ISAAC_RL_PORT')}  SteamAppId={env.get('SteamAppId', '<unset>')}")

    launched_at = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0,
        )
    except FileNotFoundError as e:
        print(f"  FAILED: {e}")
        return False

    print(f"  pid = {proc.pid}, waiting {wait_s}s to see if it stays alive...")
    time.sleep(wait_s)

    if proc.poll() is None:
        print(f"  ✓ ALIVE after {wait_s}s. Terminating for cleanup.")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        alive = True
    else:
        print(f"  ✗ EXITED with return code {proc.returncode} in less than {wait_s}s")
        alive = False

    print_recent_log(log_paths_for_strategy(name, isaac_dir), since=launched_at)
    return alive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isaac", required=True, help="Full path to isaac-ng.exe")
    ap.add_argument("--wait", type=float, default=8.0, help="Seconds to observe each attempt (default 8)")
    args = ap.parse_args()

    binary = Path(args.isaac)
    if not binary.exists():
        print(f"error: {binary} does not exist", file=sys.stderr)
        return 2
    isaac_dir = binary.parent

    # Ensure test dirs exist and have steam_appid.txt.
    for name in ("strat1_bare", "strat2_cwd_with_appid", "strat3_cwd_no_appid"):
        d = Path.cwd() / ".isaac-test" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "steam_appid.txt").write_text("250900\n", encoding="utf-8")

    results: dict[str, bool] = {}

    # Strategy 1: launch as-is, no cwd change, no env changes, no --set-stage.
    env = os.environ.copy()
    env["ISAAC_RL_PORT"] = "9500"
    results["1_bare"] = try_strategy(
        "1_bare",
        [str(binary), "--luadebug"],
        cwd=isaac_dir,
        env=env,
        wait_s=args.wait,
        isaac_dir=isaac_dir,
    )

    # Strategy 2: keep cwd = Isaac install dir (assets load correctly) and add steam_appid.txt there.
    # This is the same as strategy 1 with the addition of steam_appid.txt right next to the binary.
    appid_next_to_bin = isaac_dir / "steam_appid.txt"
    already_existed = appid_next_to_bin.exists()
    try:
        if not already_existed:
            appid_next_to_bin.write_text("250900\n", encoding="utf-8")
        results["2_with_appid"] = try_strategy(
            "2_with_appid",
            [str(binary), "--luadebug"],
            cwd=isaac_dir,
            env=env,
            wait_s=args.wait,
            isaac_dir=isaac_dir,
        )
    finally:
        if not already_existed and appid_next_to_bin.exists():
            appid_next_to_bin.unlink()

    # Strategy 3: per-instance cwd (what train.py currently does).
    per_cwd = Path.cwd() / ".isaac-test" / "strat3_cwd_no_appid"
    (per_cwd / "steam_appid.txt").unlink(missing_ok=True)
    results["3_per_cwd_no_appid"] = try_strategy(
        "3_per_cwd_no_appid",
        [str(binary), "--luadebug"],
        cwd=per_cwd,
        env=env,
        wait_s=args.wait,
        isaac_dir=isaac_dir,
    )

    # Strategy 4: per-instance cwd WITH steam_appid.txt.
    per_cwd2 = Path.cwd() / ".isaac-test" / "strat2_cwd_with_appid"
    (per_cwd2 / "steam_appid.txt").write_text("250900\n", encoding="utf-8")
    results["4_per_cwd_with_appid"] = try_strategy(
        "4_per_cwd_with_appid",
        [str(binary), "--luadebug"],
        cwd=per_cwd2,
        env=env,
        wait_s=args.wait,
        isaac_dir=isaac_dir,
    )

    # Strategy 5: --set-stage 1 on top of strategy 2 (the "safest" cwd).
    try:
        if not already_existed:
            appid_next_to_bin.write_text("250900\n", encoding="utf-8")
        results["5_with_set_stage"] = try_strategy(
            "5_with_set_stage",
            [str(binary), "--luadebug", "--set-stage", "1"],
            cwd=isaac_dir,
            env=env,
            wait_s=args.wait,
            isaac_dir=isaac_dir,
        )
    finally:
        if not already_existed and appid_next_to_bin.exists():
            appid_next_to_bin.unlink()

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  strategy {name}")

    winners = [n for n, ok in results.items() if ok]
    print()
    if winners:
        print(f"WORKING: {', '.join(winners)}")
    else:
        print("NONE of the strategies kept Isaac alive. Read the log tails above —")
        print("Isaac writes its termination reason to log.txt right before exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
