"""Disable / re-enable every Isaac mod except the RL bridge.

Isaac's per-mod enabled state lives in a file called `disable.it` inside each
mod folder in the game's mods directory. If that file exists, the mod is
disabled; if absent, enabled. This script toggles those files.

Usage (PowerShell):

    # See what's currently enabled:
    python tools\manage_mods.py list

    # Disable everything except isaac-rl-bridge (safe: leaves them installed,
    # you can flip back with `enable-all` after training):
    python tools\manage_mods.py disable-others

    # Restore everything you had before:
    python tools\manage_mods.py enable-all
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


BRIDGE_NAME = "isaac-rl-bridge"


def mods_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path.home() / "Documents" / "My Games" / "Binding of Isaac Repentance" / "mods"
    if system == "Linux":
        return Path.home() / ".local/share/binding of isaac repentance/mods"
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Binding of Isaac Repentance/mods"
    raise RuntimeError(f"unsupported OS: {system}")


def list_mods(mods_root: Path) -> list[tuple[Path, bool, bool]]:
    """Return [(mod_dir, enabled, is_bridge), ...] for every mod present."""
    out = []
    if not mods_root.is_dir():
        return out
    for entry in sorted(mods_root.iterdir()):
        if not entry.is_dir():
            continue
        # Ignore workshop lock files and the like.
        if entry.name.startswith("."):
            continue
        disabled = (entry / "disable.it").exists()
        is_bridge = entry.name == BRIDGE_NAME or BRIDGE_NAME in entry.name.lower()
        out.append((entry, not disabled, is_bridge))
    return out


def cmd_list(root: Path) -> int:
    mods = list_mods(root)
    if not mods:
        print(f"no mods found in {root}")
        return 0
    print(f"{len(mods)} mods in {root}")
    print()
    for path, enabled, is_bridge in mods:
        mark = "✓" if enabled else "·"
        tag = "  <-- RL BRIDGE" if is_bridge else ""
        print(f"  [{mark}] {path.name}{tag}")
    enabled_count = sum(1 for _, e, _ in mods if e)
    print()
    print(f"summary: {enabled_count} enabled / {len(mods)} total")
    return 0


def cmd_disable_others(root: Path) -> int:
    mods = list_mods(root)
    disabled_now = 0
    kept = 0
    for path, enabled, is_bridge in mods:
        if is_bridge:
            # Leave the bridge alone — make sure it's ENABLED.
            marker = path / "disable.it"
            if marker.exists():
                marker.unlink()
                print(f"  ENABLING {path.name}")
            else:
                print(f"  keeping enabled: {path.name}")
            kept += 1
            continue
        if enabled:
            (path / "disable.it").touch()
            print(f"  disabling: {path.name}")
            disabled_now += 1
        else:
            print(f"  already disabled: {path.name}")
    print()
    print(f"disabled {disabled_now}, kept {kept} bridge mod(s) enabled.")
    print("Launch Isaac once (through Steam or train.py) to have it pick up the change.")
    return 0


def cmd_enable_all(root: Path) -> int:
    mods = list_mods(root)
    reenabled = 0
    for path, enabled, _is_bridge in mods:
        marker = path / "disable.it"
        if marker.exists():
            marker.unlink()
            print(f"  enabling: {path.name}")
            reenabled += 1
    print()
    print(f"re-enabled {reenabled} mod(s).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Show all installed mods and their enabled state.")
    sub.add_parser("disable-others",
                   help="Disable every mod except isaac-rl-bridge. Reversible via enable-all.")
    sub.add_parser("enable-all", help="Re-enable every mod that this script previously disabled.")
    args = ap.parse_args()

    root = mods_dir()
    if not root.exists():
        print(f"error: mods folder does not exist: {root}", file=sys.stderr)
        return 2

    if args.cmd == "list":
        return cmd_list(root)
    if args.cmd == "disable-others":
        return cmd_disable_others(root)
    if args.cmd == "enable-all":
        return cmd_enable_all(root)
    ap.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
