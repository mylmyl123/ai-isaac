"""Disable / re-enable every Isaac mod except the RL bridge.

Isaac's per-mod enabled state lives in a file called `disable.it` inside each
mod folder in the game's mods directory. If that file exists, the mod is
disabled; if absent, enabled. This script toggles those files.

Isaac reads mods from TWO locations, both of which we scan:
  1. Next to isaac-ng.exe — used for Steam Workshop subscriptions:
       <steam>/steamapps/common/The Binding of Isaac Rebirth/mods/
  2. Documents folder — used for hand-installed mods:
       %USERPROFILE%/Documents/My Games/Binding of Isaac Repentance/mods/

Pass --isaac <path-to-isaac-ng.exe> if the auto-detection can't find your
install (e.g. Steam on a non-standard drive).

Usage (PowerShell):

    # See what's currently enabled (across both locations):
    python tools\manage_mods.py list

    # Disable everything except isaac-rl-bridge (safe: leaves them installed,
    # you can flip back with `enable-all` after training):
    python tools\manage_mods.py disable-others

    # Restore everything you had before:
    python tools\manage_mods.py enable-all

    # If auto-detection fails, point at your Isaac binary:
    python tools\manage_mods.py list --isaac "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


BRIDGE_NAME = "isaac-rl-bridge"


# Same defaults as train.py / launch_isaac.py.
DEFAULT_ISAAC_BINARIES_WINDOWS = [
    r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"C:\Program Files\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"D:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"D:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
    r"E:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
]
DEFAULT_ISAAC_BINARIES_LINUX = [
    str(Path.home() / ".steam/steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng"),
    str(Path.home() / ".local/share/Steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng"),
]
DEFAULT_ISAAC_BINARIES_DARWIN = [
    str(Path.home() / "Library/Application Support/Steam/steamapps/common/The Binding of Isaac Rebirth/isaac-ng"),
]


def user_mods_dir() -> Path | None:
    system = platform.system()
    if system == "Windows":
        return Path.home() / "Documents" / "My Games" / "Binding of Isaac Repentance" / "mods"
    if system == "Linux":
        return Path.home() / ".local/share/binding of isaac repentance/mods"
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Binding of Isaac Repentance/mods"
    return None


def install_mods_dir(explicit_isaac: str | None) -> Path | None:
    """The `steamapps/common/.../mods` folder next to isaac-ng.exe."""
    candidates: list[str] = []
    if explicit_isaac:
        candidates.append(explicit_isaac)
    system = platform.system()
    candidates += {
        "Windows": DEFAULT_ISAAC_BINARIES_WINDOWS,
        "Linux": DEFAULT_ISAAC_BINARIES_LINUX,
        "Darwin": DEFAULT_ISAAC_BINARIES_DARWIN,
    }.get(system, [])
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p.parent / "mods"
    return None


def all_mods_roots(explicit_isaac: str | None) -> list[Path]:
    """Every mods folder Isaac reads from, in priority order."""
    out: list[Path] = []
    inst = install_mods_dir(explicit_isaac)
    if inst is not None and inst.is_dir():
        out.append(inst)
    user = user_mods_dir()
    if user is not None and user.is_dir():
        out.append(user)
    return out


def list_mods(root: Path) -> list[tuple[Path, bool, bool]]:
    """Return [(mod_dir, enabled, is_bridge), ...] for every mod present."""
    out = []
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        disabled = (entry / "disable.it").exists()
        is_bridge = entry.name == BRIDGE_NAME or BRIDGE_NAME in entry.name.lower()
        out.append((entry, not disabled, is_bridge))
    return out


def cmd_list(roots: list[Path]) -> int:
    total = 0
    enabled_total = 0
    for root in roots:
        mods = list_mods(root)
        if not mods:
            print(f"[{root}] no mods")
            continue
        print(f"[{root}] {len(mods)} mods")
        for path, enabled, is_bridge in mods:
            mark = "✓" if enabled else "·"
            tag = "  <-- RL BRIDGE" if is_bridge else ""
            print(f"  [{mark}] {path.name}{tag}")
        enabled_count = sum(1 for _, e, _ in mods if e)
        total += len(mods)
        enabled_total += enabled_count
        print(f"  subtotal: {enabled_count} enabled / {len(mods)} total")
        print()
    print(f"overall: {enabled_total} enabled / {total} total across {len(roots)} folder(s)")
    return 0


def cmd_disable_others(roots: list[Path]) -> int:
    disabled_now = 0
    kept = 0
    bridge_seen = False
    for root in roots:
        mods = list_mods(root)
        if not mods:
            continue
        print(f"[{root}]")
        for path, enabled, is_bridge in mods:
            if is_bridge:
                bridge_seen = True
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
    if not bridge_seen:
        print()
        print("WARNING: no isaac-rl-bridge mod was found in either location.")
        print("         Copy mods/isaac-rl-bridge/ into one of the mod folders above.")
    print("Launch Isaac once (through Steam or train.py) to have it pick up the change.")
    return 0


def cmd_enable_all(roots: list[Path]) -> int:
    reenabled = 0
    for root in roots:
        mods = list_mods(root)
        if not mods:
            continue
        print(f"[{root}]")
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
    ap.add_argument("--isaac", default=None,
                    help="Path to isaac-ng.exe (used to find the workshop mods folder next to it).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Show all installed mods and their enabled state.")
    sub.add_parser("disable-others",
                   help="Disable every mod except isaac-rl-bridge. Reversible via enable-all.")
    sub.add_parser("enable-all", help="Re-enable every mod that this script previously disabled.")
    args = ap.parse_args()

    roots = all_mods_roots(args.isaac)
    if not roots:
        print("error: could not find any mods folder.", file=sys.stderr)
        print("       Pass --isaac <path-to-isaac-ng.exe> so we can locate the workshop folder.", file=sys.stderr)
        return 2

    if args.cmd == "list":
        return cmd_list(roots)
    if args.cmd == "disable-others":
        return cmd_disable_others(roots)
    if args.cmd == "enable-all":
        return cmd_enable_all(roots)
    ap.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())

