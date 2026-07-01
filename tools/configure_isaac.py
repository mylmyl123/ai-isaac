"""Patch Isaac's options.ini for training-friendly behavior.

The Isaac game options live at:

  Windows: %USERPROFILE%\\Documents\\My Games\\Binding of Isaac Repentance\\options.ini
  Linux:   ~/.local/share/binding of isaac repentance/options.ini
  macOS:   ~/Library/Application Support/Binding of Isaac Repentance/options.ini

We flip the following knobs, backing up the original as options.ini.pre-rl-bak:

  PauseOnFocusLost      = 0    # don't pause when the window isn't focused
  Fullscreen            = 0    # windowed mode (multi-instance requires this)
  Filter                = 0    # no HQx upscaling — big CPU/GPU savings
  MusicVolume           = 0
  SFXVolume             = 0
  VSync                 = 0    # let each Isaac run as fast as it can
  WindowWidth           = 480
  WindowHeight          = 270
  MaxScale              = 1
  MaxRenderScale        = 1

Isaac ALSO has a background-FPS throttle that isn't in options.ini — it's
hard-coded to about 5 FPS when the window isn't focused. There's no clean
knob for it in vanilla, but disabling pause-on-focus-loss + running the
windows visible (not minimized) keeps them at the normal 30/60 Hz.

Usage (PowerShell):

    python tools\\configure_isaac.py apply    # set training-friendly options
    python tools\\configure_isaac.py restore  # restore your original options.ini
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import sys
from pathlib import Path


def options_path() -> Path | None:
    system = platform.system()
    if system == "Windows":
        return Path.home() / "Documents" / "My Games" / "Binding of Isaac Repentance" / "options.ini"
    if system == "Linux":
        return Path.home() / ".local/share/binding of isaac repentance/options.ini"
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Binding of Isaac Repentance/options.ini"
    return None


TRAINING_OPTIONS = {
    "PauseOnFocusLost": "0",
    "Fullscreen": "0",
    "Filter": "0",
    "MusicVolume": "0",
    "MusicVolumeChannel": "0",
    "SFXVolume": "0",
    "SFXVolumeChannel": "0",
    "VSync": "0",
    "WindowWidth": "480",
    "WindowHeight": "270",
    "MaxScale": "1.0",
    "MaxRenderScale": "1",
    "EnableMouseControl": "0",
    "HideHUD": "0",     # keep HUD; you want to see what's happening
    "PopUps": "1",      # keep item popups
}


def patch_options(text: str) -> tuple[str, dict[str, str]]:
    """Apply TRAINING_OPTIONS to an options.ini text. Returns (new_text, changed)."""
    changed: dict[str, str] = {}
    out_lines = []
    seen: set[str] = set()

    for line in text.splitlines():
        m = re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*)$", line)
        if not m:
            out_lines.append(line)
            continue
        indent, key, eq, val = m.groups()
        if key in TRAINING_OPTIONS:
            desired = TRAINING_OPTIONS[key]
            if val.strip() != desired:
                changed[key] = f"{val.strip()} -> {desired}"
                line = f"{indent}{key}{eq}{desired}"
            seen.add(key)
        out_lines.append(line)

    # Append any options that weren't present.
    missing = [k for k in TRAINING_OPTIONS if k not in seen]
    if missing:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        for k in missing:
            out_lines.append(f"{k}={TRAINING_OPTIONS[k]}")
            changed[k] = f"(missing) -> {TRAINING_OPTIONS[k]}"

    return "\n".join(out_lines) + "\n", changed


def cmd_apply(path: Path) -> int:
    if not path.exists():
        print(f"error: options.ini not found at {path}", file=sys.stderr)
        print("       Launch Isaac once through Steam so it creates the file, then retry.", file=sys.stderr)
        return 2

    backup = path.with_suffix(path.suffix + ".pre-rl-bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backed up original -> {backup}")
    else:
        print(f"backup already exists: {backup} (leaving as-is)")

    text = path.read_text(encoding="utf-8", errors="replace")
    new_text, changed = patch_options(text)
    if not changed:
        print("no changes needed — options.ini already training-friendly.")
        return 0

    path.write_text(new_text, encoding="utf-8")
    print(f"patched {path}:")
    for k, v in changed.items():
        print(f"  {k}: {v}")
    return 0


def cmd_restore(path: Path) -> int:
    backup = path.with_suffix(path.suffix + ".pre-rl-bak")
    if not backup.exists():
        print(f"no backup to restore ({backup} does not exist)", file=sys.stderr)
        return 2
    shutil.copy2(backup, path)
    print(f"restored {path} from {backup}")
    return 0


def cmd_show(path: Path) -> int:
    if not path.exists():
        print(f"error: options.ini not found at {path}", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"[{path}]")
    for k in TRAINING_OPTIONS:
        m = re.search(rf"^\s*{re.escape(k)}\s*=\s*(.*)$", text, re.MULTILINE)
        current = m.group(1).strip() if m else "(absent)"
        desired = TRAINING_OPTIONS[k]
        mark = "✓" if current == desired else "·"
        print(f"  [{mark}] {k:22s} current={current!r:12s} target={desired!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("apply", help="Apply training-friendly options (backup made once).")
    sub.add_parser("restore", help="Restore your original options.ini from the backup.")
    sub.add_parser("show", help="Print current vs. target values.")
    args = ap.parse_args()

    path = options_path()
    if path is None:
        print("unsupported OS", file=sys.stderr)
        return 2

    if args.cmd == "apply":
        return cmd_apply(path)
    if args.cmd == "restore":
        return cmd_restore(path)
    if args.cmd == "show":
        return cmd_show(path)
    ap.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
