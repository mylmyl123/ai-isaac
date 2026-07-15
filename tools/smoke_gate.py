"""Smoke-gate analyzer for the Isaac RL curriculum.

Reads (1) Isaac's log.txt (mod DebugString output) and (2) the training run's
episodes.csv, and asserts the environment is set up correctly BEFORE a long
training run is trusted. A plain "did kills_mean rise?" check would false-pass
the wrong-enemy bug (a Maw is killable and gives reward just like a Horf), so
this gate checks enemy IDENTITY, not just that kills happened.

Checks (Stage 0 example, expected_enemy_type=12 for Horf):
  A. The mod spawned the intended enemy type   (log: "spawned type=<T>")
  B. Kills came from that same enemy type       (log: "kill npc_type=<T>")
  C. Episodes actually completed with kills     (episodes.csv ep_kills > 0)
  D. Episodes ended cleanly, not via crash loop (log: death vs crash ratio)

Exit code 0 = all gates pass (safe to launch the full run), 1 = a gate failed.

Usage:
    python tools/smoke_gate.py --run-dir runs/cleanrl_ppo_stage0/<ts> \
        --isaac-log "<...>/Binding of Isaac Repentance/log.txt" \
        --expected-enemy-type 12
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

SPAWN_RE = re.compile(r"spawned type=(\d+)")
KILL_RE = re.compile(r"kill npc_type=(\d+)")
DEATH_RE = re.compile(r"handle_player_death firing")
STAGE_RE = re.compile(r"curriculum stage=(\S+)")


def _line(ok: bool, label: str, detail: str) -> bool:
    tag = f"{GREEN}[ PASS ]{RESET}" if ok else f"{RED}[ FAIL ]{RESET}"
    print(f"  {tag} {label:36s} {detail}")
    return ok


def _warn(label: str, detail: str) -> None:
    print(f"  {YELLOW}[ WARN ]{RESET} {label:36s} {detail}")


def parse_isaac_log(path: Path) -> dict:
    """Extract spawn types, kill types, death count from the mod's DebugString.

    Only the isaac-rl-bridge lines matter. We scan the whole file; on Windows
    log.txt is rewritten per launch so it reflects the most recent session.
    """
    spawns: Counter = Counter()
    kills: Counter = Counter()
    deaths = 0
    stages: Counter = Counter()
    text = path.read_text(encoding="utf-8", errors="replace")
    for ln in text.splitlines():
        if "isaac-rl-bridge" not in ln:
            continue
        m = SPAWN_RE.search(ln)
        if m:
            spawns[int(m.group(1))] += 1
            continue
        m = KILL_RE.search(ln)
        if m:
            kills[int(m.group(1))] += 1
            continue
        if DEATH_RE.search(ln):
            deaths += 1
            continue
        m = STAGE_RE.search(ln)
        if m:
            stages[m.group(1)] += 1
    return {"spawns": spawns, "kills": kills, "deaths": deaths, "stages": stages}


def parse_episodes_csv(path: Path) -> dict:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    n = len(rows)
    total_kills = sum(int(r["ep_kills"]) for r in rows) if n else 0
    eps_with_kill = sum(1 for r in rows if int(r["ep_kills"]) > 0) if n else 0
    terminated = sum(1 for r in rows if int(r["terminated"]) == 1) if n else 0
    return {
        "n_episodes": n,
        "total_kills": total_kills,
        "eps_with_kill": eps_with_kill,
        "terminated": terminated,
        "mean_kills": (total_kills / n) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Training run dir containing episodes.csv")
    ap.add_argument("--isaac-log", required=True, help="Path to Isaac's log.txt")
    ap.add_argument("--expected-enemy-type", type=int, required=True,
                    help="EntityType the stage should spawn (Horf=12, AttackFly=18)")
    ap.add_argument("--min-episodes", type=int, default=3,
                    help="Minimum completed episodes to trust the smoke result")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    isaac_log = Path(args.isaac_log)
    ecsv = run_dir / "episodes.csv"

    print()
    print(f"==== Isaac RL — smoke gate (expected enemy type={args.expected_enemy_type}) ====")

    # Existence preconditions (hard fail — nothing to check otherwise).
    if not isaac_log.exists():
        print(f"  {RED}[ FAIL ]{RESET} isaac log.txt not found at: {isaac_log}")
        return 1
    if not ecsv.exists():
        print(f"  {RED}[ FAIL ]{RESET} episodes.csv not found at: {ecsv}")
        return 1

    log = parse_isaac_log(isaac_log)
    ep = parse_episodes_csv(ecsv)
    T = args.expected_enemy_type
    ok = True

    # A. Spawn identity.
    spawn_types = log["spawns"]
    spawned_T = spawn_types.get(T, 0)
    other_spawns = {k: v for k, v in spawn_types.items() if k != T}
    ok &= _line(
        spawned_T > 0 and not other_spawns,
        "A. mod spawned intended enemy",
        f"type {T} x{spawned_T}" + (f"; OTHER spawns={dict(other_spawns)}" if other_spawns else ""),
    )

    # B. Kill identity — THE check that catches the wrong-enemy bug.
    kill_types = log["kills"]
    killed_T = kill_types.get(T, 0)
    other_kills = {k: v for k, v in kill_types.items() if k != T}
    if sum(kill_types.values()) == 0:
        ok &= _line(False, "B. kills came from intended enemy",
                    "NO kill npc_type lines in log — update the mod (reward.lua kill logging)?")
    else:
        ok &= _line(
            killed_T > 0 and not other_kills,
            "B. kills came from intended enemy",
            f"type {T} x{killed_T}" + (f"; OTHER kills={dict(other_kills)}" if other_kills else ""),
        )

    # C. Episodes completed with kills.
    ok &= _line(
        ep["n_episodes"] >= args.min_episodes,
        "C. enough completed episodes",
        f"{ep['n_episodes']} episodes (want >= {args.min_episodes})",
    )
    ok &= _line(
        ep["eps_with_kill"] > 0 and ep["total_kills"] > 0,
        "   episodes contain kills",
        f"{ep['eps_with_kill']}/{ep['n_episodes']} eps had kills, {ep['total_kills']} total, mean {ep['mean_kills']:.2f}/ep",
    )

    # D. Clean episode ends (deaths), not a crash loop. Not a hard fail on its
    #    own but flagged loudly — a crash loop invalidates the run.
    if log["deaths"] == 0 and ep["n_episodes"] > 0:
        _warn("D. episodes end via death handler",
              "0 death-handler firings but episodes completed — check ep_end_reason (crash loop?)")
    else:
        _line(True, "D. episodes end via death handler",
              f"{log['deaths']} death-handler firings")

    # Stage sanity (informational).
    if log["stages"]:
        print(f"  {YELLOW}[ INFO ]{RESET} {'curriculum stage(s) seen':36s} {dict(log['stages'])}")

    print()
    if ok:
        print(f"{GREEN}SMOKE GATE PASSED{RESET} — environment identity verified. Safe to launch the full run.")
        return 0
    print(f"{RED}SMOKE GATE FAILED{RESET} — do NOT launch the full run until the failed checks are fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
