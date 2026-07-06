"""Print a PASS/FAIL health readout for the most recent Dreamer TB export.

Invoked by ``scripts/check_progress.ps1``. Reads the JSON summary produced
by ``export_tb_summary.py`` and prints colored PASS/FAIL against key
training-health thresholds (throughput, entropy, WM loss, KL, reward
trend), followed by a bottom-line KEEP GOING / STOP / TOO EARLY verdict.

Usage:
    python scripts/_check_progress.py <path-to-tb-summary.json>
"""
from __future__ import annotations

import json
import statistics as st
import sys


# ANSI colors (Windows Terminal, VS Code terminal, etc. support these).
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _last(per: dict, key: str) -> float | None:
    if key not in per:
        return None
    vs = per[key]["timeseries"]["values"]
    return vs[-1] if vs else None


def _first(per: dict, key: str) -> float | None:
    if key not in per:
        return None
    vs = per[key]["timeseries"]["values"]
    return vs[0] if vs else None


def _trend(per: dict, key: str, pct: float = 0.25) -> float | None:
    """Difference between mean of last pct% and mean of first pct% of the
    run. Positive => improving."""
    if key not in per:
        return None
    vs = per[key]["timeseries"]["values"]
    if len(vs) < 4:
        return None
    n = max(1, int(len(vs) * pct))
    return st.mean(vs[-n:]) - st.mean(vs[:n])


def _check(label: str, val: float | None, thr: float, direction: str, ctx: str = "") -> bool:
    """Print a single PASS/FAIL line. Returns True if the check passed."""
    if val is None:
        print(f"  [ SKIP ] {label:32s} (metric missing)")
        return True   # not a hard fail
    ok = val >= thr if direction == ">=" else val <= thr
    color = GREEN if ok else RED
    tag = "  OK  " if ok else " FAIL "
    print(f"  {color}[{tag}]{RESET} {label:32s} {val:+9.3f}   (want {direction} {thr}) {ctx}")
    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/_check_progress.py <tb-summary.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1]) as f:
        d = json.load(f)

    md = d["metadata"]
    per = d["per_scalar"]

    total_steps = md["total_steps"]
    dur_h = md["wall_clock_duration_s"] / 3600.0
    sps = total_steps / md["wall_clock_duration_s"] if md["wall_clock_duration_s"] > 0 else 0.0

    print(f"steps: {total_steps:,}   wall-clock: {dur_h:.2f}h   overall sps: {sps:.2f}")
    print()

    # Health checks (each returns True on pass).
    sps_last = _last(per, "perf/sps")
    _check(
        "perf/sps  (throughput)", sps_last, 2.0, ">=",
        "-> below 2, hardware may be misconfigured (window minimized?)",
    )

    entropy_last = _last(per, "loss/actor_entropy")
    _check(
        "actor_entropy (exploration)", entropy_last, 1.5, ">=",
        "-> collapse => policy becomes deterministic",
    )

    wm_loss = _last(per, "loss/total")
    wm_threshold = 5.0 if total_steps > 500_000 else 15.0
    _check(
        "WM loss/total", wm_loss, wm_threshold, "<=",
        f"-> want <= {wm_threshold} at step {total_steps:,}",
    )

    kl_last = _last(per, "loss/kl")
    _check(
        "KL   (WM stability)", kl_last, 20.0, "<=",
        "-> if > 20 or exploding, WM prior/posterior diverging",
    )

    # Trend metrics.
    ep_r_trend = _trend(per, "rollout/ep_reward", 0.25)
    if ep_r_trend is not None:
        ok = ep_r_trend > 0
        color = GREEN if ok else RED
        tag = "  OK  " if ok else " FAIL "
        print(
            f"  {color}[{tag}]{RESET} {'ep_reward TREND (last vs first)':32s} "
            f"{ep_r_trend:+9.3f}   (want > 0, i.e. improving)"
        )

    best_last = _last(per, "rollout/ep_reward_best")
    best_first = _first(per, "rollout/ep_reward_best")
    if best_first is not None and best_last is not None:
        ok = best_last > best_first
        color = GREEN if ok else RED
        tag = "  OK  " if ok else " FAIL "
        print(
            f"  {color}[{tag}]{RESET} {'ep_reward_best (new high)':32s} "
            f"{best_last:+9.3f}   (started at {best_first:+.3f})"
        )

    n_eps = _last(per, "rollout/n_episodes")
    if n_eps is not None:
        print(f"  [ INFO ] {'episodes completed':32s} {int(n_eps):>9d}")

    # Reward-event firing check.
    reward_keys = sorted(k for k in per.keys() if k.startswith("reward/"))
    if reward_keys:
        print()
        print("  Reward-event breakdown (last values):")
        for k in reward_keys:
            v = _last(per, k)
            if v is not None:
                print(f"    {k:30s} {v:+.4f}")
    else:
        print()
        print(
            f"  {YELLOW}[ WARN ]{RESET} no reward/* metrics — agent may not be "
            "triggering sparse rewards yet"
        )

    # Bottom-line advice.
    print()
    print("---- Recommendation ----")

    hard_fails: list[str] = []
    if sps_last is not None and sps_last < 2.0:
        hard_fails.append("throughput too low (probably window minimized)")
    if entropy_last is not None and entropy_last < 1.0:
        hard_fails.append("entropy collapsed")
    if wm_loss is not None and wm_loss > 50 and total_steps > 100_000:
        hard_fails.append("WM not converging")
    if ep_r_trend is not None and ep_r_trend < -1.0 and total_steps > 200_000:
        hard_fails.append("ep_reward regressing")

    if hard_fails:
        print(f"  {RED}STOP{RESET} and diagnose: {', '.join(hard_fails)}")
        print("  Push data with .\\scripts\\push_data.ps1 and share.")
    elif total_steps < 100_000:
        print(f"  {YELLOW}TOO EARLY{RESET} — need >=100k steps for meaningful trend. Keep training.")
    else:
        print(f"  {GREEN}KEEP GOING{RESET} — no red flags. Check again tomorrow.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
