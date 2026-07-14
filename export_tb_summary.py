"""Extract TensorBoard scalar timeseries into a JSON summary.

Reads a TB event file and produces a compact JSON summary of the key
training metrics: min/max/mean/last/trend for each scalar. Also includes
the raw timeseries (downsampled to <= 200 points per scalar) so you can
send this file to me for analysis.

Usage:
    python export_tb_summary.py <run_dir_or_event_file> [--out summary.json]

Examples:
    python export_tb_summary.py runs/stage1_single_room/20260703-125800/
    python export_tb_summary.py runs/stage1_single_room/latest/ --out my_run.json

The output JSON includes:
    - metadata: run name, total steps, wall-clock duration
    - per_scalar: {name: {min, max, mean, last, first, count, timeseries}}
    - health_check: automated diagnostics for common failure modes

Small file (<50KB typically). Share this with me and I can diagnose
training issues from the numbers.

Depends on:
    pip install tbparse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_scalars(path: Path) -> dict:
    """Load all scalar values from a TensorBoard event file or directory."""
    try:
        from tbparse import SummaryReader
    except ImportError:
        print("ERROR: tbparse not installed. Run: pip install tbparse", file=sys.stderr)
        sys.exit(1)

    reader = SummaryReader(str(path), extra_columns={"wall_time"})
    df = reader.scalars
    if df is None or len(df) == 0:
        print(f"ERROR: no scalars found in {path}", file=sys.stderr)
        sys.exit(1)
    return df


def summarize(df, max_points_per_scalar: int = 200) -> dict:
    """Convert dataframe to a compact JSON-serializable summary."""
    out = {
        "metadata": {},
        "per_scalar": {},
        "health_check": {},
    }
    tags = sorted(df["tag"].unique())
    steps = df["step"].max()
    wall_start = df["wall_time"].min() if "wall_time" in df.columns else None
    wall_end = df["wall_time"].max() if "wall_time" in df.columns else None
    out["metadata"] = {
        "total_steps": int(steps),
        "n_scalars": len(tags),
        "wall_clock_start": float(wall_start) if wall_start is not None else None,
        "wall_clock_end": float(wall_end) if wall_end is not None else None,
        "wall_clock_duration_s": float(wall_end - wall_start) if wall_start is not None else None,
        "tags": tags,
    }

    # Per-scalar summaries.
    for tag in tags:
        sub = df[df["tag"] == tag].sort_values("step")
        values = sub["value"].to_numpy()
        step_arr = sub["step"].to_numpy()
        n = len(values)
        if n == 0:
            continue
        # Downsample to at most max_points_per_scalar.
        if n > max_points_per_scalar:
            idx = list(range(0, n, max(1, n // max_points_per_scalar)))
            values_ds = values[idx].tolist()
            steps_ds = step_arr[idx].tolist()
        else:
            values_ds = values.tolist()
            steps_ds = step_arr.tolist()
        out["per_scalar"][tag] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "first": float(values[0]),
            "last": float(values[-1]),
            "count": int(n),
            "trend_ratio_last_vs_first": float(values[-1] / max(abs(values[0]), 1e-8)),
            "timeseries": {
                "steps": steps_ds,
                "values": [float(v) for v in values_ds],
            },
        }

    # Health check: automatic diagnostics.
    #
    # Post 2026-07-13 reset: cleanrl_ppo.py writes scalars under 'charts/' and
    # 'loss/' prefixes (CleanRL convention). Accept both the new prefixed names
    # and the legacy bare names so old JSON exports still work if you diff.
    def _pick(*keys):
        for k in keys:
            if k in ps:
                return ps[k]
        return None

    checks = out["health_check"]
    ps = out["per_scalar"]

    e = _pick("loss/entropy")
    if e:
        checks["entropy"] = {
            "last": e["last"],
            "min": e["min"],
            "status": "healthy" if e["last"] > 0.8 else ("collapsed" if e["last"] < 0.3 else "warning"),
            "note": "target >0.8 for good exploration. <0.3 = policy is deterministic (bad). Max for MultiDiscrete([9,5,2,2,2]) is ~3.8.",
        }

    kills = _pick("charts/kills_mean", "behavior/kills")
    if kills:
        checks["kills_per_episode"] = {
            "first": kills["first"],
            "last": kills["last"],
            "max": kills["max"],
            "trend": "improving" if kills["last"] > kills["first"] + 0.5 else ("degrading" if kills["last"] < kills["first"] - 0.5 else "flat"),
            "note": "Stage A: random baseline is ~1-2 kills/ep. Learning shows this climbing to 4-6+. Flat at 1-2 through 100k steps = pipeline broken.",
        }

    ep_len = _pick("charts/ep_len_mean", "ep_len_mean")
    if ep_len:
        checks["episode_length"] = {
            "last": ep_len["last"],
            "trend": "increasing" if ep_len["last"] > ep_len["first"] * 1.5 else ("decreasing" if ep_len["last"] < ep_len["first"] * 0.7 else "stable"),
            "note": "Stage A: shorter = agent dying faster. Increasing = agent surviving longer. Stage E: >1500 with no progression = camping.",
        }

    ep_r = _pick("charts/ep_r_mean", "ep_r_mean")
    if ep_r:
        checks["reward"] = {
            "first": ep_r["first"],
            "last": ep_r["last"],
            "max": ep_r["max"],
            "trend": "improving" if ep_r["last"] > ep_r["first"] + 0.5 else ("degrading" if ep_r["last"] < ep_r["first"] - 0.5 else "stagnant"),
            "note": "3-term reward: r_kill=+1, r_death=-1, r_step=-0.001. So ep_r >0 = net-positive (killing faster than dying).",
        }

    v = _pick("loss/value")
    if v:
        checks["value_function"] = {
            "first": v["first"],
            "last": v["last"],
            "trend": "converging" if v["last"] < v["first"] else "diverging",
            "note": "should decrease over time. Diverging value = LR too high or GAE-lambda misconfigured.",
        }

    kl = _pick("loss/approx_kl")
    if kl:
        checks["approx_kl"] = {
            "last": kl["last"],
            "max": kl["max"],
            "note": "target <0.02 per update. >0.05 = policy updates are too aggressive (LR too high or clip_coef too permissive).",
        }

    clip = _pick("loss/clipfrac")
    if clip:
        checks["clip_frac"] = {
            "last": clip["last"],
            "note": "fraction of samples PPO-clipped. Healthy: 0.1-0.3. Near 0 = updates too small. Near 1 = ratio wildly off, clip is doing everything.",
        }

    sps = _pick("charts/sps")
    if sps:
        checks["throughput"] = {
            "last": sps["last"],
            "mean": sps["mean"],
            "note": "steps/sec across all envs. 2 envs @ 15 sps each = expect ~30 total.",
        }

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="TB run directory or specific event file")
    ap.add_argument("--out", default="tb_summary.json", help="Output JSON path (default: tb_summary.json)")
    ap.add_argument("--max-points", type=int, default=200, help="Max timeseries points per scalar (default: 200)")
    args = ap.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading TB scalars from {path}...")
    df = load_scalars(path)
    print(f"  loaded {len(df)} scalar samples across {df['tag'].nunique()} tags")

    summary = summarize(df, max_points_per_scalar=args.max_points)

    out_path = Path(args.out).resolve()
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {out_path}")
    print(f"  file size: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"  total steps: {summary['metadata']['total_steps']:,}")
    if summary['metadata']['wall_clock_duration_s']:
        h = summary['metadata']['wall_clock_duration_s'] / 3600
        print(f"  training duration: {h:.1f} hours")
    print(f"  scalars: {summary['metadata']['n_scalars']}")
    print("\nAutomatic health checks:")
    for check, data in summary["health_check"].items():
        status = data.get("status") or data.get("trend") or "-"
        print(f"  [{check}] {status}")
        if "note" in data:
            print(f"    {data['note']}")

    print("\nShare this file (or paste the contents) to get analysis of your training run.")


if __name__ == "__main__":
    main()
