# Isaac RL — quick "is my run making progress?" readout.
#
# Reads the most recent run's TensorBoard events without needing to open
# TB in a browser. Prints a colored PASS/FAIL against key success criteria
# so you can decide whether to keep training or Ctrl-C and try something
# else.
#
# Usage:
#   .\scripts\check_progress.ps1                # most recent run
#   .\scripts\check_progress.ps1 -RunDir "runs\dreamer_stage1_single_room_xs\<ts>"
#
# Success criteria (rough, based on 3060 Ti XS config, ~1 week budget):
#   sps            >= 3          (throughput; below this, week-budget won't work)
#   actor_entropy  >= 1.5        (exploration alive)
#   loss/total     <=15 by 50k,  (WM converging)
#                  <= 5 by 500k
#   ep_reward      trending UP over the last 100k steps
#   ep_reward_best rising monotonically
#
# The script never modifies anything. Safe to run repeatedly.

param(
    [string]$RunDir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
}
$env:PYTHONPATH = "python"

# Resolve run directory
if ($RunDir -eq "") {
    $runsRoot = Join-Path $RepoRoot "runs"
    if (-not (Test-Path $runsRoot)) {
        Write-Error "No runs\ dir. Has training been run?"; exit 1
    }
    $latest = Get-ChildItem -Path $runsRoot -Directory | ForEach-Object {
        Get-ChildItem -Path $_.FullName -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { Write-Error "No timestamped runs found."; exit 1 }
    $RunDir = $latest.FullName
}

Write-Host ""
Write-Host "==== Isaac RL — Progress Check ====" -ForegroundColor Cyan
Write-Host "run: $RunDir"
Write-Host ""

# Use export_tb_summary.py to get a JSON dump, then parse it.
$tmpJson = Join-Path $env:TEMP "isaacrl_progress_$([guid]::NewGuid()).json"
try {
    python export_tb_summary.py $RunDir --out $tmpJson 2>&1 | Out-Null
    if (-not (Test-Path $tmpJson)) {
        Write-Error "Failed to export TB summary"
        exit 1
    }
    python - $tmpJson <<'PYEOF'
import json, sys, statistics as st
path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

md = d['metadata']
per = d['per_scalar']

def last(k):
    if k not in per: return None
    ts = per[k]['timeseries']
    return ts['values'][-1] if ts['values'] else None

def first(k):
    if k not in per: return None
    ts = per[k]['timeseries']
    return ts['values'][0] if ts['values'] else None

def trend_last_pct(k, pct=0.25):
    """Mean of last pct% - mean of first pct% of the run. Positive => improving."""
    if k not in per: return None
    vs = per[k]['timeseries']['values']
    if len(vs) < 4: return None
    n = max(1, int(len(vs) * pct))
    return st.mean(vs[-n:]) - st.mean(vs[:n])

total_steps = md['total_steps']
dur_h = md['wall_clock_duration_s'] / 3600
sps = total_steps / md['wall_clock_duration_s'] if md['wall_clock_duration_s'] > 0 else 0

print(f"steps: {total_steps:,}   wall-clock: {dur_h:.2f}h   overall sps: {sps:.2f}")
print()

# Health checks. Each row: (label, current_value, threshold, direction, extra_context)
def check(label, val, thr, direction, ctx=""):
    if val is None:
        print(f"  [ SKIP ] {label:32s} (metric missing)"); return
    ok = val >= thr if direction == ">=" else val <= thr
    tag = "  OK   " if ok else " FAIL  "
    color_marker = "\033[92m" if ok else "\033[91m"
    reset = "\033[0m"
    print(f"  {color_marker}[{tag}]{reset} {label:32s} {val:+9.3f}   (want {direction} {thr}) {ctx}")

sps_last = last('perf/sps')
check("perf/sps  (throughput)", sps_last, 3.0, ">=",
      "-> if under 3, 1-week budget won't work")

entropy_last = last('loss/actor_entropy')
check("actor_entropy (exploration)", entropy_last, 1.5, ">=",
      "-> collapse => policy becomes deterministic, will not learn further")

wm_loss = last('loss/total')
wm_threshold = 5.0 if total_steps > 500_000 else 15.0
check("WM loss/total", wm_loss, wm_threshold, "<=",
      f"-> want ≤ {wm_threshold} at step {total_steps:,}")

kl_last = last('loss/kl')
check("KL   (WM stability)", kl_last, 20.0, "<=",
      "-> if > 20 or exploding, WM prior/posterior diverging")

# Trend metrics — improvement over the run
ep_r_trend = trend_last_pct('rollout/ep_reward', 0.25)
if ep_r_trend is not None:
    d_ok = ep_r_trend > 0
    tag = "  OK   " if d_ok else " FAIL  "
    c = "\033[92m" if d_ok else "\033[91m"
    print(f"  {c}[{tag}]\033[0m {'ep_reward TREND (last vs first)':32s} {ep_r_trend:+9.3f}   (want > 0, i.e. improving)")

best_last = last('rollout/ep_reward_best')
best_first = first('rollout/ep_reward_best')
if best_first is not None and best_last is not None:
    improved = best_last > best_first
    tag = "  OK   " if improved else " FAIL  "
    c = "\033[92m" if improved else "\033[91m"
    print(f"  {c}[{tag}]\033[0m {'ep_reward_best (new high)':32s} {best_last:+9.3f}   (started at {best_first:+.3f})")

n_eps = last('rollout/n_episodes')
if n_eps is not None:
    print(f"  [ INFO ] {'episodes completed':32s} {int(n_eps):>9d}")

# Reward-event firing check: are ANY of the sparse rewards showing up?
reward_keys = sorted(k for k in per.keys() if k.startswith('reward/'))
if reward_keys:
    print()
    print(f"  Reward-event breakdown (last values):")
    for k in reward_keys:
        v = last(k)
        if v is not None:
            print(f"    {k:30s} {v:+.4f}")
else:
    print(f"\n  \033[93m[ WARN ]\033[0m no reward/* metrics — agent may not be triggering sparse rewards yet")

# Bottom-line advice
print()
print("---- Recommendation ----")

hard_fails = []
if sps_last is not None and sps_last < 3.0:
    hard_fails.append("throughput too low")
if entropy_last is not None and entropy_last < 1.0:
    hard_fails.append("entropy collapsed")
if wm_loss is not None and wm_loss > 50 and total_steps > 100_000:
    hard_fails.append("WM not converging")
if ep_r_trend is not None and ep_r_trend < -1.0 and total_steps > 200_000:
    hard_fails.append("ep_reward regressing")

if hard_fails:
    print(f"  \033[91mSTOP\033[0m and diagnose: {', '.join(hard_fails)}")
    print("  Push data with .\\scripts\\push_data.ps1 and share.")
elif total_steps < 100_000:
    print(f"  \033[93mTOO EARLY\033[0m — need >=100k steps for meaningful trend. Keep training.")
else:
    print(f"  \033[92mKEEP GOING\033[0m — no red flags. Check again tomorrow.")
PYEOF
} finally {
    if (Test-Path $tmpJson) { Remove-Item $tmpJson -Force }
}
