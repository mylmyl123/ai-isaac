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

# Export TB scalars to a temporary JSON, then hand it to the Python
# analyzer. We keep the analyzer as a separate .py file (not an inline
# heredoc) so PowerShell doesn't mangle its argv / string escaping.
$tmpJson = Join-Path $env:TEMP "isaacrl_progress_$([guid]::NewGuid()).json"
try {
    python export_tb_summary.py $RunDir --out $tmpJson 2>&1 | Out-Null
    if (-not (Test-Path $tmpJson)) {
        Write-Error "Failed to export TB summary"
        exit 1
    }
    python scripts\_check_progress.py $tmpJson
} finally {
    if (Test-Path $tmpJson) { Remove-Item $tmpJson -Force }
}
