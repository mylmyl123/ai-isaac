# Isaac RL — Dreamer training launcher.
#
# Runs stage 1 by default. Ctrl-C to stop cleanly: the trainer catches SIGINT
# and saves a final checkpoint + flushes TensorBoard before exiting.
#
# Usage:
#   .\scripts\run.ps1                          # stage 1, default settings
#   .\scripts\run.ps1 -Stage 2                 # stage 2
#   .\scripts\run.ps1 -Stage 4                 # stage 4
#   .\scripts\run.ps1 -Smoke                   # M1 smoke: 100k steps, n_envs=2
#   .\scripts\run.ps1 -NEnvs 4                 # override n_envs
#   .\scripts\run.ps1 -Isaac "C:\path\isaac-ng.exe"    # override binary path
#
# Isaac binary auto-detected from standard Steam install locations.

param(
    [string]$Stage = "1",
    [switch]$Smoke,
    [int]$NEnvs = 0,             # 0 = use YAML default
    [string]$Isaac = "",         # empty = auto-detect from Steam
    [switch]$NoTensorboard
)

$ErrorActionPreference = "Stop"

# Repo root = parent of scripts/
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# venv activation
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
}

$env:PYTHONPATH = "python"

# Config path
$stageConfigs = @{
    "1" = "python\isaac_rl\dreamer\configs\stage1_single_room.yaml"
    "2" = "python\isaac_rl\dreamer\configs\stage2_floor_clear.yaml"
    "4" = "python\isaac_rl\dreamer\configs\stage4_full_run.yaml"
}
if (-not $stageConfigs.ContainsKey($Stage)) {
    Write-Error "Unknown stage '$Stage'. Valid: 1, 2, 4."
    exit 1
}
$configPath = $stageConfigs[$Stage]

# Build the argument list
$cmd = @(
    "train.py",
    "--algo", "dreamer",
    "--config", $configPath
)

if (-not $NoTensorboard) { $cmd += "--tensorboard" }
if ($Isaac -ne "") { $cmd += @("--isaac", $Isaac) }

# Overrides
$overrides = @()
if ($Smoke) {
    $overrides += "total_env_steps=100000"
    if ($NEnvs -eq 0) { $overrides += "n_envs=2" }
}
if ($NEnvs -gt 0) { $overrides += "n_envs=$NEnvs" }
if ($overrides.Count -gt 0) {
    $cmd += "--override"
    $cmd += $overrides
}

Write-Host ""
Write-Host "==== Isaac RL — Dreamer stage $Stage ====" -ForegroundColor Cyan
Write-Host "config:    $configPath"
if ($Smoke)     { Write-Host "mode:      SMOKE (100k steps)" -ForegroundColor Yellow }
if ($Isaac)     { Write-Host "isaac:     $Isaac" }
if ($overrides) { Write-Host "overrides: $($overrides -join ' ')" }
Write-Host "cmd:       python $($cmd -join ' ')"
Write-Host ""
Write-Host "Press Ctrl-C to stop cleanly — final checkpoint will be saved." -ForegroundColor Green
Write-Host ""

python @cmd
