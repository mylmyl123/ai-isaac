# Isaac RL — human demonstration recorder launcher.
#
# Launches Isaac with RECORD_MODE flag set. You play the game normally with
# keyboard/gamepad; every 15 Hz tick (obs + your input state) gets written to
# demos/session_<timestamp>.jsonl. Later fed into BC training to bootstrap the
# RL actor with human demonstrations.
#
# Usage:
#   .\scripts\record.ps1                       # auto-detect Isaac from Steam
#   .\scripts\record.ps1 -Isaac "C:\path\to\isaac-ng.exe"
#   .\scripts\record.ps1 -OutDir demos\my-runs
#   .\scripts\record.ps1 -Port 9600            # if 9500 is in use
#
# Press Ctrl+C in THIS PowerShell window to stop recording. Isaac's window
# stays open; close it separately.
#
# Isaac binary auto-detected from standard Steam install locations.

param(
    [string]$Isaac = "",
    [string]$OutDir = "demos",
    [int]$Port = 9500,
    [int]$AcceptTimeoutS = 300
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# venv activation
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
}
$env:PYTHONPATH = "python"

# Auto-detect Isaac
if ($Isaac -eq "") {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
        "${env:ProgramFiles}\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
        "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
        "C:\Program Files\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) {
            $Isaac = $c
            break
        }
    }
    if ($Isaac -eq "") {
        Write-Error "Isaac binary not found in standard Steam locations. Pass -Isaac <path>."
        exit 1
    }
}

if (-not (Test-Path $Isaac)) {
    Write-Error "Isaac binary not found at: $Isaac"
    exit 1
}

Write-Host ""
Write-Host "==== Isaac RL — Human Demo Recorder ====" -ForegroundColor Cyan
Write-Host "isaac:   $Isaac"
Write-Host "port:    $Port"
Write-Host "outdir:  $OutDir"
Write-Host ""
Write-Host "* Play Isaac normally. Movement WASD/arrows, shoot with arrows/gamepad." -ForegroundColor Green
Write-Host "* Every 2 frames (~15 Hz) your obs + input state is recorded." -ForegroundColor Green
Write-Host "* Ctrl+C in THIS window to stop. Isaac window stays open." -ForegroundColor Green
Write-Host ""

python -m isaac_rl.record `
    --isaac $Isaac `
    --port $Port `
    --out $OutDir `
    --accept-timeout-s $AcceptTimeoutS
