# Isaac RL — human demonstration recorder launcher.
#
# Launches Isaac with RECORD_MODE flag set. You play the game normally with
# keyboard/gamepad; every 15 Hz tick (obs + your input state) gets written to
# demos/session_<timestamp>_run_NNN.jsonl. ONE JSONL PER ISAAC RUN — pressing
# R to restart automatically closes the current file and opens the next one.
# Runs shorter than 150 ticks (~10s) are auto-discarded so restart-scumming
# doesn't clutter the BC corpus. Fed into BC training to bootstrap the RL
# actor with human demonstrations.
#
# Usage:
#   .\scripts\record.ps1                       # auto-detect Isaac from Steam
#   .\scripts\record.ps1 -Isaac "C:\path\to\isaac-ng.exe"
#   .\scripts\record.ps1 -OutDir demos\my-runs
#   .\scripts\record.ps1 -Port 9600            # if 9500 is in use
#   .\scripts\record.ps1 -MinTicks 300         # stricter cutoff (~20s)
#
# Press Ctrl+C in THIS PowerShell window to stop recording. Isaac's window
# stays open; close it separately if you want.
#
# Isaac binary auto-detected from standard Steam install locations.

param(
    [string]$Isaac = "",
    [string]$OutDir = "demos",
    [int]$Port = 9500,
    [int]$AcceptTimeoutS = 300,
    [int]$MinTicks = 150
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

# Set env vars at PowerShell scope too, not just via subprocess.Popen's env=
# argument. Belt-and-suspenders — if Python's subprocess.Popen env-passing
# fails to propagate on Windows for whatever reason (some AV, some launcher
# indirection), Isaac still sees these via the inherited process env.
# Verified in mod via boot-time DebugString of os.getenv results.
$env:ISAAC_RL_RECORD = "1"
$env:ISAAC_RL_PORT = "$Port"

python -m isaac_rl.record `
    --isaac $Isaac `
    --port $Port `
    --out $OutDir `
    --accept-timeout-s $AcceptTimeoutS `
    --min-ticks $MinTicks
