# Isaac RL — clear training data and checkpoints.
#
# Deletes runs/ (or a specific stage subdir) so a fresh training run doesn't
# collide with a previous one. Also optionally clears ckpts/ (checkpoints
# copied here by push_data.ps1) and any pushed tb_dreamer_*.json files.
#
# Confirmation-gated by default — pass -Yes to skip the prompt.
#
# Usage:
#   .\scripts\clear_data.ps1                       # prompt, delete runs\ only
#   .\scripts\clear_data.ps1 -Yes                  # runs\ only, no prompt
#   .\scripts\clear_data.ps1 -Stage 1              # only runs\dreamer_stage1_*
#   .\scripts\clear_data.ps1 -All                  # runs\ + ckpts\ + tb_*.json
#   .\scripts\clear_data.ps1 -All -Yes             # everything, no prompt

param(
    [string]$Stage = "",
    [switch]$All,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Build the target list.
$targets = @()

if ($Stage -ne "") {
    $stageMap = @{
        "1" = "runs\dreamer_stage1_single_room"
        "2" = "runs\dreamer_stage2_floor_clear"
        "4" = "runs\dreamer_stage4_full_run"
    }
    if (-not $stageMap.ContainsKey($Stage)) {
        Write-Error "Unknown stage '$Stage'. Valid: 1, 2, 4."
        exit 1
    }
    $targets += $stageMap[$Stage]
} else {
    $targets += "runs"
}

if ($All) {
    $targets += "ckpts"
    $tbJsons = Get-ChildItem -Path $RepoRoot -Filter "tb_dreamer_*.json" -ErrorAction SilentlyContinue
    foreach ($f in $tbJsons) { $targets += $f.FullName }
}

# Filter to things that actually exist.
$existing = $targets | Where-Object { Test-Path (Join-Path $RepoRoot $_) }

if ($existing.Count -eq 0) {
    Write-Host "nothing to delete."
    exit 0
}

Write-Host ""
Write-Host "==== Isaac RL — clear data ====" -ForegroundColor Cyan
Write-Host "Will DELETE:" -ForegroundColor Yellow
foreach ($t in $existing) {
    $full = Join-Path $RepoRoot $t
    if (Test-Path $full -PathType Container) {
        $size = (Get-ChildItem $full -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $sizeMB = if ($size) { [math]::Round($size / 1MB, 1) } else { 0 }
        Write-Host "  $t   (${sizeMB} MB, directory)"
    } else {
        $sizeMB = [math]::Round((Get-Item $full).Length / 1MB, 2)
        Write-Host "  $t   (${sizeMB} MB, file)"
    }
}
Write-Host ""

if (-not $Yes) {
    $confirm = Read-Host "Type 'yes' to confirm"
    if ($confirm -ne "yes") {
        Write-Host "aborted."
        exit 0
    }
}

foreach ($t in $existing) {
    $full = Join-Path $RepoRoot $t
    Write-Host "removing $t..."
    Remove-Item -Recurse -Force $full
}

Write-Host ""
Write-Host "done." -ForegroundColor Green
