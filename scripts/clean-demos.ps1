# Delete short / broken demo sessions from demos/ (or a custom directory).
#
# Any session_*.jsonl file with fewer than -MinTicks lines is treated as a
# 'bad run' \u2014 either user restarted immediately, Isaac crashed, mod misfired,
# or the connection died before real gameplay started. Lists what would be
# deleted first (dry run); pass -Delete to actually remove them.
#
# Usage:
#   .\scripts\clean-demos.ps1                          # dry run, threshold 150 ticks
#   .\scripts\clean-demos.ps1 -Delete                  # actually delete
#   .\scripts\clean-demos.ps1 -MinTicks 300 -Delete    # stricter threshold (~20s)
#   .\scripts\clean-demos.ps1 -OutDir demos\bc         # custom directory

param(
    [string]$OutDir = "demos",
    [int]$MinTicks = 150,
    [switch]$Delete
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path $OutDir)) {
    Write-Host "no such directory: $OutDir" -ForegroundColor Yellow
    exit 0
}

$files = Get-ChildItem -Path $OutDir -Filter "session_*.jsonl" -File
if ($files.Count -eq 0) {
    Write-Host "no sessions found in $OutDir" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "==== Demo cleanup: $OutDir (min $MinTicks ticks) ====" -ForegroundColor Cyan
Write-Host ""

$toDelete = @()
$totalKeep = 0
$totalKeepTicks = 0

foreach ($f in $files) {
    # Count lines cheaply. JSONL = one obs frame per line.
    $ticks = (Get-Content $f.FullName | Measure-Object -Line).Lines
    $sizeKB = [math]::Round($f.Length / 1024, 1)
    if ($ticks -lt $MinTicks) {
        Write-Host ("BAD  {0,6} ticks  {1,7} KB  {2}" -f $ticks, $sizeKB, $f.Name) -ForegroundColor Red
        $toDelete += $f
    } else {
        Write-Host ("KEEP {0,6} ticks  {1,7} KB  {2}" -f $ticks, $sizeKB, $f.Name) -ForegroundColor Green
        $totalKeep++
        $totalKeepTicks += $ticks
    }
}

Write-Host ""
Write-Host ("Summary: {0} kept ({1} ticks total), {2} to delete" -f $totalKeep, $totalKeepTicks, $toDelete.Count) -ForegroundColor Cyan

if ($toDelete.Count -eq 0) {
    Write-Host "nothing to clean." -ForegroundColor Green
    exit 0
}

if (-not $Delete) {
    Write-Host ""
    Write-Host "Dry run \u2014 rerun with -Delete to actually remove these files." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Deleting $($toDelete.Count) files..." -ForegroundColor Yellow
foreach ($f in $toDelete) {
    Remove-Item $f.FullName -Force
}
Write-Host "done." -ForegroundColor Green
