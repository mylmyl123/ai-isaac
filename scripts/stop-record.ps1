# Stop an active demo recording session from any PowerShell window.
#
# Creates demos\STOP which the recorder polls once per second (alongside
# Ctrl+C / SIGTERM). Use this if Ctrl+C in the recorder's own PowerShell
# window won't respond \u2014 e.g. some IDE-embedded terminals, or when the
# Python signal handler is delayed for any reason.
#
# Usage:
#   .\scripts\stop-record.ps1                     # stops sessions writing to demos/
#   .\scripts\stop-record.ps1 -OutDir demos\bc    # if you passed -OutDir to record.ps1
#
# The recorder deletes the STOP file when it starts a fresh session, so this
# is idempotent \u2014 running it twice does no harm.

param(
    [string]$OutDir = "demos"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
$stopFile = Join-Path $OutDir "STOP"
New-Item -ItemType File -Path $stopFile -Force | Out-Null

Write-Host "stop signal written to: $stopFile" -ForegroundColor Green
Write-Host "recorder should exit within ~1s."
