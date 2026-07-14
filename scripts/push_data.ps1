# Isaac RL — Data collection & push helper.
#
# Exports the TensorBoard scalar timeseries from the most recent run (or a
# named run) into a small JSON, commits BOTH the JSON summary AND the
# `latest.pt` checkpoint, and pushes to origin. This is what you run after
# Ctrl-C-ing training so I can analyze the run remotely.
#
# Usage:
#   .\scripts\push_data.ps1                        # most recent stage1 run
#   .\scripts\push_data.ps1 -RunGlob "dreamer_stage2_*"
#   .\scripts\push_data.ps1 -RunDir "runs\dreamer_stage1_single_room\20260704-231512"
#   .\scripts\push_data.ps1 -NoCheckpoint          # JSON only, skip the .pt
#
# What gets committed:
#   tb_dreamer_<stage>_<step>.json   (~50 KB, always)
#   ckpts\latest_<stage>_<step>.pt   (~150 MB, unless -NoCheckpoint)
#
# The checkpoint gets Git LFS-friendly names but is committed as-is — if
# your repo enforces file size limits, use -NoCheckpoint and upload the
# .pt via a bucket / scp / email separately.

param(
    [string]$RunDir = "",
    [string]$RunGlob = "*",     # matches cleanrl_ppo_*, dreamer_*, ppo_*, anything under runs/
    [switch]$NoCheckpoint,
    [string]$Message = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
}
$env:PYTHONPATH = "python"

# Resolve run directory.
if ($RunDir -eq "") {
    $runsRoot = Join-Path $RepoRoot "runs"
    if (-not (Test-Path $runsRoot)) {
        Write-Error "No runs\ directory found — has training been run?"
        exit 1
    }
    # Most recent timestamped subdir matching the glob.
    $candidates = Get-ChildItem -Path $runsRoot -Directory | Where-Object { $_.Name -like $RunGlob }
    if (-not $candidates) {
        Write-Error "No run directories matching '$RunGlob' under runs\"
        exit 1
    }
    # Each stage dir has timestamped subdirs; pick the newest timestamp across all matching stages.
    $latestRun = $candidates | ForEach-Object {
        Get-ChildItem -Path $_.FullName -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latestRun) {
        Write-Error "No timestamped runs under matching stage directories."
        exit 1
    }
    $RunDir = $latestRun.FullName
}

if (-not (Test-Path $RunDir)) {
    Write-Error "Run dir does not exist: $RunDir"
    exit 1
}

Write-Host ""
Write-Host "==== Isaac RL — push run data ====" -ForegroundColor Cyan
Write-Host "run dir: $RunDir"

# Determine stage tag from the parent dir name. Handles all naming schemes:
#   cleanrl_ppo_stageA  -> stageA
#   dreamer_stage1_single_room -> stage1
#   ppo_stage2 -> stage2
#   whatever_else -> whatever_else (fallback)
$stageDir = Split-Path -Parent $RunDir
$stageName = Split-Path -Leaf $stageDir
if ($stageName -match 'stage[A-Za-z0-9]+') {
    $stageTag = $Matches[0]
} else {
    $stageTag = $stageName -replace '^(cleanrl_ppo|dreamer|ppo)_', ''
}

# ---- 1. Export TB summary --------------------------------------------------
$tbEvents = Get-ChildItem -Path $RunDir -Filter "events.out.tfevents.*" -ErrorAction SilentlyContinue
if (-not $tbEvents) {
    Write-Warning "No TB event files in $RunDir — was --tensorboard passed to run.ps1?"
} else {
    # Read a step count from the tb summary to name the file. Fall back to timestamp.
    $ts = Split-Path -Leaf $RunDir
    $tbJsonRel = "tb_${stageTag}_${ts}.json"
    Write-Host "exporting TB scalars -> $tbJsonRel" -ForegroundColor Green
    python export_tb_summary.py $RunDir --out $tbJsonRel
    # `git add` prints an ignore-hint to stderr for some paths; under
    # $ErrorActionPreference = "Stop" PowerShell can bail on the *next* pipe
    # step. Swallow non-zero exit + hints explicitly. Sanity check the file
    # actually got staged before continuing.
    cmd /c "git add `"$tbJsonRel`" 2>&1" | Out-Null
    $staged = git diff --cached --name-only
    if ($staged -notcontains $tbJsonRel) {
        Write-Warning "git add did not stage $tbJsonRel — retrying with -f"
        cmd /c "git add -f `"$tbJsonRel`" 2>&1" | Out-Null
    }
}

# ---- 2. Copy latest checkpoint --------------------------------------------
$ckptSrc = Join-Path $RunDir "latest.pt"
if (-not $NoCheckpoint) {
    if (-not (Test-Path $ckptSrc)) {
        Write-Warning "No latest.pt in $RunDir — was training aborted before first checkpoint?"
    } else {
        $ckptDstDir = Join-Path $RepoRoot "ckpts"
        New-Item -ItemType Directory -Path $ckptDstDir -Force | Out-Null
        $ts = Split-Path -Leaf $RunDir
        $ckptDstRel = "ckpts\latest_${stageTag}_${ts}.pt"
        Copy-Item $ckptSrc (Join-Path $RepoRoot $ckptDstRel) -Force
        $sizeMB = [math]::Round((Get-Item (Join-Path $RepoRoot $ckptDstRel)).Length / 1MB, 1)
        Write-Host "copied checkpoint (${sizeMB} MB) -> $ckptDstRel" -ForegroundColor Green

        if ($sizeMB -gt 100) {
            Write-Warning "Checkpoint is ${sizeMB} MB — GitHub rejects files >100MB. Consider -NoCheckpoint and use a bucket / scp / Google Drive."
        }
        # `.pt` is git-ignored in this repo; use -f to force-add so the
        # ignore hint doesn't abort the outer script under strict mode.
        cmd /c "git add -f `"$ckptDstRel`" 2>&1" | Out-Null
    }
} else {
    Write-Host "skipping checkpoint (-NoCheckpoint)"
}

# ---- 2b. Copy the config + git SHA + episodes CSV + mod log tail ----------
# Post 2026-07-13 nuclear reset: cleanrl_ppo.py saves config.yaml, git_sha.txt,
# and episodes.csv in the run dir. Also grab the last 500 lines of Isaac's
# log.txt so I can see any mod-side errors, restart cycles, STAGE setup
# failures, etc.
foreach ($aux in @("config.yaml", "git_sha.txt", "episodes.csv")) {
    $auxSrc = Join-Path $RunDir $aux
    if (Test-Path $auxSrc) {
        $auxDstRel = "runs_meta\${stageTag}_${ts}_${aux}"
        New-Item -ItemType Directory -Path (Join-Path $RepoRoot "runs_meta") -Force | Out-Null
        Copy-Item $auxSrc (Join-Path $RepoRoot $auxDstRel) -Force
        cmd /c "git add `"$auxDstRel`" 2>&1" | Out-Null
        Write-Host "copied run metadata -> $auxDstRel" -ForegroundColor Green
    }
}

# Mod log tail: last 500 lines of Isaac's log.txt, filtered to isaac-rl-bridge
# lines. Catches Lua errors, STAGE setup failures, restart cycles, socket
# problems — all things TB scalars can't tell me.
$isaacLog = "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\log.txt"
if (Test-Path $isaacLog) {
    $modLogRel = "runs_meta\${stageTag}_${ts}_isaac_log_tail.txt"
    # Grab last 2000 lines, filter to our mod's messages + any Lua error lines.
    $tail = Get-Content $isaacLog -Tail 2000 -ErrorAction SilentlyContinue
    if ($tail) {
        $filtered = $tail | Where-Object {
            $_ -match "isaac-rl-bridge|Lua Error|Lua Debug|handle_player_death|STAGE|restart|Fatal"
        }
        # Keep last 500 filtered lines (older lines might be from a previous run).
        if ($filtered.Count -gt 500) { $filtered = $filtered[-500..-1] }
        Set-Content -Path (Join-Path $RepoRoot $modLogRel) -Value $filtered -Encoding UTF8
        cmd /c "git add `"$modLogRel`" 2>&1" | Out-Null
        Write-Host "copied Isaac log tail ($($filtered.Count) lines) -> $modLogRel" -ForegroundColor Green
    }
} else {
    Write-Warning "Isaac log.txt not found at $isaacLog"
}

# ---- 3. Commit + push -----------------------------------------------------
$status = git status --porcelain
if (-not $status) {
    Write-Host "nothing to commit (no changes)"
    exit 0
}

if ($Message -eq "") {
    $ts = Split-Path -Leaf $RunDir
    $Message = "data: ${stageName} @ ${ts}"
}

git commit -m $Message
git push origin HEAD

Write-Host ""
Write-Host "pushed. Latest commit:" -ForegroundColor Green
git log -1 --oneline
