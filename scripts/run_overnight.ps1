# Isaac RL — overnight run queue.
#
# Runs a QUEUE of training configs back-to-back so the machine isn't idle
# overnight. Deploys the mod once, then for each queued run: launches training,
# exports the TB summary, and pushes the data — so if a later run crashes or
# the machine dies, every COMPLETED run is already committed and safe.
#
# Default queue (4 runs, ~200k each, ~2h/run at ~28 sps = a full night):
#   1. r_hit=0.3                      — does rewarding every hit make the shoot
#                                        head commit? (the Phase-2c fix, isolated)
#   2. r_hit=0.3 + ent_coef=0.001     — if #1's shoot head still won't commit,
#                                        lower the entropy pressure holding it uniform
#   3. r_hit=0.6                      — stronger hit signal in case 0.3 is too weak
#   4. r_hit=0.3 + closer spawn (90-170px) — easier accidental hits to bootstrap
#                                        aiming, if the far spawn (200-500) is the
#                                        cold-start bottleneck. Still 1 Horf, single
#                                        variable. (Multi-enemy deferred: the fixed-
#                                        slot MLP can't do it well yet — see the
#                                        architecture analysis; that's a rearchitecture
#                                        job, not an overnight run.)
#
# Each run is try/catch-wrapped: one failure logs and continues to the next.
# No TensorBoard here (sequential runs would fight for :6006); read the pushed
# tb_stage*_<ts>.json instead.
#
# Usage:
#   .\scripts\run_overnight.ps1                 # default 4-run queue, auto-push
#   .\scripts\run_overnight.ps1 -NoPush         # export locally, don't push
#   .\scripts\run_overnight.ps1 -Steps 120000   # override per-run step budget
#   .\scripts\run_overnight.ps1 -Isaac "C:\...\isaac-ng.exe"

param(
    [string]$Config = "configs\curriculum.yaml",
    [string]$Isaac = "",
    [int]$Steps = 200000,
    [switch]$NoPush,
    [int]$SettleSeconds = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
$env:PYTHONPATH = "python"

function Section($msg) { Write-Host ""; Write-Host "==== $msg ====" -ForegroundColor Cyan }

# ---- Resolve Isaac binary --------------------------------------------------
if ($Isaac -eq "") {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
        "${env:ProgramFiles}\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
        "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
        "D:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe",
        "E:\Steam\steamapps\common\The Binding of Isaac Rebirth\isaac-ng.exe"
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { $Isaac = $c; break } }
    if ($Isaac -eq "") { Write-Error "Isaac binary not found. Pass -Isaac <path>."; exit 1 }
}
if (-not (Test-Path $Isaac)) { Write-Error "Isaac binary not found at: $Isaac"; exit 1 }

# ---- THE QUEUE -------------------------------------------------------------
# Each entry: a label (-> run_name) and the list of --override key=value pairs.
# run_name MUST be unique so run dirs don't collide. stage/r_hit/ent_coef are
# all real PPOConfig fields (train.py coerces int/float/bool).
$queue = @(
    @{ name = "sweep_hit03";        overrides = @("stage=0", "r_hit=0.3", "pbrs_coef=0.0") },
    @{ name = "sweep_hit03_ent001"; overrides = @("stage=0", "r_hit=0.3", "pbrs_coef=0.0", "ent_coef=0.001") },
    @{ name = "sweep_hit06";        overrides = @("stage=0", "r_hit=0.6", "pbrs_coef=0.0") },
    @{ name = "sweep_hit03_close";  overrides = @("stage=0", "r_hit=0.3", "pbrs_coef=0.0", "spawn_min=90", "spawn_max=170") }
)

Section "Isaac RL overnight queue"
Write-Host "isaac:      $Isaac"
Write-Host "per-run:    $Steps steps (~$([math]::Round($Steps/28.0/3600,1))h each)"
Write-Host "queue:      $($queue.Count) runs -> ~$([math]::Round($queue.Count*$Steps/28.0/3600,1))h total"
Write-Host "push:       $(-not $NoPush)"
$queue | ForEach-Object { Write-Host ("  - {0,-22} {1}" -f $_.name, ($_.overrides -join ' ')) }

# ---- Deploy mod ONCE -------------------------------------------------------
Section "Deploying mod"
$src = Join-Path $RepoRoot "mods\isaac-rl-bridge"
$dst = "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\mods\isaac-rl-bridge"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
Copy-Item -Recurse $src $dst
Write-Host "deployed: $dst" -ForegroundColor Green

function Get-LatestRunDir($runName) {
    $stageDir = Join-Path $RepoRoot "runs\$runName"
    if (-not (Test-Path $stageDir)) { return $null }
    $latest = Get-ChildItem -Path $stageDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { return $latest.FullName } else { return $null }
}

# ---- Run the queue ---------------------------------------------------------
$results = @()
$i = 0
foreach ($job in $queue) {
    $i++
    $runName = "cleanrl_ppo_$($job.name)"
    Section "[$i/$($queue.Count)] $runName"
    $ok = $true
    try {
        $ovr = @("total_env_steps=$Steps", "run_name=$runName") + $job.overrides
        $argList = @("train.py", "--config", $Config, "--isaac", $Isaac, "--override") + $ovr
        Write-Host "python $($argList -join ' ')" -ForegroundColor DarkGray
        & python @argList
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "[$runName] train.py exited $LASTEXITCODE (Ctrl+C or crash) — data may still be on disk."
        }
    } catch {
        $ok = $false
        Write-Warning "[$runName] threw: $($_.Exception.Message) — continuing to next queued run."
    }

    # ---- Export + push this run's data (even if it errored — partial data is useful) ----
    try {
        $runDir = Get-LatestRunDir $runName
        if ($runDir) {
            $summaryOut = Join-Path $RepoRoot "tb_$($job.name).json"
            & python export_tb_summary.py $runDir --out $summaryOut
            Write-Host "exported: $summaryOut" -ForegroundColor Green
            if (Test-Path (Join-Path $RepoRoot "scripts\_check_progress.py")) {
                & python scripts\_check_progress.py $summaryOut
            }
            if (-not $NoPush) {
                & .\scripts\push_data.ps1 -RunDir $runDir -Message "overnight sweep: $($job.name)"
            }
        } else {
            Write-Warning "[$runName] no run dir found under runs\$runName — nothing to export."
        }
    } catch {
        Write-Warning "[$runName] export/push failed: $($_.Exception.Message) — data is still on disk in runs\$runName."
    }

    $results += [pscustomobject]@{ run = $runName; ok = $ok }
    # Let Isaac processes fully exit + release ports before the next run binds them.
    Start-Sleep -Seconds $SettleSeconds
}

Section "Overnight queue complete"
$results | ForEach-Object { Write-Host ("  {0,-32} {1}" -f $_.run, ($(if($_.ok){"ok"}else{"errored"}))) }
Write-Host ""
Write-Host "Pushed tb_sweep_*.json for each run. Pull the repo and I'll analyze which config learned." -ForegroundColor Yellow
