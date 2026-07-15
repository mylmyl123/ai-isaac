# Isaac RL — end-to-end training automation.
#
# One command runs the whole Phase-2 pipeline:
#   1. Deploy the mod into Isaac's user mods folder (copy mods\isaac-rl-bridge).
#   2. Run a SHORT smoke run (default 8k steps) to populate log.txt + episodes.csv.
#   3. Run the smoke GATE (tools\smoke_gate.py): asserts the stage spawns AND
#      kills the intended enemy (Horf=12 on stage 0), enough episodes, clean
#      deaths. Aborts the pipeline if the gate fails — a plain "kills rose"
#      check would false-pass the wrong-enemy bug.
#   4. Run the FULL training run (default 200k steps) with TensorBoard.
#   5. Export the TB summary + copy the checkpoint (via export_tb_summary.py).
#      Optionally push (-Push) via scripts\push_data.ps1.
#
# Usage:
#   .\scripts\run_training.ps1                          # stage from config, full pipeline
#   .\scripts\run_training.ps1 -Stage 0                 # force Horf control task
#   .\scripts\run_training.ps1 -SkipSmoke               # skip smoke run+gate (danger)
#   .\scripts\run_training.ps1 -SmokeOnly               # smoke run + gate, no full run
#   .\scripts\run_training.ps1 -Push                    # auto-push data at the end
#   .\scripts\run_training.ps1 -Isaac "C:\...\isaac-ng.exe"
#   .\scripts\run_training.ps1 -SmokeSteps 10000 -FullSteps 300000
#
# Isaac binary is auto-detected from standard Steam locations if -Isaac omitted.
# Ctrl+C during a training phase stops that phase; the pipeline halts.

param(
    [string]$Config = "configs\curriculum.yaml",
    [string]$Stage = "",              # "" = use whatever the config says
    [string]$Isaac = "",
    [int]$SmokeSteps = 8000,
    [int]$FullSteps = 200000,
    [int]$SmokeMinEpisodes = 3,
    [switch]$SkipSmoke,
    [switch]$SmokeOnly,
    [switch]$Push,
    [switch]$NoTensorboard
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (Test-Path .\.venv\Scripts\Activate.ps1) { . .\.venv\Scripts\Activate.ps1 }
$env:PYTHONPATH = "python"

function Section($msg) {
    Write-Host ""
    Write-Host "==== $msg ====" -ForegroundColor Cyan
}

# ---- Resolve Isaac binary (same locations as record.ps1) ------------------
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

# ---- Resolve the stage that will actually run (for the smoke gate's -----
# ---- expected-enemy-type). Stage 0 = Horf (12); everything else = AttackFly (18).
$effectiveStage = $Stage
if ($effectiveStage -eq "") {
    $cfgText = Get-Content $Config -Raw
    if ($cfgText -match '(?m)^\s*stage:\s*"?([^"\r\n]+)"?') { $effectiveStage = $Matches[1].Trim() }
    else { $effectiveStage = "0" }
}
$expectedEnemy = if ($effectiveStage -eq "0") { 12 } else { 18 }

Section "Isaac RL training pipeline"
Write-Host "isaac:          $Isaac"
Write-Host "config:         $Config"
Write-Host "stage:          $effectiveStage  (expected enemy type=$expectedEnemy)"
Write-Host "smoke steps:    $SmokeSteps"
Write-Host "full steps:     $FullSteps"

# ---- 1. Deploy the mod ----------------------------------------------------
Section "1/5  Deploying mod"
$src = Join-Path $RepoRoot "mods\isaac-rl-bridge"
$dst = "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\mods\isaac-rl-bridge"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
Copy-Item -Recurse $src $dst
Write-Host "deployed: $dst" -ForegroundColor Green

$isaacLog = "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\log.txt"

# ---- helper: find newest run dir for this stage under runs\ ---------------
function Get-LatestRunDir($stageArg) {
    $runName = "cleanrl_ppo_stage$stageArg"
    $stageDir = Join-Path $RepoRoot "runs\$runName"
    if (-not (Test-Path $stageDir)) { return $null }
    $latest = Get-ChildItem -Path $stageDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { return $latest.FullName } else { return $null }
}

# ---- helper: run one training invocation ----------------------------------
# Always sets an explicit run_name derived from effectiveStage so the run dir
# is deterministic (runs\cleanrl_ppo_stage<X>[_suffix]\<ts>) regardless of what
# run_name the config carries or which -Stage is forced.
function Invoke-Train($steps, $runNameSuffix, $withTb) {
    $runName = "cleanrl_ppo_stage$effectiveStage"
    if ($runNameSuffix) { $runName = "${runName}_$runNameSuffix" }
    $overrides = @("total_env_steps=$steps", "run_name=$runName")
    if ($Stage -ne "") { $overrides += "stage=$Stage" }
    $argList = @("train.py", "--config", $Config, "--isaac", $Isaac)
    if ($withTb) { $argList += "--tensorboard" }
    $argList += "--override"; $argList += $overrides
    Write-Host "python $($argList -join ' ')" -ForegroundColor DarkGray
    & python @argList
    return $LASTEXITCODE
}

# ---- 2. Smoke run ---------------------------------------------------------
if (-not $SkipSmoke) {
    Section "2/5  Smoke run ($SmokeSteps steps)"
    $code = Invoke-Train $SmokeSteps "smoke" $false
    if ($code -ne 0) { Write-Error "Smoke run exited with code $code — aborting."; exit $code }

    # ---- 3. Smoke GATE ----------------------------------------------------
    Section "3/5  Smoke gate (identity assertions)"
    $smokeRunDir = Get-LatestRunDir "${effectiveStage}_smoke"
    if (-not $smokeRunDir) { Write-Error "Could not find smoke run dir under runs\ — did train.py write episodes.csv?"; exit 1 }
    & python tools\smoke_gate.py --run-dir $smokeRunDir --isaac-log $isaacLog --expected-enemy-type $expectedEnemy --min-episodes $SmokeMinEpisodes
    $gate = $LASTEXITCODE
    if ($gate -ne 0) {
        Write-Host ""
        Write-Error "SMOKE GATE FAILED — not launching the full run. Fix the failed checks first."
        exit $gate
    }
    # Let the smoke run's Isaac processes fully exit and release ports before
    # the full run binds the same base_port. train.py shuts its fleet down on
    # exit; this is a settle margin against port-bind races.
    Start-Sleep -Seconds 5
} else {
    Write-Host ""
    Write-Warning "Skipping smoke run + gate (-SkipSmoke). The full run is NOT identity-verified."
}

if ($SmokeOnly) {
    Section "Done (smoke only)"
    Write-Host "Smoke run + gate complete. Re-run without -SmokeOnly to launch the full run." -ForegroundColor Green
    exit 0
}

# ---- 4. Full run ----------------------------------------------------------
Section "4/5  Full training run ($FullSteps steps)"
$code = Invoke-Train $FullSteps "" (-not $NoTensorboard)
if ($code -ne 0) { Write-Warning "Full run exited with code $code (Ctrl+C is fine — data is still on disk)." }

# ---- 5. Export / push -----------------------------------------------------
Section "5/5  Export data"
$fullRunDir = Get-LatestRunDir $effectiveStage
if ($fullRunDir) {
    $summaryOut = Join-Path $RepoRoot "tb_latest.json"
    & python export_tb_summary.py $fullRunDir --out $summaryOut
    Write-Host "TB summary: $summaryOut" -ForegroundColor Green
    if (Test-Path (Join-Path $RepoRoot "scripts\_check_progress.py")) {
        & python scripts\_check_progress.py $summaryOut
    }
    if ($Push) {
        Section "Pushing data"
        & .\scripts\push_data.ps1 -RunDir $fullRunDir
    } else {
        Write-Host ""
        Write-Host "To push this run's data:  .\scripts\push_data.ps1 -RunDir `"$fullRunDir`"" -ForegroundColor Yellow
    }
} else {
    Write-Warning "No full run dir found under runs\ — nothing to export."
}

Section "Pipeline complete"
