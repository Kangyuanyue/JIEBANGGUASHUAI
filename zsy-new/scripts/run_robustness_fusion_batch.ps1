param(
  [string]$Python = "python",
  [string]$ProjectRoot = "",
  [string]$AudioRoot = "output/augmented_speaker_audio"
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
  $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

Set-Location $ProjectRoot

$scenarios = @(
  @{Name = "clean"; Trials = "output/cnceleb2_clean_trials_1000.csv"},
  @{Name = "noise5"; Trials = "output/aug_trials_noise5_1000.csv"},
  @{Name = "noise0"; Trials = "output/aug_trials_noise0_1000.csv"},
  @{Name = "noise-5"; Trials = "output/aug_trials_noise-5_1000.csv"},
  @{Name = "rir"; Trials = "output/aug_trials_rir_1000.csv"},
  @{Name = "rir_noise0"; Trials = "output/aug_trials_rir_noise0_1000.csv"}
)

$resultDir = "output/robustness_fusion"
$logDir = Join-Path $resultDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($scenario in $scenarios) {
  if (-not (Test-Path -LiteralPath $scenario.Trials)) {
    Write-Warning "Skip $($scenario.Name): trial file not found: $($scenario.Trials)"
    continue
  }

  $name = $scenario.Name
  $log = Join-Path $logDir "$name.log"
  $err = Join-Path $logDir "$name.err"

  Write-Output "=== START $name $(Get-Date -Format o) ==="
  & $Python scripts/evaluate_speaker_trials_cached.py `
    --trials $scenario.Trials `
    --audio-root $AudioRoot `
    --config configs/final_robust_speaker_fusion.json `
    --output "$resultDir/$name`_eval.json" `
    --score-dump "$resultDir/$name`_scores.json" `
    --progress-every 100 `
    1> $log 2> $err

  if ($LASTEXITCODE -ne 0) {
    throw "Scenario $name failed. See $err"
  }
  Write-Output "=== END $name $(Get-Date -Format o) ==="
}

