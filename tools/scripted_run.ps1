# Scripted Skate 3 run: boots, walks the menus to "Play Skate 3", then waits.
# Usage: tools\scripted_run.ps1 -Name auto7 -Seconds 155 -Env @{PS3RECOMP_HLE_TRACE='130'}
param(
  [string]$Name = "run",
  [int]$Seconds = 155,
  [hashtable]$Env = @{},
  [switch]$NoBuild,
  # Save/cache folder. Default: a fresh one per run (out\skate3\runs\<Name>\data),
  # so runs never touch out\skate3\data and always start a new career.
  [string]$Data = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$cmake = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
if (-not $NoBuild) {
  & $cmake --build out\skate3\build --config Release --target ps3game -- /m 2>&1 | Select-String -Pattern "error|warning C" | Select-Object -First 10
}
$root = "C:\Users\wiked\Documents\RCPS3\Roms\Skate 3 (USA, Asia) (En,Fr,Es)"
if (-not $Data) {
  $Data = "out\skate3\runs\$Name-$(Get-Date -Format yyyyMMdd-HHmmss)\data"
  New-Item -ItemType Directory -Force (Split-Path $Data) | Out-Null
  # Seed: a profile without a career (the route below assumes the new-career flow).
  if (Test-Path out\skate3\data_seed) { Copy-Item out\skate3\data_seed $Data -Recurse }
  else { New-Item -ItemType Directory -Force $Data | Out-Null }
}
$presses = @('start@30', 'start@35')
for ($t = 45; $t -le 101; $t += 8) { $presses += "cross@$t" }
$presses += 'down@105', 'cross@109'
$vars = @{ PS3RECOMP_INPUT_SCRIPT = ($presses -join ',') } + $Env
foreach ($k in $vars.Keys) { Set-Item "Env:$k" $vars[$k] }
try {
  $p = Start-Process -FilePath out\skate3\build\Release\ps3game.exe `
    -ArgumentList "`"$root\PS3_GAME\USRDIR\EBOOT.elf`"", "--game-root", "`"$root`"", "--data", "`"$Data`"" `
    -RedirectStandardOutput "out\skate3\$Name.out.txt" -RedirectStandardError "out\skate3\$Name.err.txt" -PassThru
  if (-not $p.WaitForExit($Seconds * 1000)) { $p.Kill() } else { "exited: $($p.ExitCode)" }
} finally {
  foreach ($k in $vars.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
}
