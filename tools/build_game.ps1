<#
.SYNOPSIS
  One-command pipeline: recompile a decrypted PS3 ELF, build ps3game.exe, and
  optionally run it.

.DESCRIPTION
  This script never decrypts anything. You supply a plain ELF that you produced
  yourself from your own copy of the game. If the file is still a signed/encrypted
  SELF (EBOOT.BIN as shipped), the script stops and tells you so.

.EXAMPLE
  .\tools\build_game.ps1 -Elf "D:\dumps\EBOOT.ELF" -GameRoot "C:\Games\Skate 3" -Run
#>
param(
  [Parameter(Mandatory = $true)] [string] $Elf,       # decrypted ELF (you provide it)
  [string] $GameRoot = "",                             # folder containing PS3_GAME
  [string] $OutDir = "",                               # where generated code + build go
  [int]    $Split = 500,                               # functions per generated file
  [switch] $Run,
  [switch] $Strict
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

# ---- tools -------------------------------------------------------------------
$vsCmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$cmake = if (Get-Command cmake -ErrorAction SilentlyContinue) { "cmake" } elseif (Test-Path $vsCmake) { $vsCmake } else { throw "cmake not found" }

# ---- check the input ---------------------------------------------------------
if (-not (Test-Path -LiteralPath $Elf)) { throw "ELF not found: $Elf" }
$magic = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Elf))[0..3]
$hex = ($magic | ForEach-Object { $_.ToString("X2") }) -join ""
if ($hex -eq "53434500") {
  Write-Host -ForegroundColor Red ("'$Elf' is a signed/encrypted SELF (as shipped on the disc). This pipeline only " +
               "accepts a decrypted ELF that you produce yourself; it does not decrypt files.")
  exit 1
}
if ($hex -ne "7F454C46") { Write-Host -ForegroundColor Red "'$Elf' is not an ELF file (magic $hex)."; exit 1 }

if (-not $OutDir) { $OutDir = Join-Path $repo "out\game" }
$gen = Join-Path $OutDir "gen"
$bld = Join-Path $OutDir "build"
New-Item -ItemType Directory -Force -Path $gen | Out-Null

# ---- 1. build the recompiler (if needed) ---------------------------------------
$toolBuild = Join-Path $repo "build"
$recomp = Join-Path $toolBuild "Release\ps3recomp.exe"
if (-not (Test-Path $recomp)) {
  & $cmake -S $repo -B $toolBuild -G "Visual Studio 17 2022" -A x64 | Out-Null
  & $cmake --build $toolBuild --config Release --target ps3recomp
  if ($LASTEXITCODE) { throw "building ps3recomp failed" }
}

# ---- 2. recompile ----------------------------------------------------------------
Write-Host "== imports"
& $recomp $Elf --imports | Select-Object -Last 1
Write-Host "== recompiling"
Get-ChildItem $gen -Filter *.cpp -ErrorAction SilentlyContinue | Remove-Item -Force
& $recomp $Elf --out-dir $gen --split $Split | Tee-Object -FilePath (Join-Path $OutDir "recomp.log") |
  Select-String -Pattern "function\(s\)|OPD scan|entry OPD"
if ($LASTEXITCODE) { throw "ps3recomp failed (see $OutDir\recomp.log)" }

# ---- 3. build ps3game ------------------------------------------------------------
Write-Host "== building ps3game"
& $cmake -S $repo -B $bld -G "Visual Studio 17 2022" -A x64 "-DPS3RECOMP_GAME_DIR=$gen" | Out-Null
& $cmake --build $bld --config Release --target ps3game -- /m /v:minimal
if ($LASTEXITCODE) { throw "building ps3game failed" }
$exe = Join-Path $bld "Release\ps3game.exe"
Write-Host "built $exe"

# ---- 4. run ------------------------------------------------------------------------
if ($Run) {
  $args = @($Elf, "--data", (Join-Path $OutDir "ps3_data"))
  if ($GameRoot) { $args += @("--game-root", $GameRoot) }
  if ($Strict) { $args += "--strict" }
  & $exe @args
  Write-Host "exit code $LASTEXITCODE"
}
