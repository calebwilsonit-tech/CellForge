# Builds a CellForge mod DLL (or the SDK examples) and optionally deploys to a game's mods/ folder.
# Usage:
#   .\tools\build_mod.ps1                                                # builds SDK examples
#   .\tools\build_mod.ps1 -ModDir .\mods_src\MySkateMod -DeployDir "$env:LOCALAPPDATA\Games\Skate3-VK\mods"
param(
  [string]$ModDir = "",
  [string]$DeployDir = ""
)

$ErrorActionPreference = "Stop"
$SdkRoot = Split-Path $PSScriptRoot -Parent
if (-not $ModDir) { $ModDir = $SdkRoot }
$ModDir = (Resolve-Path $ModDir).Path

function Find-CMake {
  $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  foreach ($root in @("C:\Program Files\Microsoft Visual Studio\2022", "C:\Program Files (x86)\Microsoft Visual Studio\2022")) {
    foreach ($ed in @("BuildTools", "Community", "Professional", "Enterprise")) {
      $p = Join-Path $root "$ed\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
      if (Test-Path $p) { return $p }
    }
  }
  throw "cmake.exe not found. Install Visual Studio 2022 Build Tools with 'Desktop development with C++'."
}

$cmake = Find-CMake
$bld = Join-Path $ModDir "build"

Write-Host "== Configuring $ModDir..."
& $cmake -S $ModDir -B $bld -G "Visual Studio 17 2022" -A x64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== Building Release DLL(s)..."
& $cmake --build $bld --config Release -- /m:4 /v:minimal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$dlls = Get-ChildItem (Join-Path $bld "Release") -Filter "*.dll" -ErrorAction SilentlyContinue
if (-not $dlls) {
  Write-Warning "No .dll files found in $bld\Release"
  exit 0
}

Write-Host "`nBuilt mod DLL(s):"
foreach ($d in $dlls) {
  Write-Host "  $($d.FullName) ($($d.Length) bytes)"
}

if ($DeployDir) {
  New-Item -ItemType Directory -Force -Path $DeployDir | Out-Null
  foreach ($d in $dlls) {
    Copy-Item $d.FullName (Join-Path $DeployDir $d.Name) -Force
    Write-Host "Deployed -> $(Join-Path $DeployDir $d.Name)"
  }
  $inis = Get-ChildItem $ModDir -Filter "*.ini" -ErrorAction SilentlyContinue
  foreach ($ini in $inis) {
    $dst = Join-Path $DeployDir $ini.Name
    if (-not (Test-Path $dst)) {
      Copy-Item $ini.FullName $dst
      Write-Host "Copied default config -> $dst"
    }
  }
}
