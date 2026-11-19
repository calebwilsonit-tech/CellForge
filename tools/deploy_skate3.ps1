# Incremental build & standalone deploy script for Skate3-VK (CellForge).
# Updates Skate3-VK.exe in ~7 seconds without recompiling generated PPU code
# and without touching persistent saves (%USERPROFILE%\Saved Games\Skate3-VK).
param(
  [string]$DeployDir = "out\skate3\Skate3-VK",
  [string]$DiscRoot = "C:\Users\wiked\Documents\RCPS3\Roms\Skate 3 (USA, Asia) (En,Fr,Es)",
  [string]$ElfPath = ""
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo

$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$bld = Join-Path $Repo "out\skate3\build"
$ico = Join-Path $Repo "out\skate3\Skate3-VK.ico"

if (-not $ElfPath) {
  $candidates = @(
    (Join-Path $DiscRoot "PS3_GAME\USRDIR\EBOOT.ELF"),
    (Join-Path $Repo "out\skate3\EBOOT.ELF")
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { $ElfPath = $c; break }
  }
}

# Ensure CMake cache has Skate3-VK output name & embedded icon without rebuilding ps3game_code
$cache = Get-Content (Join-Path $bld "CMakeCache.txt") -Raw
if ($cache -notmatch "PS3RECOMP_GAME_NAME:STRING=Skate3-VK") {
  Write-Host "== Configuring Skate3-VK name & icon..."
  & $cmake -S $Repo -B $bld "-DPS3RECOMP_GAME_NAME=Skate3-VK" "-DPS3RECOMP_GAME_ICON=$ico"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "== 1/4 Building ps3_spu & ps3_gfx..."
& $cmake --build $bld --config Release --target ps3_spu ps3_gfx -- /m:4 /v:minimal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 2/4 Building ps3_runtime..."
& $cmake --build $bld --config Release --target ps3_runtime -- /m:4 /p:BuildProjectReferences=false /v:minimal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 3/4 Linking Skate3-VK.exe..."
& $cmake --build $bld --config Release --target ps3game -- /m:4 /p:BuildProjectReferences=false /v:minimal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exeSrc = Join-Path $bld "Release\Skate3-VK.exe"
if (-not (Test-Path $exeSrc)) { throw "Missing $exeSrc" }

$dest = if ([IO.Path]::IsPathRooted($DeployDir)) { $DeployDir } else { Join-Path $Repo $DeployDir }
New-Item -ItemType Directory -Force $dest | Out-Null
New-Item -ItemType Directory -Force (Join-Path $dest "mods") | Out-Null
Copy-Item $exeSrc (Join-Path $dest "Skate3-VK.exe") -Force

$saves = Join-Path $env:USERPROFILE "Saved Games\Skate3-VK"
New-Item -ItemType Directory -Force $saves | Out-Null

$iniPath = Join-Path $dest "Skate3-VK.ini"
if (-not (Test-Path $iniPath)) {
  @(
    "# Skate3-VK (CellForge) standalone configuration"
    "# Updates only replace Skate3-VK.exe; your saves stay safe in Saved Games."
    "title=Skate3-VK"
    "disc=$DiscRoot"
    "elf=$ElfPath"
    "data=%USERPROFILE%\Saved Games\Skate3-VK"
    "icon=$DiscRoot\PS3_GAME\ICON0.PNG"
  ) | Set-Content -Path $iniPath -Encoding UTF8
  Write-Host "Created $iniPath"
}

Write-Host ""
Write-Host "Deployed standalone Skate3-VK to: $dest"
Write-Host "Persistent saves folder:          $saves"
