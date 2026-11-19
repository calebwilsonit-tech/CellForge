# Scaffolds a new standalone CellForge mod project.
# Usage:
#   .\tools\new_mod.ps1 -Name MySkateMod
#   .\tools\new_mod.ps1 -Name MySkateMod -OutDir C:\Users\you\MyMods
param(
  [Parameter(Mandatory = $true)]
  [string]$Name,
  [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$SdkRoot = Split-Path $PSScriptRoot -Parent
$SafeName = ($Name -replace '[^A-Za-z0-9_]', '_')
if (-not $OutDir) { $OutDir = Join-Path $SdkRoot "mods_src" }
$ModDir = Join-Path $OutDir $SafeName

if (Test-Path $ModDir) {
  throw "Folder already exists: $ModDir"
}
New-Item -ItemType Directory -Force -Path $ModDir | Out-Null

$SdkCmakePath = ($SdkRoot -replace '\\', '/')

@"
cmake_minimum_required(VERSION 3.20)
project($SafeName LANGUAGES CXX)

set(CELLFORGE_BUILD_SDK_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("$SdkCmakePath" "${CMAKE_BINARY_DIR}/cellforge_mod_sdk")

cellforge_add_mod($SafeName $SafeName.cpp)
"@ | Set-Content -Path (Join-Path $ModDir "CMakeLists.txt") -Encoding UTF8

@"
// $SafeName — CellForge Mod
#include "cellforge_mod.h"

namespace {

// Example hook slot:
// cellforge::Hook g_example_hook;
// void my_hook(ppu::Context& c) {
//   g_example_hook.call_original(c);
//   // Modify registers (c.r[3], c.f[1], ...) or guest memory here.
// }

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  cellforge::Config cfg;
  cfg.load("mods/$SafeName.ini");

  cellforge::log("$SafeName: loaded!");
  return 0;
}
"@ | Set-Content -Path (Join-Path $ModDir "$SafeName.cpp") -Encoding UTF8

@"
# Configuration for $SafeName (copy to your game's mods/ folder)
enabled=1
"@ | Set-Content -Path (Join-Path $ModDir "$SafeName.ini") -Encoding UTF8

Write-Host "Created mod project at: $ModDir"
Write-Host "Build & deploy with:"
Write-Host "  `"$SdkRoot\tools\build_mod.ps1`" -ModDir `"$ModDir`" -DeployDir `"$env:LOCALAPPDATA\Games\Skate3-VK\mods`""
