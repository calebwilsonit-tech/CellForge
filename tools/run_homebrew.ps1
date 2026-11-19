<#
.SYNOPSIS
  Recompiles, builds and runs PS3 homebrew/test ELFs (that you provide) headless,
  sharing one build so only each program's generated code is recompiled.

.EXAMPLE
  .\tools\run_homebrew.ps1 -Dir C:\path\to\elfs -Names ppu_thread,gs_gcm_hello_world
#>
param(
  [Parameter(Mandatory = $true)] [string] $Dir,
  [string[]] $Names = @(),
  [int] $Seconds = 10,
  [switch] $Window
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$vsCmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$cmake = if (Get-Command cmake -ErrorAction SilentlyContinue) { "cmake" } else { $vsCmake }
$recomp = Join-Path $repo "build\Release\ps3recomp.exe"
$work = Join-Path $repo "out\homebrew"
$bld = Join-Path $work "build"
New-Item -ItemType Directory -Force -Path $work, (Join-Path $work "bin") | Out-Null

$elfs = Get-ChildItem (Join-Path $Dir "*.elf")
if ($Names.Count) { $elfs = $elfs | Where-Object { $Names -contains $_.BaseName } }

foreach ($e in $elfs) {
  $name = $e.BaseName
  $gen = Join-Path $work "gen\$name"
  Remove-Item -Recurse -Force $gen -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $gen | Out-Null
  $r = & $recomp $e.FullName --out-dir $gen --split 200 2>&1
  if ($LASTEXITCODE) { Write-Host "== $name : recompile failed"; $r | Select-Object -Last 3; continue }
  & $cmake -S $repo -B $bld -G "Visual Studio 17 2022" -A x64 "-DPS3RECOMP_GAME_DIR=$gen" | Out-Null
  $b = & $cmake --build $bld --config Release --target ps3game -- /m /v:minimal 2>&1
  if ($LASTEXITCODE) { Write-Host "== $name : build failed"; $b | Select-String "error" | Select-Object -First 5; continue }
  $exe = Join-Path $work "bin\$name.exe"
  Copy-Item (Join-Path $bld "Release\ps3game.exe") $exe -Force

  $out = Join-Path $work "$name.out.txt"; $err = Join-Path $work "$name.log.txt"
  $args = @("`"$($e.FullName)`"", "--no-mods", "--data", "`"$(Join-Path $work 'data')`"", "--game-root", "`"$Dir`"")
  if (-not $Window) { $args += "--headless" }
  $p = Start-Process -FilePath $exe -ArgumentList $args -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -NoNewWindow
  $timedOut = -not $p.WaitForExit($Seconds * 1000)
  if ($timedOut) { Stop-Process -Id $p.Id -Force }
  Write-Host ("== {0} : {1}" -f $name, $(if ($timedOut) { "still running after $Seconds s (stopped)" } else { "exit $($p.ExitCode)" }))
  Get-Content $err -TotalCount 25 | ForEach-Object { "   log| $_" }
  Get-Content $out -TotalCount 15 | ForEach-Object { "   out| $_" }
}
