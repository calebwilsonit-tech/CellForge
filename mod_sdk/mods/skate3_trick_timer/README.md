# Skate3-VK Speedrun Split Timer & Line Tracker (`skate3_trick_timer`)

A sleek, draggable, always-on-top dark-mode speedrun & skate-line split timer HUD built on the **CellForge Mod SDK**.

## Hotkeys
- **`F7`** — Show / Hide the HUD overlay
- **`F8`** — Start run / Record Split
- **`F9`** — Pause / Resume
- **`F10`** — Reset run

## Features
- Millisecond-precision `MM:SS.mmm` live timer with split table
- Automatic **Personal Best (PB)** persistence (`mods/skate3_trick_timer_pb.txt`) with green/red split deltas (`-0.42s` / `+0.18s`)
- Customizable split names and window opacity in `mods/skate3_trick_timer.ini`
- Drag the HUD anywhere on screen by clicking and dragging it

## Build & Deploy
```powershell
..\..\tools\build_mod.ps1 -ModDir . -DeployDir "$env:LOCALAPPDATA\Games\Skate3-VK\mods"
```
