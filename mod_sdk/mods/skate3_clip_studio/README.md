# Skate3-VK Clip Studio (`skate3_clip_studio`)

An instant lossless PNG screenshot and trick-sequence burst capture mod for **Skate3-VK**, built on the **CellForge Mod SDK**.

## Hotkeys
- **`F12`** — Capture a lossless PNG screenshot of the game window to `%USERPROFILE%\Pictures\Skate3-VK\`
- **`F11`** — Toggle **Trick-Sequence Burst Mode** (captures a rapid sequence of PNG frames into a timestamped subfolder)
- **`F5`** — Open the dark-mode **Clip Studio** quick panel (Take Screenshot, Toggle Burst, Open Pictures Folder)

## Configuration (`mods/skate3_clip_studio.ini`)
- `output_dir` — Destination folder (supports `%USERPROFILE%` and other Windows env vars)
- `burst_fps` — Frames per second during Burst Mode (1–15, default `5`)
- `watermark` — `1` to stamp a subtle `Skate3-VK` badge in the bottom-right corner, `0` for clean raw capture
- `shutter_beep` — `1` for subtle audio feedback on capture

## Build & Deploy
```powershell
..\..\tools\build_mod.ps1 -ModDir . -DeployDir "$env:LOCALAPPDATA\Games\Skate3-VK\mods"
```
