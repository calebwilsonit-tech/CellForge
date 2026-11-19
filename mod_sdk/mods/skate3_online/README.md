# Skate3-VK Online Mod (Path A: Direct P2P / UDP Relay)

A native Windows Winsock2 UDP multiplayer mod for **Skate3-VK** built on the **CellForge Mod SDK** — no PS3 `sys_net` or EA Blaze servers required.

## Features
- **Direct P2P (`mode=host` / `mode=join`)** — Works over LAN, port forwarding (UDP `37464`), or zero-config mesh VPNs like **Tailscale** / **ZeroTier**.
- **Room-Code Relay (`mode=relay`)** — Run `python relay_server.py --port 37464` on any reachable PC/VPS and have friends join the same `room=SK8A` code without port forwarding.
- **In-Game Lobby & Session Panel (press `F6`)** — View connected players & live RTT, send session chat messages, drop Session Marker callouts, and switch mode/host/room & reconnect live without restarting the game.
- **60 Hz Skater Kinematics + Quaternion NLERP** — Smooth position, velocity dead-reckoning, and shortest-path quaternion interpolation for remote skaters.
- **Settings → Mods Integration** — Enable/disable or edit `skate3_online.ini` directly from CellForge's `F1` Settings menu.

## Build & Deploy
```powershell
..\..\tools\build_mod.ps1 -ModDir . -DeployDir "$env:LOCALAPPDATA\Games\Skate3-VK\mods"
```
Copy `skate3_online.ini` alongside `skate3_online.dll` into your game's `mods/` folder.

## Architecture
- `protocol.h` — Compact binary UDP packets (`S3ON` magic, `Hello`, `Welcome`, 60 Hz `State`, `Event` [Chat / Marker / Trick / PeerInfo], `Ping`/`Pong`, `Leave`).
- `net_client.h/.cpp` — Non-blocking Winsock2 UDP engine on a dedicated background thread with host relay, peer name sync, RTT measurement, and quaternion NLERP interpolation.
- `skate3_online.cpp` — CellForge mod DLL (`ps3recomp_mod_init`) with F6 dark-mode lobby UI and optional BLUS30464 guest RAM entity bridge.
- `relay_server.py` — Zero-dependency Python room-code UDP relay server with automatic peer roster announcements.
- `test_net.cpp` — Automated Host + Client localhost UDP loopback test.
