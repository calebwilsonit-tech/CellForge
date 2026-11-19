# Skate3-VK Online Mod (Path A: Direct P2P / UDP Relay)

A native Windows Winsock2 UDP multiplayer mod for **Skate3-VK** built on the **CellForge Mod SDK** — no PS3 `sys_net` or EA Blaze servers required.

## Modes
1. **Direct P2P (`mode=host` / `mode=join`)** — Works over LAN, port forwarding (UDP `37464`), or zero-config mesh VPNs like **Tailscale** / **ZeroTier**.
2. **Room-Code Relay (`mode=relay`)** — Run `python relay_server.py --port 37464` on any reachable PC/VPS and have friends set the same `room=SK8A` code in `mods/skate3_online.ini`.

## Build & Deploy
```powershell
..\..\tools\build_mod.ps1 -ModDir . -DeployDir "$env:LOCALAPPDATA\Games\Skate3-VK\mods"
```

## Architecture
- `protocol.h` — Compact binary UDP packets (`S3ON` magic, `Hello`, `Welcome`, 60 Hz `State` kinematics, `Event`, `Ping`/`Pong`, `Leave`).
- `net_client.h/.cpp` — Non-blocking Winsock2 UDP engine on a dedicated background thread with automatic host relay, out-of-order drop, RTT measurement, and linear state interpolation.
- `skate3_online.cpp` — CellForge mod DLL (`ps3recomp_mod_init`) bridging `SkaterState` to/from PS3 guest RAM via `mods/skate3_online.ini`.
- `relay_server.py` — Zero-dependency Python room-code UDP relay server.
- `test_net.cpp` — Automated Host + Client localhost UDP loopback test.
