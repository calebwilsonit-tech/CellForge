# CellForge

A static recompiler and native Windows x64 + Vulkan runtime for the PlayStation 3 Cell Broadband Engine (PowerPC 64 + AltiVec PPU, SPU, and RSX GPU). CellForge lifts PPU code from a plain, already-decrypted PS3 ELF into C++ and compiles it into a standalone native Windows executable — such as **Skate3-VK** for *Skate 3* (`BLUS30464`).

> **Legal / Scope Note:** No game files, encryption keys, or decryption code are included or supported. Encrypted `EBOOT.BIN` (SELF) files are rejected up front; you must provide your own dumped `PS3_GAME` folder and decrypted `EBOOT.ELF`.

---

## Quick Start (Standalone Installer)

The easiest way to build and play a detached game is with the single-file Windows installer from [GitHub Releases](https://github.com/calebwilsonit-tech/CellForge/releases/latest):

1. Download and run **`CellForge-Setup.exe`** from the [Releases page](https://github.com/calebwilsonit-tech/CellForge/releases).
2. **Prerequisites:** CellForge checks for its source tree, CMake, Visual Studio 2022 C++ Build Tools, and `ps3recomp.exe`, and can download/install anything missing automatically.
3. **Select Paths:**
   - **Game Folder:** your dumped folder containing `PS3_GAME/`
   - **Decrypted ELF:** your decrypted `EBOOT.ELF`
   - **Install Folder:** any folder where you want the finished, detached game saved
4. Press **Build**. CellForge recompiles the ELF, embeds `PS3_GAME/ICON0.PNG` as the executable icon, and deploys a completely standalone game folder (e.g. `Skate3-VK.exe`, `Skate3-VK.ini`, `mods/`).
5. **Play:** Launch from CellForge or double-click `Skate3-VK.exe` directly — once built, the game has **zero runtime dependency** on CellForge or the compiler (only standard GPU Vulkan drivers and your `PS3_GAME` assets).

### Persistent Saves & Fast Updates
- Save data is stored under `%USERPROFILE%\Saved Games\Skate3-VK` (or `%USERPROFILE%\Saved Games\CellForge\<TITLE_ID>`), completely separate from build folders so updates never wipe your progress.
- **Update Runtime** (in `CellForge-Setup.exe` or via `tools\deploy_skate3.ps1`) rebuilds only the SPU/Vulkan/OS runtime and re-links the game in **~7 seconds** without recompiling the generated PPU C++ or touching your saves or `.ini`.

---

## Building from Source

Requires Windows x64, Visual Studio 2022 (or Build Tools with the *Desktop development with C++* workload), and CMake.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

This builds:
- `build\Release\CellForge-Setup.exe` — dark-mode Win32 + GDI+ installer, updater, and launcher GUI (displays live `PIC1.PNG` / `ICON0.PNG` artwork)
- `build\Release\ps3recomp.exe` — PPU static recompiler CLI
- `build\Release\ps3build.exe` — headless builder CLI

### Test Suites (`ctest`)
All 5 suites run via `ctest`:
- `unit_tests` — PPU decoder table round-trip, VMX/FP/carry helper semantics
- `spu` — SPU ISA, AVX2 JIT, and interpreter verification
- `boot` — LV2 kernel, VFS, RSX GPU vs. CPU reference parity, `cellAudio`, SPU thread groups, and SPURS
- `builder` — builder validation (SELF rejection, `PARAM.SFO`), settings round-trip, and environment detection
- `recomp_roundtrip` — lifts a generated test ELF with `ps3recomp`, compiles `ps3_test_run`, and verifies jump tables, FP, VMX, atomics, OPD discovery, hooks, and mod DLL loading

`PS3RECOMP_AVX2` (default `ON`) enables AVX2/FMA code generation.

---

## CLI & Script Usage

### Recompiler CLI
```powershell
ps3recomp input.elf -o out.cpp
ps3recomp input.elf --out-dir gen --split 500     # balanced chunks for parallel MSVC builds
ps3recomp input.elf --imports                     # list system-library imports (module, NID, name)
          [--func 0xADDR[:SIZE]]... [--func-file FILE] [--no-opd-scan] [--no-hooks] [--disasm]
```

### One-Command Game Build
```powershell
.\tools\build_game.ps1 -Elf <path\to\EBOOT.ELF> -GameRoot "<folder with PS3_GAME>" -Run
```

### Fast Incremental Deploy (`Skate3-VK`)
```powershell
.\tools\deploy_skate3.ps1
```

### Headless Builder CLI
```powershell
build\Release\ps3build.exe --game "<game folder>" --elf "<decrypted ELF>" [--play] [--validate-only]
```

### Running a Standalone Game Directly
Double-clicking `Skate3-VK.exe` auto-discovers `Skate3-VK.ini`, `cellforge.ini`, or `ps3game.ini` next to the executable (supporting `%ENVVAR%` expansion). You can also pass explicit arguments:

```powershell
Skate3-VK.exe <EBOOT.ELF> [--game-root <dir>] [--data <saves dir>] [--mods <dir> | --no-mods]
              [--title "<window title>"] [--icon <path>] [--strict] [--headless] [--cpu-render] [-- args...]
```

---

## Architecture & What Works

- **PPU Static Recompiler (`src/ppu`, `src/tools`)**:
  - Integer (including carry/overflow, 64-bit rotates, multiply-high, divide, CLZ, byte-reversed and load/store-multiple), CR logic, traps, time base, `lwarx`/`stwcx.` compare-and-swap atomics, scalar FP (single/double, FMA, conversions, FPSCR), and the full AltiVec/VMX set plus Cell `lvlx`/`lvrx`/`stvlx`/`stvrx`.
  - Function discovery via entry OPD, full-image OPD scan, `bl` targets, and explicit `--func` seeds; absolute and TOC-relative `switch` jump tables.
  - Incremental `--split N` output (only changed chunks are rewritten). *Skate 3* (`BLUS30464`): 83,310 functions, 0 warnings, 211 files.
- **SPU AVX2 JIT & Interpreter (`src/spu`, `src/runtime/lv2_spu.cpp`, `hle_spurs.cpp`)**:
  - Full SPU ISA interpreter paired with a native **x86-64 AVX2 SPU JIT** (`spu_jit.cpp`) featuring per-4KB-page Local Store invalidation, a 16 MB code buffer, and a content-addressed block cache for fast SPURS job switching and VP6 video/audio decoding.
  - LV2 SPU thread groups (`sys_spu_image_import`, create/start/join/terminate, MFC DMA GET/PUT/lists, `GETLLAR`/`PUTLLC`/`PUTLLUC` lock-line atomics, mailboxes, signal notification, `0xF0000000` LS window, event ports/queues) and SPURS workload scheduling.
- **RSX Vulkan GPU Renderer & Presenter (`src/runtime/rsx*.cpp`, `spirv_gen.cpp`, `vk_renderer.cpp`, `src/gfx`)**:
  - Direct RSX vertex- and fragment-program translation to SPIR-V (no external shader compiler), including GPU vertex program flow control (`BRA`/`BRI`/`BRB`/`RET`), `DDX`/`DDY`, hardware `D32_SFLOAT_S8_UINT` depth+stencil, dynamic VSM character/NPC shadows + full-distance shadow cascades (`D24S8` stencil sampling and `A8B8G8R8 -> Y16_X16` conversion), and native BC1–BC3 texture uploads.
  - High-throughput draw path (~54–60 FPS in *Skate 3* gameplay on RTX 4060 Laptop GPU): `UNIFORM_BUFFER_DYNAMIC` with per-frame descriptor set caching, deduplicated Vulkan pipeline/state commands, O(1) u16 index remapping & specialized vertex packing, surface address-range culling, GPU-resident render targets with lazy readback, and automatic fallback to the CPU reference renderer (`--cpu-render`).
  - Vulkan window presenter (`vk_presenter.cpp`) preferring discrete GPUs, with `ICON0.PNG` window/taskbar icons, live FPS title counter, and Windows 11 dark title bar support.
- **Universal Controller Input (`src/runtime/input.cpp`, `hid_pad.cpp`, `hle_pad.cpp`)**:
  - Works out of the box with **any** USB or Bluetooth controller on Windows — no Steam Input or DS4Windows required:
    - Native Sony HID for **DualSense, DualSense Edge, and DualShock 4** (USB & Bluetooth)
    - Compacted 4-slot **XInput** for Xbox One/Series & XInput controllers (USB & Bluetooth, even when reconnecting on slot 1–3)
    - Descriptor-driven **generic Windows HID** parser for Nintendo Switch Pro, 8BitDo, Logitech, Stadia, and third-party USB/Bluetooth gamepads
- **Audio (`audio.cpp`, `audio_wasapi.cpp`, `hle_audio.cpp`)**:
  - `cellAudio` 2/8-channel ports mixed every 5.33 ms at 48 kHz with 8-channel stereo downmix, port levels, notify queues, and `AddData`, outputting directly via Windows WASAPI.
- **OS / HLE Layer (`src/runtime`)**:
  - `sceLibStub` import resolution with SHA-1 NIDs (`nid_names.inc`), LV2 syscalls (`lv2.cpp` — host threads, mutexes, conds, semaphores, event queues/ports, timers, memory, TTY), `sysPrxForUser`, `cellSysmodule`, `cellSysutil`, VFS (`/dev_bdvd`, `/app_home`, `/dev_hdd0`, `/dev_hdd1`, `/dev_usb000`), `cellSaveData`, `cellGame`, `cellRtc`, and universal `cellPad`.
  - Memory map: HLE data `0x2F000000`, heap `0x30000000` (~213 MB), mmapper `0x40000000+`, RSX local memory `0xC0000000`, stacks/TLS `0xD0000000`, SPU LS window `0xF0000000`.

### Debug & Profiling Environment Variables
- `PS3RECOMP_PROFILE=<file.txt>` — enable sampling CPU/GPU/syscall profiler (analyze with `python tools/profile_report.py <file.txt> <game.map> 20`)
- `PS3RECOMP_FPS_LOG=1` — log live 2-second FPS, draws/frame, and any GPU draw fallback reasons
- `PS3RECOMP_RSX_TRACE=1` — per-second RSX counters, first use of every method, initial draw vertices/programs
- `PS3RECOMP_SYSCALL_TRACE=1` — first 400 syscalls with arguments, caller (`LR`), and timestamp
- `PS3RECOMP_DUMP_FRAME=<file.bmp>[,<flip>]` — save a presented frame to disk
- `PS3RECOMP_NO_LAZY_READBACK=1` — disable page-guarded lazy GPU readback

---

## Mods & Mod SDK

Every recompiled function is dispatched through a slot table, so hooking a guest address replaces it for every caller.

Use the standalone **[CellForge Mod SDK](mod_sdk/README.md)** (`mod_sdk/`) to scaffold, build, and deploy native C++20 DLL mods in two commands:

```powershell
.\mod_sdk\tools\new_mod.ps1 -Name MySkateMod
.\mod_sdk\tools\build_mod.ps1 -ModDir .\mod_sdk\mods_src\MySkateMod -DeployDir "$env:LOCALAPPDATA\Games\Skate3-VK\mods"
```

The single-header SDK ([mod_sdk/include/cellforge_mod.h](mod_sdk/include/cellforge_mod.h)) provides `CELLFORGE_MOD_ENTRY`, RAII function hooking (`cellforge::Hook`), big-endian PS3 guest memory helpers, wildcard pattern scanning, and an `.ini` config reader. Drop compiled `.dll` files into the game's `mods/` folder to load them automatically on startup, and toggle or configure them from the in-game **Settings → Mods** menu (`F1`). Pass `--no-mods` to disable mods or `--no-hooks` to `ps3recomp` to emit direct calls.

### Included Mods (`mod_sdk/mods/`)
- **`skate3_online`** — Native Winsock2 UDP multiplayer mod with an in-game **`F6`** dark-mode Lobby & Chat panel, 60 Hz skater kinematics, and quaternion NLERP interpolation. Selecting `host` automatically launches `relay_server.py` **headless** (no terminal window, auto-closed via Windows Job Object on exit) on UDP `37464` so friends can join by IP + room code with just one forwarded UDP port. *Status: transport, room codes, RTT, lobby, chat, and session markers work end-to-end; rendering remote players as live 3D skaters in-world is not finished yet (the guest-memory addresses for the skater/ghost pool in `skate3_online.ini` are still blank).*
- **`skate3_trick_timer`** — Draggable, always-on-top **`F7`** Speedrun Split Timer & Line Tracker HUD (`F8` Split, `F9` Pause, `F10` Reset) with automatic Personal Best (`PB`) deltas.
- **`skate3_clip_studio`** — Instant lossless PNG screenshot (`F12`), trick-sequence burst mode (`F11`), and **`F5`** Clip Studio control panel saving directly to `%USERPROFILE%\Pictures\Skate3-VK\`.

---

## Current Limitations

- Guests are assumed to return via `blr` (no `setjmp`/`longjmp` or unusual LR manipulation).
- `VSCR[SAT]` is not updated by saturating vector ops; FP record forms do not update `CR1`; FPSCR exceptions and non-default rounding modes are not modelled.
- RSX: fragment shader branching/loops, vertex subroutines (`CAL`/`CLI`/`CLB`), vertex textures, points/lines on GPU, MRT, swizzled/tiled/16-bit render targets, and 3D/fp16 register aliasing are still in progress.
- Audio: streams other than 48 kHz are not resampled; surround output is downmixed to stereo.
- Raw SPUs, SPURS task ELFs inside tasksets, hardware audio/video decoders (`cellAdec`), async/stream file I/O (`cellFsAio*`, `cellFsStRead*`), keyboard/mouse input, runtime PRX loading, and PS3 network sockets (`sys_net` / `sceNp`) are stubbed or not yet implemented.
