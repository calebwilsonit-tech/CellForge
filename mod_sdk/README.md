# CellForge Mod SDK

Create native C++20 DLL mods for any PS3 game recompiled with **CellForge** (including **Skate3-VK**).

Because CellForge dispatches every recompiled PPU function through a live function table, mods can hook any guest function by its PS3 address, call the original implementation, read/write big-endian PS3 guest memory directly, scan for byte patterns, and load their own `.ini` configs — with zero binary patching required.

---

## Quick Start

### 1. Create a New Mod
```powershell
.\tools\new_mod.ps1 -Name MySkateMod
```

### 2. Build & Deploy to Your Game
```powershell
.\tools\build_mod.ps1 -ModDir .\mods_src\MySkateMod -DeployDir "$env:LOCALAPPDATA\Games\Skate3-VK\mods"
```

### 3. Launch the Game
Run `Skate3-VK.exe` normally. Every `.dll` in the `mods/` folder is loaded automatically on startup (pass `--no-mods` to disable mods, or `--mods <dir>` to use a different folder).

---

## Minimal Example

```cpp
#include "cellforge_mod.h"

namespace {

cellforge::Hook g_target;

void my_hook(ppu::Context& c) {
  g_target.call_original(c);       // run the original PS3 function
  c.f[1] *= 1.5;                   // boost float return value by 1.5x
}

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  if (g_target.install(0x0012ABCD, my_hook)) {
    cellforge::log("MySkateMod: hooked 0x0012ABCD!");
  }
  return 0;
}
```

---

## API Reference (`include/cellforge_mod.h`)

### Entry Point & Logging
- `CELLFORGE_MOD_ENTRY(api)` — exports `ps3recomp_mod_init(const Ps3RecompModApi* api)`. Return `0` on success.
- `cellforge::init(api)` — validates `PS3RECOMP_MOD_API_VERSION` and initializes the helpers.
- `cellforge::log(msg)` / `cellforge::logf(fmt, ...)` — logs to `[ps3] [mod] ...`.

### Function Hooking & Calling
- `cellforge::Hook` — RAII hook helper with `.install(guest_addr, fn)`, `.call_original(ctx)`, `.remove()`.
- `cellforge::hook(addr, fn)` / `cellforge::unhook(addr)` / `cellforge::call(ctx, addr)`.
- **PPU Calling Convention (`ppu::Context`)**:
  - `c.r[3] .. c.r[10]` — integer/pointer arguments (`c.r[3]` is also the integer return value)
  - `c.f[1] .. c.f[13]` — floating-point arguments (`c.f[1]` is also the float return value)
  - `c.v[2] .. c.v[13]` — AltiVec/VMX vector arguments (`c.v[2]` is vector return value)
  - `c.r[1]` — stack pointer, `c.r[2]` — TOC pointer, `c.lr` — link register

### Big-Endian PS3 Guest Memory
- `cellforge::guest_base()` — raw pointer to guest address `0x00000000` (4 GiB address space)
- `read_u8 / read_u16 / read_u32 / read_u64 / read_i32 / read_f32 / read_f64`
- `write_u8 / write_u16 / write_u32 / write_u64 / write_i32 / write_f32 / write_f64`
- `read_cstr(ea, max_len)`, `read_bytes(ea, dst, len)`, `write_bytes(ea, src, len)`

### Pattern Scanning & Config
- `cellforge::find_pattern(start_ea, length, "38 60 ?? ?? 4E 80 00 20")` — scan guest memory with `??` wildcards.
- `cellforge::Config` — lightweight `.ini` loader (`load`, `get_string`, `get_int`, `get_float`, `get_bool`).

---

## Finding PS3 Function Addresses

1. **Recompiler disassembly:** Run `ps3recomp EBOOT.ELF --disasm` or inspect the generated `fn_XXXXXXXX` functions in `gen/ppu_recomp_*.cpp`.
2. **Linker map:** Check `Skate3-VK.map` for every recompiled `fn_XXXXXXXX` symbol.
3. **Reverse engineering tools:** Open your decrypted `EBOOT.ELF` in Ghidra or IDA Pro (PowerPC 64-bit big-endian) — the virtual addresses match CellForge's guest addresses 1:1.

---

## Included Examples

- [examples/hello_mod/hello_mod.cpp](examples/hello_mod/hello_mod.cpp) — minimal mod with `.ini` config and logging
- [examples/memory_patch_mod/memory_patch_mod.cpp](examples/memory_patch_mod/memory_patch_mod.cpp) — configurable function hook & float return scaling

Build the examples with:
```powershell
.\tools\build_mod.ps1
```
