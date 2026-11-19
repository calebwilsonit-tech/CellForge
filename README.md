# ps3-recomp

A static recompiler for the Cell PPU (PowerPC 64 + AltiVec, big-endian). It lifts
PPU code from a plain, already-decrypted ELF into C++ that a normal compiler turns
into a native x86-64 program. Nothing is interpreted or JIT-compiled at runtime.

No game files, keys or decryption code are included or supported: SELF/encrypted
input is rejected. The tests use a hand-assembled ELF generated at build time.

## Build and test

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

- `unit_tests` - every decoder table entry round-trips; VMX/FP/carry helper semantics.
- `recomp_roundtrip` - `make_test_elf` writes a tiny ELF -> `ps3recomp` lifts it -> the
  output is compiled into `ps3_test_run`, which runs it and checks the results, including
  jump tables, FP, VMX, atomics, OPD discovery, hooks and loading a mod DLL.

`PS3RECOMP_AVX2` (default ON) builds with AVX2/FMA.

## Usage

```
ps3recomp input.elf -o out.cpp
ps3recomp input.elf --out-dir gen --split 500     # many files of 500 functions each
          [--func 0xADDR[:SIZE]]... [--func-file FILE] [--no-opd-scan] [--no-hooks] [--disasm]
```

Link the generated files with `ps3_runtime` and call `ppu_register_functions()` at startup.

## What works

- **Instruction set**: integer (incl. carry/overflow forms, 64-bit rotates, multiply-high,
  divide, count-leading-zeros, byte-reversed and multiple loads/stores), condition-register
  logic, traps, time base, `lwarx`/`stwcx.` atomics (real compare-and-swap), cache ops,
  scalar FP (single + double, fused multiply-add, conversions, FPSCR moves) and the full
  AltiVec/VMX integer, float, permute, pack and compare set, plus the Cell `lvlx/lvrx/stvlx/stvrx`.
  The instruction list lives in `src/ppu/ppu_ops.inc` and `src/ppu/ppu_vmx_ops.inc`.
- **Function discovery**: entry OPD, OPD scan of the whole image (catches functions only
  called through pointers), `bl` targets, explicit `--func` seeds.
- **Jump tables**: `switch` tables (absolute and TOC-relative) are recovered and become a
  native C++ `switch` with `goto`s.
- **Output**: one file, or split into chunks for large games (`--split N`: at most N
  functions and about the average code size per file, so parallel builds stay balanced).
  Unchanged files are not rewritten and stale ones are removed, so rebuilds are incremental.
  Skate 3 (BLUS30464): 83,310 functions, 0 warnings, 211 files.
- **Hooks and mods**: see below.

## Real-program testing

`tools\run_homebrew.ps1 -Dir <folder of decrypted homebrew/test ELFs> [-Names a,b] [-Window]`
recompiles, builds and runs each program headless and prints its log. Verified so far with
PS3 homebrew test programs: PPU threads + file output (`ppu_thread`, `dump_stack`), and GCM
graphics rendering correctly on the GPU (`gs_gcm_basic_triangle`, `gs_gcm_cube` — textured,
with a text overlay — and `gs_gcm_hello_world`).

Debug switches (environment variables):
- `PS3RECOMP_RSX_TRACE=1` — per-second RSX counters, first use of every method, first draws' vertices/programs
- `PS3RECOMP_SYSCALL_TRACE=1` — first 400 syscalls with arguments, caller (LR) and time
- `PS3RECOMP_DUMP_FRAME=<file.bmp>[,<flip>]` — save one presented frame

## Builder app (GUI)

`build\Release\ps3recomp_builder.exe` — pick the game folder (the one containing `PS3_GAME`)
and the decrypted ELF you produced yourself, press **Build**, then **Play**. The log pane
streams the recompiler and compiler output; **Cancel** stops a build; the last-used paths are
saved in `builder.ini`. Generated code, the build and save data go to `out\<TITLE_ID>\`.
It never decrypts anything and rejects the encrypted `EBOOT.BIN` from the disc.

Console twin for scripts/tests:

```
build\Release\ps3build.exe --game "<game folder>" --elf "<your decrypted ELF>" [--play] [--validate-only]
```

## One command

```
.\tools\build_game.ps1 -Elf <your decrypted EBOOT.ELF> -GameRoot "<folder with PS3_GAME>" -Run
```

Recompiles, builds `ps3game.exe` and runs it. The script never decrypts: given the
shipped (encrypted) `EBOOT.BIN` it stops and says it needs a decrypted ELF that you
produce yourself.

## Running a recompiled program

```
ps3recomp EBOOT.ELF --imports                 # list system-library imports (module, NID, name)
ps3recomp EBOOT.ELF --out-dir gen --split 500
cmake -S . -B build-game -DPS3RECOMP_GAME_DIR=gen
cmake --build build-game --config Release --target ps3game
build-game/Release/ps3game.exe EBOOT.ELF [--mods dir | --no-mods] [--strict] [-- args]
```

`ppu::boot` (src/runtime/boot.cpp) loads the ELF's segments, reads the process and
PRX parameters, resolves imports, loads mods and runs the primary thread with the PS3
startup registers (argc/argv/envp, thread id, TLS info, malloc page size).

### OS layer (src/runtime)

- **Imports**: `sceLibStub` tables are parsed; each import slot gets an OPD whose code
  address is in the function table, so the game's own call stubs reach C++.
  NIDs are SHA-1 based (`src/elf/nid.cpp`); `nid_names.inc` names them in logs.
  Unimplemented imports log once (`module::name (NID)`) and return CELL_OK; `--strict` throws.
- **LV2 syscalls** (`lv2.cpp`): process, PPU threads (real host threads), mutex, cond,
  semaphore, event queue/port, timers, time, memory, TTY. Unknown syscalls log once.
- **HLE modules** (`hle_modules.cpp`): sysPrxForUser (TLS, threads, lwmutex/lwcond,
  malloc, string functions, printf family), cellSysmodule, cellSysutil basics.
- **File system** (`vfs.cpp`, `hle_fs.cpp`): `/dev_bdvd` and `/app_home` map read-only onto
  the game folder; `/dev_hdd0`, `/dev_hdd1`, `/dev_usb000` map onto a writable data folder
  (`--data`, default `ps3_data`). Paths cannot escape their mount. cellFs open/read/write/
  seek/stat/dir/mkdir/rename/unlink/truncate; cellGame boot check, content permit and
  PARAM.SFO parameters.
- **Controller** (`input.cpp`, `hle_pad.cpp`): cellPad on top of XInput (Xbox-style pads,
  A = cross, B = circle, X = square, Y = triangle). `input::set_provider` swaps the source.
- **Graphics, phase 1** (`rsx.cpp`, `hle_gcm.cpp`, `src/gfx`): cellGcm init, configuration,
  IO mapping with guest-visible offset tables, control registers, labels, reports, display
  buffers, flip/vblank handlers, and cellVideoOut. A puller thread executes the RSX FIFO
  (jumps, calls, returns, references, semaphores, colour clears, flips). Flips are shown in a
  Vulkan window (discrete GPU preferred; `--headless` disables it). `ps3_gfx_demo` checks the window.
- **Graphics, phase 2a — reference renderer** (`rsx_program.cpp`, `rsx_draw.cpp`): decoders for
  the RSX vertex- and fragment-program instruction formats (field layouts checked against
  published references), an interpreter for both, and a CPU rasterizer that draws straight
  into the render targets in guest memory: vertex arrays (float, half, short, byte, packed),
  u16/u32 index buffers, triangles/strips/fans/quads/quad strips/polygons, near-plane
  clipping, viewport/scissor, culling, a fill rule for shared edges, Z16/Z24S8 depth,
  blending, colour masks, depth clears, and textures (A8R8G8B8, R5G6B5, A1R5G5B5, A4R4G4B4,
  B8, DXT1/3/5; linear or swizzled; nearest/bilinear; wrap/mirror/clamp). Correct but slow:
  it is the reference the GPU backend is checked against (`--cpu-render` forces it).
- **Graphics, phase 2b — GPU renderer** (`vk_renderer.cpp`, `spirv_gen.cpp`): the CPU front end
  does vertex work; the GPU rasterises. Each RSX fragment program is translated to a SPIR-V
  fragment shader (generated directly, no external compiler). Render targets live on the GPU
  (headless Vulkan device, discrete GPU preferred) and are written back to guest memory at
  sync points: flips, semaphore releases, CPU-side clears, and before a rendered surface is
  read as a texture. Pipelines, shaders, samplers and textures are cached. Draws the GPU path
  can't handle fall back to the CPU renderer automatically. Tests render every scenario on
  both renderers and require identical images.
- **Audio** (`audio.cpp`, `audio_wasapi.cpp`, `hle_audio.cpp`): cellAudio ports (2/8 channels,
  8/16/32 blocks of 256 frames, big-endian float) mixed every 5.33 ms at 48 kHz, 8-channel
  downmix to stereo, port levels, read-index/tag/timestamp tracking, notify event queues, and
  AddData; output through WASAPI (Windows' built-in audio, no extra libraries). cellAudioOut
  reports a stereo LPCM device. `ps3_audio_demo` plays a test tone.
- **SPUs** (`src/spu`, `lv2_spu.cpp`): an SPU interpreter (full SPU ISA from the public
  instruction-set document) and LV2 SPU thread groups: `sys_spu_image_import`, group
  create/start/join/terminate, thread args, local-store read/write, MFC DMA (GET/PUT, lists,
  GETLLAR/PUTLLC/PUTLLUC lock-line atomics, SNDSIG), mailboxes, signal notification, the
  SPU-thread LS window at 0xF0000000, event ports (send/throw event) and bound queues
  (`sys_spu_thread_receive_event`). Each SPU thread is a host thread.
- **SPURS** (`hle_spurs.cpp`): the PPU API (initialise, workloads, ready count, signals,
  shutdown/remove, event-queue attach, tasksets) plus a C++ SPU kernel per SPURS SPU that
  schedules workloads like the firmware kernel and runs each workload's policy module in the
  SPU interpreter (kernel context at LS 0x100, module at 0xA00, kernel entry points at 0x290
  and 0x808). SPURS *tasks* (task ELFs inside a taskset) are not supported yet.
- **Save data** (`hle_savedata.cpp`): cellSaveData Auto/List/Fixed Load2/Save2 driving the game's
  list/fixed, stat and file callbacks; saves live in the data folder under
  `/dev_hdd0/home/00000001/savedata/<dir>/`.
- **Other services** (`hle_misc.cpp`, `hle_modules.cpp`): message dialogs (answered OK/YES via
  `cellSysutilCheckCallback`), system cache, cellRtc, sys_mmapper, `sys_ppu_thread_once`, and an
  offline console for cellNetCtl / sceNp / trophies. Game-shipped PRX files (encrypted SPRX) are
  not loaded; `sys_prx_load_module` returns a dummy module.
- Memory map: HLE data 0x2F000000, heap 0x30000000 (~213 MB), mmapper areas 0x40000000+,
  RSX local memory 0xC0000000, stacks/TLS 0xD0000000, SPU-thread LS window 0xF0000000.

## Mods

Every recompiled function has a slot, and guest-to-guest calls go through it. So
replacing a function replaces it for every caller:

```cpp
ppu::Fn original = ppu::hook_function(0x0012ABCD, my_function);  // in-process
ppu::unhook_function(0x0012ABCD);
```

Mod DLLs are loaded with `ppu::load_mods("mods", guest_base, log)`. A mod exports
`ps3recomp_mod_init(const Ps3RecompModApi*)` and uses the API's `hook`, `unhook`, `call`
and `guest_base`. See `src/runtime/ppu_mod_api.h` and `tests/example_mod.cpp`.

`--no-hooks` emits direct calls instead (slightly faster, not hookable).

## Not done yet

- Guests are assumed to return with `blr` (no setjmp/longjmp or LR tricks).
- VSCR[SAT] is not set by saturating vector ops; FP record forms don't update CR1; FPSCR
  exceptions and non-default rounding modes are not modelled.
- GPU path: vertex programs still run on the CPU (fine for vertex counts, a future
  optimisation), DDX/DDY return 0, stencil and MRT unsupported. Not yet in either
  renderer: shader branching/loops/condition codes,
  vertex textures, points/lines, stencil, MRT, swizzled/tiled/16-bit render targets,
  texture remap/mipmaps/cubemaps/3D textures, fp16 register aliasing. Audio: no resampling
  of game streams other than 48 kHz, no personal/secondary outputs, surround output is
  downmixed to stereo. Raw SPUs, SPURS tasks/job chains, SPU static recompilation (SPUs are
  interpreted), audio/video decoders (cellAdec), async/stream file I/O (cellFsAio*,
  cellFsStRead*), keyboard/mouse, network sockets (sys_net). Those imports are stubs today.
- Imported variables, event flags, rwlocks, timer objects, memory containers, loading PRX
  files at runtime. mmapper does not alias one memory object mapped at two addresses.
- Syscall numbers, error codes and struct layouts come from public reverse-engineering
  references; they are verified only against this project's tests, not real game traces.
- Function-end detection is heuristic; check `--disasm` output on real binaries.
