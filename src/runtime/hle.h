#pragma once
// HLE ("high-level emulation") of PS3 system libraries.
//
// Games import firmware library functions by module name + NID. At boot the loader
// gives every import slot a function descriptor (OPD) in guest memory whose code
// address is registered in the normal function table, so a guest call through an
// import stub lands in a C++ implementation here. Imports with no implementation
// get a logging stub that returns CELL_OK (or throws in strict mode).
//
// HLE functions are ordinary ppu::Fn: arguments in c.r[3..10] / c.f[1..], result in c.r[3].
// They can be hooked by mods like any recompiled function.

#include <cstdint>
#include <string>
#include <vector>

#include "ppu_runtime.h"
#include "ps3_exec.h"

namespace hle {

void register_function(const char* module, const char* name, ppu::Fn fn);

struct ResolveStats {
  uint32_t modules = 0, functions = 0, implemented = 0, stubbed = 0, variables = 0;
};

// Builds OPDs for every import and writes them into the import slots.
ResolveStats resolve_imports(const std::vector<elf::ImportModule>& imports);

// Registers the built-in modules (sysPrxForUser, cellSysmodule, cellSysutil, sys_fs,
// cellGame, sys_io/cellPad, ...).
void register_builtin_modules();
void register_fs_modules();   // hle_fs.cpp
void register_pad_modules();  // hle_pad.cpp
void register_gcm_modules();  // hle_gcm.cpp
void register_audio_modules();  // hle_audio.cpp
void register_misc_modules();   // hle_misc.cpp: syscache, netctl, np, trophy
void register_spurs_module();   // hle_spurs.cpp
void register_savedata_module(); // hle_savedata.cpp

// Queues a system event (status, param) for every callback registered with
// cellSysutilRegisterCallback; delivered from cellSysutilCheckCallback.
void queue_sysutil_event(uint64_t status, uint64_t param);

// Creates a guest function descriptor (OPD) whose code is the host function `fn`,
// so guest code (or the kernel) can call it like any guest function pointer.
uint32_t make_opd(ppu::Fn fn);

// The registered implementation of module::name (or, if the module name differs,
// any module exporting that NID). nullptr if none.
ppu::Fn lookup(const std::string& module, const std::string& name);

// Guest address of the HLE OPD for module/name, 0 if that import wasn't resolved.
uint32_t import_opd(const std::string& module, const std::string& name);

// Function-table code address of an imported function (what ppu::hook_function takes), found by module + name
// or, when `name` is "nid:0xXXXXXXXX", by NID. 0 if the game doesn't import it.
uint32_t import_code(const std::string& module, const std::string& name);

// Hook point used by the built-in modules for _sys_printf & co.
std::string format_guest(ppu::Context& c, uint32_t fmt_addr, int first_gpr, int first_fpr = 1);

}  // namespace hle
