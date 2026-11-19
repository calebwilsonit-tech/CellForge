#pragma once
// Mod API. A mod is a DLL placed in the mods folder that exports:
//
//   extern "C" __declspec(dllexport) int ps3recomp_mod_init(const Ps3RecompModApi* api);
//
// Return 0 on success. Mods talk to the game only through the function pointers
// below (the DLL has its own copy of any inline runtime state, so it must not use
// ppu::hook_function etc. directly).
//
// Typical mod:
//
//   static ppu::Fn g_orig;
//   static void my_update(ppu::Context& c) {
//     g_orig(c);              // run the original guest function
//     c.r[3] += 1;            // ...then change its result
//   }
//   extern "C" __declspec(dllexport) int ps3recomp_mod_init(const Ps3RecompModApi* api) {
//     g_orig = api->hook(0x0012ABCD, my_update);
//     return g_orig ? 0 : 1;
//   }

#include <cstdint>

#include "ppu_runtime.h"

#define PS3RECOMP_MOD_API_VERSION 1

struct Ps3RecompModApi {
  uint32_t version;
  uint8_t* guest_base;                             // guest address 0
  ppu::Fn (*hook)(uint32_t addr, ppu::Fn fn);      // returns previous implementation or nullptr
  void (*unhook)(uint32_t addr);                   // restore the recompiled original
  void (*call)(ppu::Context& c, uint32_t addr);    // call any guest function by address
  void (*log)(const char* msg);
};

#if defined(_WIN32)
#define PS3RECOMP_MOD_EXPORT extern "C" __declspec(dllexport)
#else
#define PS3RECOMP_MOD_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace ppu {

// Loads every mod library in `dir` and calls its init function. Returns the number
// of mods that initialised successfully. Errors are reported through `log`.
int load_mods(const char* dir, uint8_t* guest_base, void (*log)(const char*));

}  // namespace ppu
