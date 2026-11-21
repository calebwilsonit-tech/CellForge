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
#include <string>
#include <vector>

#include "ppu_runtime.h"

#define PS3RECOMP_MOD_API_VERSION 1

struct Ps3RecompModApi {
  uint32_t version;
  uint8_t* guest_base;                             // guest address 0
  ppu::Fn (*hook)(uint32_t addr, ppu::Fn fn);      // returns previous implementation or nullptr
  void (*unhook)(uint32_t addr);                   // restore the recompiled original
  void (*call)(ppu::Context& c, uint32_t addr);    // call any guest function by address
  void (*log)(const char* msg);
  // Appended after v1 (older mods never read it): code address of an imported HLE function, for hook().
  // `name` is a function name or "nid:0xXXXXXXXX". Returns 0 if the game doesn't import it.
  uint32_t (*hle_code)(const char* module, const char* name);
  // Permanent zero-initialised guest memory (never freed) for strings, structs and buffers handed to the game. 0 on failure.
  uint32_t (*guest_alloc)(uint32_t size, uint32_t align);
  // Guest function descriptor (OPD) whose code is the host function `fn`: a pointer the game can call, store in a
  // vtable or hand back as a callback. 0 on failure.
  uint32_t (*make_opd)(ppu::Fn fn);
};

#if defined(_WIN32)
#define PS3RECOMP_MOD_EXPORT extern "C" __declspec(dllexport)
#else
#define PS3RECOMP_MOD_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace ppu {

struct ModStatus {
  std::string filename;
  std::string path;
  bool loaded = false;
  bool disabled = false;
  std::string detail;
};

// Loads every enabled mod library in `dir` and calls its init function. Returns the number
// of mods that initialised successfully. Errors are reported through `log`.
int load_mods(const char* dir, uint8_t* guest_base, void (*log)(const char*));

// Returns the list of mods seen during the last load_mods() call (for the Settings Mod Menu).
std::vector<ModStatus> get_mod_status_list();

}  // namespace ppu
