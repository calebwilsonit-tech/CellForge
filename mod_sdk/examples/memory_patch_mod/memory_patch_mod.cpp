// Example CellForge mod demonstrating:
//   1. Reading settings from `mods/memory_patch_mod.ini`
//   2. Big-endian PS3 guest memory patching
//   3. Hooking a recompiled PPU function by guest address and calling the original

#include "cellforge_mod.h"

namespace {

cellforge::Hook g_target_hook;
float g_multiplier = 1.0f;

// Example PPU hook: runs the original guest function, then scales float return value `f1`.
void hooked_float_fn(ppu::Context& c) {
  g_target_hook.call_original(c);
  c.f[1] *= static_cast<double>(g_multiplier);
}

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  cellforge::Config cfg;
  cfg.load("mods/memory_patch_mod.ini");

  // Optional: hook a PPU function address specified in `mods/memory_patch_mod.ini`, e.g.:
  //   hook_addr=0x00123450
  //   multiplier=1.5
  const std::string addr_str = cfg.get_string("hook_addr", "");
  g_multiplier = cfg.get_float("multiplier", 1.0f);

  if (!addr_str.empty()) {
    const uint32_t addr = static_cast<uint32_t>(std::strtoul(addr_str.c_str(), nullptr, 16));
    if (g_target_hook.install(addr, hooked_float_fn)) {
      cellforge::logf("memory_patch_mod: hooked 0x%08X (multiplier=%.2f)", addr, g_multiplier);
    } else {
      cellforge::logf("memory_patch_mod: address 0x%08X is not a recompiled function entry", addr);
    }
  } else {
    cellforge::log("memory_patch_mod: loaded (set hook_addr=0x... in mods/memory_patch_mod.ini to test hooking)");
  }

  return 0;
}
