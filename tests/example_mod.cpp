// Example mod: makes the guest function shl2() return one more than it used to.
// Real mods look the same, with the address of a game function instead.

#include "ppu_mod_api.h"
#include "test_layout.h"

namespace {

ppu::Fn g_original = nullptr;

void shl2_plus_one(ppu::Context& c) {
  g_original(c);  // run the recompiled original
  c.r[3] += 1;    // then adjust its return value
}

}  // namespace

PS3RECOMP_MOD_EXPORT int ps3recomp_mod_init(const Ps3RecompModApi* api) {
  if (api->version != PS3RECOMP_MOD_API_VERSION) return 1;
  g_original = api->hook(test_layout::k_shl2, shl2_plus_one);
  if (!g_original) return 2;
  api->log("example_mod: hooked shl2");
  return 0;
}
