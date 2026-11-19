// Minimal CellForge example mod.
// Drop the compiled `hello_mod.dll` into your game's `mods/` folder
// (e.g. %LOCALAPPDATA%\Games\Skate3-VK\mods\).

#include "cellforge_mod.h"

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  cellforge::Config cfg;
  const bool has_cfg = cfg.load("mods/hello_mod.ini");
  const std::string greeting = cfg.get_string("greeting", "Hello from CellForge Mod SDK!");

  cellforge::logf("hello_mod: %s (config=%s, guest_base=%p)",
                  greeting.c_str(), has_cfg ? "loaded" : "default",
                  static_cast<void*>(cellforge::guest_base()));

  return 0;
}
