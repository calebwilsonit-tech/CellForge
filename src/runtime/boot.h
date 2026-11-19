#pragma once
// Boots a recompiled PS3 executable: loads the (decrypted) ELF's data into guest
// memory, resolves its system-library imports, loads mods, and runs the primary
// PPU thread from the entry point. The recompiled code itself must already be
// linked into the program (ppu_register_functions()).

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ppu {

struct BootOptions {
  std::string elf_path;               // the game's decrypted EBOOT.ELF
  std::string mods_dir;               // "" = no mods
  // Folder containing PS3_GAME (mounted read-only as /dev_bdvd). "" = derive from
  // elf_path when it sits in PS3_GAME/USRDIR, else the ELF's own folder.
  std::string game_root;
  std::string data_root = "ps3_data"; // writable host folder for /dev_hdd0 etc.
  std::string window_title;           // "" = auto from PARAM.SFO (e.g. "Skate3-VK")
  std::string icon_path;              // "" = auto from PS3_GAME/ICON0.PNG
  std::vector<std::string> args;      // argv[1..]
  bool strict = false;                // throw on unimplemented syscalls/imports
  bool headless = false;              // no window: frames are produced but not shown
  bool gpu_render = true;             // draw 3D on the GPU (Vulkan); false = CPU reference renderer
  std::function<void(std::string_view)> tty;  // guest console output (default: stdout)
  std::function<void(std::string_view)> log;  // runtime log (default: stderr)
};

// Returns the guest process exit status (or a negative value on a load error).
int boot(const BootOptions& opt);

}  // namespace ppu
