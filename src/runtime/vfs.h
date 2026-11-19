#pragma once
// Virtual file system: maps PS3 device paths (/dev_bdvd, /dev_hdd0, /app_home, ...)
// onto host directories. Guest paths can never escape their mount's host directory.

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace vfs {

namespace fs = std::filesystem;

struct Mount {
  fs::path host;
  bool read_only = false;
};

// Mount `device` (e.g. "/dev_bdvd") at a host directory. Replaces an existing mount.
void mount(const std::string& device, const fs::path& host, bool read_only);
void unmount_all();

struct Resolved {
  fs::path host;
  bool read_only = false;
};
// Resolves a guest path. Returns nullopt for unknown devices or paths that try to
// leave the mount (e.g. "/dev_hdd0/../../x").
std::optional<Resolved> resolve(std::string_view guest_path);

// Standard layout for a game:
//   game_root   folder containing PS3_GAME (disc dump) -> /dev_bdvd (read-only),
//               PS3_GAME/USRDIR -> /app_home (read-only)
//   data_root   writable host folder -> /dev_hdd0, /dev_hdd1, /dev_usb000
void mount_standard(const fs::path& game_root, const fs::path& data_root);

// ---- PARAM.SFO ----
struct Sfo {
  std::map<std::string, std::string> strings;
  std::map<std::string, uint32_t> ints;
  bool load(const fs::path& file);
};
const Sfo& game_sfo();          // PARAM.SFO of the mounted game (empty if none)
void set_game_sfo(const Sfo& s);

}  // namespace vfs
