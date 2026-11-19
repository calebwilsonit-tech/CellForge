#pragma once
// CellForge Builder & Bootstrapper pipeline:
// - Detects & installs missing prerequisites (CellForge source from GitHub,
//   Visual Studio 2022 C++ Build Tools + CMake, and ps3recomp.exe).
// - Recompiles the user's decrypted ELF, embeds ICON0.PNG as the executable icon,
//   and deploys a completely detached standalone game folder (e.g. Skate3-VK.exe +
//   Skate3-VK.ini + mods/) with persistent saves in %USERPROFILE%\Saved Games.
// Nothing here decrypts anything; an encrypted SELF (EBOOT.BIN) is rejected.

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>

namespace builder {

namespace fs = std::filesystem;

struct Settings {
  fs::path game_root;    // folder containing PS3_GAME
  fs::path elf;          // decrypted ELF (user-provided)
  fs::path work_dir;     // build scratch dir (generated C++, obj files); default %LOCALAPPDATA%\CellForge\work\<id>
  fs::path install_dir;  // detached output folder where Skate3-VK.exe + .ini + mods/ are saved
  int split = 500;       // functions per generated file
};

using Log = std::function<void(const std::string&)>;

struct Validation {
  bool ok = false;
  std::string error;
  std::string title, title_id;  // from PARAM.SFO when present
};

struct PrereqStatus {
  bool has_source = false;
  bool has_cmake = false;
  bool has_msvc = false;
  bool has_recomp = false;
  fs::path repo_dir;
  fs::path cmake_exe;
  fs::path recomp_exe;
  bool all_ready() const { return has_source && has_cmake && has_msvc; }
};

// Cheap checks only: folder layout, ELF magic (rejects encrypted SELF), PARAM.SFO.
Validation validate(const Settings& s);

// Finds the repository root (the folder with CMakeLists.txt and src/) from the
// running program's location. Empty if not found.
fs::path find_repo_root(const fs::path& start);
// Locates cmake.exe (PATH, Visual Studio's bundled copy, Program Files). Empty if none.
fs::path find_cmake();
// Checks whether Visual Studio 2022 C++ Build Tools (cl.exe) are installed.
bool has_msvc_build_tools();
// Checks all prerequisites (local repo or %LOCALAPPDATA%\CellForge\src, CMake, MSVC, ps3recomp).
PrereqStatus check_prereqs(const fs::path& start_dir);
// Downloads/installs any missing prerequisites (CellForge source from GitHub,
// VS 2022 C++ Build Tools + CMake, and builds ps3recomp.exe). Blocking.
bool ensure_prereqs(const fs::path& start_dir, const Log& log, std::atomic<bool>& cancel,
                    bool force_update_source = false);

// Sanitized executable/display name for a game (e.g. "Skate3-VK" for BLUS30464/BLES00760).
std::string game_name(const Validation& v);
// Default scratch work folder: <repo>/out/<TITLE_ID> if in repo, else %LOCALAPPDATA%\CellForge\work\<TITLE_ID>.
fs::path default_work_dir(const fs::path& repo, const Validation& v);
// Default detached install folder: %LOCALAPPDATA%\Games\<GameName>.
fs::path default_install_dir(const Validation& v);
// Persistent saves folder (under %USERPROFILE%\Saved Games) so updates never wipe saves.
fs::path saves_dir(const Validation& v);

// Recompiles, builds, and deploys the detached game to s.install_dir. Blocking.
bool build(const Settings& s, const fs::path& repo, const Log& log, std::atomic<bool>& cancel);

// Fast incremental runtime rebuild (~7s) + re-deploy to s.install_dir without
// recompiling generated PPU code or touching saves/.ini.
bool update_runtime(const Settings& s, const fs::path& repo, const Log& log, std::atomic<bool>& cancel);

// Path of the built game executable inside the work folder or detached install folder.
fs::path game_exe(const Settings& s);
fs::path installed_exe(const Settings& s, const Validation& v);

// Starts the game detached. Prefers the detached installed exe (zero args).
bool launch(const Settings& s, std::string& error);

// Persists the last-used paths next to the program.
void save_settings(const fs::path& file, const Settings& s);
Settings load_settings(const fs::path& file);

}  // namespace builder
