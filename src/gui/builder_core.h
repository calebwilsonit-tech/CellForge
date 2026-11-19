#pragma once
// The "builder" pipeline behind the GUI and the CLI: given the user's own game folder
// and a decrypted ELF they produced themselves, recompile it, build ps3game.exe and
// launch it. Nothing here decrypts anything; an encrypted SELF is rejected up front.

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>

namespace builder {

namespace fs = std::filesystem;

struct Settings {
  fs::path game_root;   // folder containing PS3_GAME
  fs::path elf;         // decrypted ELF (user-provided)
  fs::path work_dir;    // generated code, build output, saves; default <repo>/out/<title id>
  int split = 500;      // functions per generated file
};

using Log = std::function<void(const std::string&)>;

struct Validation {
  bool ok = false;
  std::string error;
  std::string title, title_id;  // from PARAM.SFO when present
};

// Cheap checks only: folder layout, ELF magic (rejects encrypted SELF), PARAM.SFO.
Validation validate(const Settings& s);

// Finds the repository root (the folder with CMakeLists.txt and src/) from the
// running program's location. Empty if not found.
fs::path find_repo_root(const fs::path& start);
// Locates cmake.exe (PATH, then Visual Studio's bundled copy). Empty if none.
fs::path find_cmake();
// Default work folder for a game: <repo>/out/<TITLE_ID or "game">.
fs::path default_work_dir(const fs::path& repo, const Validation& v);

// Recompiles and builds. Blocking; call from a worker thread. `cancel` is polled
// between steps and kills the running child process when set.
bool build(const Settings& s, const fs::path& repo, const Log& log, std::atomic<bool>& cancel);

// Fast incremental runtime rebuild (~7s): rebuilds ps3_spu, ps3_runtime, and re-links
// the game executable without recompiling the generated PPU code or touching saves.
bool update_runtime(const Settings& s, const fs::path& repo, const Log& log, std::atomic<bool>& cancel);

// Persistent saves folder (under %USERPROFILE%\Saved Games) so updates never wipe saves.
fs::path saves_dir(const Validation& v);

// Path of the built game executable inside the work folder (may not exist yet).
fs::path game_exe(const Settings& s);
// Starts the game detached. Returns false and sets `error` on failure.
bool launch(const Settings& s, std::string& error);

// Persists the last-used paths next to the program.
void save_settings(const fs::path& file, const Settings& s);
Settings load_settings(const fs::path& file);

}  // namespace builder
