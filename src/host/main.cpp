// Host program for a recompiled game. Build it together with the generated C++
// (see PS3RECOMP_GAME_DIR in CMakeLists.txt) and run:
//
//   game.exe <path to decrypted EBOOT.ELF> [--mods <dir>] [--strict] [-- guest args...]

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "boot.h"
#include "options.h"
#include "settings_ui.h"

namespace {

std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  return s.substr(i);
}

std::string expand_env(const std::string& s) {
#if defined(_WIN32)
  if (s.find('%') == std::string::npos) return s;
  char buf[4096];
  const DWORD n = ExpandEnvironmentStringsA(s.c_str(), buf, sizeof(buf));
  if (n > 0 && n < sizeof(buf)) return std::string(buf);
#endif
  return s;
}

void try_load_ini(const std::filesystem::path& p, ppu::BootOptions& opt) {
  std::ifstream f(p);
  if (!f) return;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = trim(line.substr(0, eq));
    const std::string v = expand_env(trim(line.substr(eq + 1)));
    if (v.empty()) continue;
    if ((k == "elf" || k == "elf_path") && opt.elf_path.empty() && std::filesystem::exists(v)) opt.elf_path = v;
    else if ((k == "disc" || k == "game_root") && opt.game_root.empty() && std::filesystem::exists(v)) opt.game_root = v;
    else if ((k == "data" || k == "data_root" || k == "saves") && opt.data_root.empty()) opt.data_root = v;
    else if ((k == "title" || k == "window_title") && opt.window_title.empty()) opt.window_title = v;
    else if ((k == "icon" || k == "icon_path") && opt.icon_path.empty() && std::filesystem::exists(v)) opt.icon_path = v;
  }
}

void auto_discover(ppu::BootOptions& opt) {
  namespace fs = std::filesystem;
  fs::path exe_dir = fs::current_path();
#if defined(_WIN32)
  wchar_t buf[MAX_PATH] = {};
  if (GetModuleFileNameW(nullptr, buf, MAX_PATH)) exe_dir = fs::path(buf).parent_path();
#endif
  // Check local and parent ini files (Skate3-VK.ini, cellforge.ini, ps3game.ini, builder.ini).
  const fs::path inis[] = {
      exe_dir / "Skate3-VK.ini",
      exe_dir / "cellforge.ini",
      exe_dir / "ps3game.ini",
      exe_dir / "builder.ini",
      exe_dir.parent_path() / "ps3game.ini",
      exe_dir.parent_path().parent_path() / "ps3game.ini",
      exe_dir.parent_path().parent_path().parent_path() / "ps3game.ini",
      exe_dir.parent_path().parent_path().parent_path() / "build" / "Release" / "builder.ini",
      fs::current_path() / "Skate3-VK.ini",
      fs::current_path() / "ps3game.ini",
      fs::current_path() / "build" / "Release" / "builder.ini",
  };
  for (const auto& ini : inis) {
    std::error_code ec;
    if (fs::exists(ini, ec)) try_load_ini(ini, opt);
  }
  if (opt.elf_path.empty()) {
    const fs::path elfs[] = {
        exe_dir / "EBOOT.ELF",
        exe_dir / "EBOOT.elf",
        exe_dir / "PS3_GAME" / "USRDIR" / "EBOOT.ELF",
        exe_dir / "PS3_GAME" / "USRDIR" / "EBOOT.elf",
    };
    for (const auto& e : elfs) {
      std::error_code ec;
      if (fs::exists(e, ec)) {
        opt.elf_path = e.string();
        break;
      }
    }
  }
  if (!opt.elf_path.empty() && opt.data_root.empty()) {
    // Persistent saves folder outside build/update directories so updates never wipe saves.
    const std::string saved_games = expand_env("%USERPROFILE%\\Saved Games\\Skate3-VK");
    if (!saved_games.empty() && saved_games.find('%') == std::string::npos) {
      opt.data_root = saved_games;
    } else {
      opt.data_root = (exe_dir / "saves").string();
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Load user graphics/audio options from options.ini next to the executable.
  std::filesystem::path exe_dir = std::filesystem::current_path();
  {
    namespace fs = std::filesystem;
#if defined(_WIN32)
    wchar_t buf[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, buf, MAX_PATH)) exe_dir = fs::path(buf).parent_path();
#endif
    cfg::ini_path() = exe_dir / "options.ini";
    cfg::load(cfg::ini_path(), cfg::get());
    cfg::sync_atomics(cfg::get());
  }

  ppu::BootOptions opt;
  opt.mods_dir = (exe_dir / "mods").string();
  bool open_settings = false;
  bool want_console = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--settings") open_settings = true;
    else if (a == "--console") want_console = true;
    else if (a == "--mods" && i + 1 < argc) opt.mods_dir = argv[++i];
    else if (a == "--no-mods") opt.mods_dir.clear();
    else if (a == "--strict") opt.strict = true;
    else if (a == "--headless") { opt.headless = true; want_console = true; }
    else if (a == "--cpu-render") opt.gpu_render = false;
    else if (a == "--game-root" && i + 1 < argc) opt.game_root = argv[++i];
    else if (a == "--data" && i + 1 < argc) opt.data_root = argv[++i];
    else if (a == "--title" && i + 1 < argc) opt.window_title = argv[++i];
    else if (a == "--icon" && i + 1 < argc) opt.icon_path = argv[++i];
    else if (a == "--") { for (++i; i < argc; ++i) opt.args.push_back(argv[i]); }
    else if (opt.elf_path.empty()) opt.elf_path = a;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
#if defined(_WIN32)
  // Because the game uses /SUBSYSTEM:WINDOWS (no idle CMD window on launch),
  // attach to a parent console only when explicitly running --headless or --console
  // and stdout/stderr are not already redirected to a file/pipe.
  if (want_console && GetFileType(GetStdHandle(STD_ERROR_HANDLE)) == FILE_TYPE_UNKNOWN) {
    if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) {
      FILE* f = nullptr;
      freopen_s(&f, "CONOUT$", "w", stdout);
      freopen_s(&f, "CONOUT$", "w", stderr);
    }
  }
#endif
  if (open_settings) {
    cfg::show_settings_dialog(cfg::ini_path());
    return 0;
  }
  if (opt.elf_path.empty()) auto_discover(opt);
  if (opt.elf_path.empty()) {
    std::fprintf(stderr, "usage: %s <EBOOT.ELF> [--mods dir | --no-mods] [--strict] [-- args...]\n", argv[0]);
    return 2;
  }
  return ppu::boot(opt);
}
