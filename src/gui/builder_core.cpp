#include "builder_core.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

#include "vfs.h"

namespace builder {
namespace {

constexpr const wchar_t* kCellForgeZipUrl =
    L"https://github.com/calebwilsonit-tech/CellForge/archive/refs/heads/master.zip";
constexpr const wchar_t* kVsBuildToolsUrl =
    L"https://aka.ms/vs/17/release/vs_BuildTools.exe";

std::string narrow(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}
std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
  return w;
}

std::wstring quote(const std::wstring& a) {
  if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
  std::wstring q = L"\"";
  for (wchar_t c : a) q += (c == L'"') ? std::wstring(L"\\\"") : std::wstring(1, c);
  return q + L"\"";
}

fs::path env_dir(const wchar_t* name, const fs::path& fallback) {
  wchar_t buf[MAX_PATH * 2] = {};
  const DWORD n = GetEnvironmentVariableW(name, buf, DWORD(std::size(buf)));
  if (n > 0 && n < std::size(buf)) return fs::path(std::wstring(buf, n));
  return fallback;
}

// Runs a program, streaming its output line by line. Returns the exit code, or -1.
int run(const fs::path& exe, const std::vector<std::wstring>& args, const Log& log, std::atomic<bool>& cancel,
        bool hide_window = true) {
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 0)) return -1;
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

  std::wstring cmd = quote(exe.wstring());
  for (const auto& a : args) cmd += L" " + quote(a);
  STARTUPINFOW si{sizeof(si)};
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = si.hStdError = wr;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION pi{};
  std::vector<wchar_t> buf(cmd.begin(), cmd.end());
  buf.push_back(0);
  const DWORD flags = hide_window ? CREATE_NO_WINDOW : 0;
  const BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, flags, nullptr, nullptr, &si, &pi);
  CloseHandle(wr);
  if (!ok) { CloseHandle(rd); log("cannot start " + narrow(exe.wstring())); return -1; }

  std::string line;
  char chunk[4096];
  DWORD got = 0;
  for (;;) {
    if (cancel) { TerminateProcess(pi.hProcess, 1); break; }
    DWORD avail = 0;
    if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr)) break;
    if (avail == 0) {
      if (WaitForSingleObject(pi.hProcess, 50) == WAIT_OBJECT_0) {
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) break;
      }
      continue;
    }
    if (!ReadFile(rd, chunk, sizeof chunk, &got, nullptr) || got == 0) break;
    for (DWORD i = 0; i < got; ++i) {
      if (chunk[i] == '\n') { if (!line.empty() && line.back() == '\r') line.pop_back(); log(line); line.clear(); }
      else line += chunk[i];
    }
  }
  if (!line.empty()) log(line);
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  CloseHandle(rd);
  return cancel ? -1 : int(code);
}

fs::path exe_dir() {
  wchar_t p[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(nullptr, p, DWORD(std::size(p)));
  return fs::path(std::wstring(p, n)).parent_path();
}

fs::path cellforge_app_dir() {
  return env_dir(L"LOCALAPPDATA", exe_dir()) / "CellForge";
}

// Writes a valid Vista+ PNG-in-ICO file directly from a PNG (e.g. PS3_GAME/ICON0.PNG).
// Zero external dependencies; accepted by MSVC rc.exe and Windows Explorer.
bool write_ico_from_png(const fs::path& png_path, const fs::path& ico_path) {
  std::ifstream in(png_path, std::ios::binary);
  if (!in) return false;
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (data.size() < 24) return false;
  static constexpr uint8_t kPngSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  if (std::memcmp(data.data(), kPngSig, 8) != 0) return false;
  const uint32_t w = (uint32_t(data[16]) << 24) | (uint32_t(data[17]) << 16) | (uint32_t(data[18]) << 8) | data[19];
  const uint32_t h = (uint32_t(data[20]) << 24) | (uint32_t(data[21]) << 16) | (uint32_t(data[22]) << 8) | data[23];
  uint8_t hdr[22] = {};
  hdr[2] = 1;  // type = ICO
  hdr[4] = 1;  // count = 1
  hdr[6] = w >= 256 ? 0 : uint8_t(w);
  hdr[7] = h >= 256 ? 0 : uint8_t(h);
  hdr[10] = 1;  // color planes
  hdr[12] = 32; // bpp
  const uint32_t sz = uint32_t(data.size());
  const uint32_t off = 22;
  std::memcpy(hdr + 14, &sz, 4);
  std::memcpy(hdr + 18, &off, 4);
  std::error_code ec;
  fs::create_directories(ico_path.parent_path(), ec);
  std::ofstream out(ico_path, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
  return out.good();
}

bool deploy_standalone(const Settings& s, const Validation& v, const fs::path& built_exe, const Log& log) {
  std::error_code ec;
  const fs::path dest = s.install_dir.empty() ? default_install_dir(v) : s.install_dir;
  const std::string gname = game_name(v);
  fs::create_directories(dest, ec);
  fs::create_directories(dest / "mods", ec);
  const fs::path dst_exe = dest / (gname + ".exe");
  if (!fs::copy_file(built_exe, dst_exe, fs::copy_options::overwrite_existing, ec)) {
    log("error: failed to copy game executable to " + narrow(dst_exe.wstring()) + ": " + ec.message());
    return false;
  }

  const fs::path save_root = saves_dir(v);
  fs::create_directories(save_root, ec);

  // Write <GameName>.ini only if missing so user settings and saves are never overwritten.
  const fs::path ini = dest / (gname + ".ini");
  if (!fs::exists(ini, ec)) {
    std::ofstream f(ini);
    f << "# " << gname << " (built with CellForge)\n";
    f << "# Completely standalone — no CellForge or compiler needed to play.\n";
    f << "title=" << gname << "\n";
    f << "disc=" << narrow(s.game_root.wstring()) << "\n";
    f << "elf=" << narrow(s.elf.wstring()) << "\n";
    f << "data=%USERPROFILE%\\Saved Games\\" << gname << "\n";
    const fs::path icon0 = s.game_root / "PS3_GAME" / "ICON0.PNG";
    if (fs::exists(icon0, ec)) f << "icon=" << narrow(icon0.wstring()) << "\n";
    log("Created " + narrow(ini.wstring()));
  }
  log("Installed standalone game: " + narrow(dst_exe.wstring()));
  log("Persistent saves folder:   " + narrow(save_root.wstring()));
  return true;
}

}  // namespace

Validation validate(const Settings& s) {
  Validation v;
  std::error_code ec;
  if (s.game_root.empty() || !fs::is_directory(s.game_root / "PS3_GAME", ec)) {
    v.error = "The game folder must contain a PS3_GAME folder (the dumped disc contents).";
    return v;
  }
  if (!fs::is_directory(s.game_root / "PS3_GAME" / "USRDIR", ec)) {
    v.error = "PS3_GAME\\USRDIR was not found in the game folder.";
    return v;
  }
  if (s.elf.empty() || !fs::is_regular_file(s.elf, ec)) {
    v.error = "Choose the decrypted ELF of the game.";
    return v;
  }
  unsigned char magic[4] = {};
  {
    std::ifstream f(s.elf, std::ios::binary);
    f.read(reinterpret_cast<char*>(magic), 4);
    if (f.gcount() != 4) { v.error = "The selected file is empty or unreadable."; return v; }
  }
  if (std::memcmp(magic, "SCE\0", 4) == 0) {
    v.error =
        "That file is the signed/encrypted EBOOT.BIN (a SELF), as shipped on the disc.\n"
        "This tool only accepts a decrypted ELF that you produce yourself from your own copy. "
        "It does not decrypt anything.";
    return v;
  }
  if (std::memcmp(magic, "\x7f" "ELF", 4) != 0) {
    v.error = "The selected file is not an ELF.";
    return v;
  }
  vfs::Sfo sfo;
  if (sfo.load(s.game_root / "PS3_GAME" / "PARAM.SFO")) {
    auto t = sfo.strings.find("TITLE"), id = sfo.strings.find("TITLE_ID");
    if (t != sfo.strings.end()) v.title = t->second;
    if (id != sfo.strings.end()) v.title_id = id->second;
  }
  v.ok = true;
  return v;
}

fs::path find_repo_root(const fs::path& start) {
  std::error_code ec;
  fs::path p = fs::absolute(start, ec);
  for (int i = 0; i < 8 && !p.empty(); ++i, p = p.parent_path()) {
    if (fs::exists(p / "CMakeLists.txt", ec) && fs::is_directory(p / "src" / "runtime", ec)) return p;
    if (p == p.parent_path()) break;
  }
  return {};
}

fs::path find_cmake() {
  wchar_t found[MAX_PATH];
  if (SearchPathW(nullptr, L"cmake.exe", nullptr, MAX_PATH, found, nullptr)) return found;
  const wchar_t* roots[] = {L"C:\\Program Files\\Microsoft Visual Studio\\2022",
                            L"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022"};
  const wchar_t* editions[] = {L"BuildTools", L"Community", L"Professional", L"Enterprise"};
  std::error_code ec;
  for (auto r : roots)
    for (auto e : editions) {
      fs::path c = fs::path(r) / e / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe";
      if (fs::exists(c, ec)) return c;
    }
  const fs::path pf = L"C:\\Program Files\\CMake\\bin\\cmake.exe";
  if (fs::exists(pf, ec)) return pf;
  return {};
}

bool has_msvc_build_tools() {
  const wchar_t* roots[] = {L"C:\\Program Files\\Microsoft Visual Studio\\2022",
                            L"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022"};
  const wchar_t* editions[] = {L"BuildTools", L"Community", L"Professional", L"Enterprise"};
  std::error_code ec;
  for (auto r : roots)
    for (auto e : editions) {
      fs::path vc = fs::path(r) / e / "VC" / "Tools" / "MSVC";
      if (fs::is_directory(vc, ec) && !fs::is_empty(vc, ec)) return true;
    }
  return false;
}

PrereqStatus check_prereqs(const fs::path& start_dir) {
  PrereqStatus st;
  std::error_code ec;
  st.repo_dir = find_repo_root(start_dir);
  if (st.repo_dir.empty()) {
    const fs::path cached = cellforge_app_dir() / "src" / "CellForge-master";
    if (fs::exists(cached / "CMakeLists.txt", ec) && fs::is_directory(cached / "src" / "runtime", ec))
      st.repo_dir = cached;
  }
  st.has_source = !st.repo_dir.empty();
  st.cmake_exe = find_cmake();
  st.has_cmake = !st.cmake_exe.empty();
  st.has_msvc = has_msvc_build_tools();

  const fs::path candidates[] = {
      exe_dir() / "ps3recomp.exe",
      st.repo_dir.empty() ? fs::path{} : (st.repo_dir / "build" / "Release" / "ps3recomp.exe"),
      cellforge_app_dir() / "bin" / "ps3recomp.exe",
  };
  for (const auto& c : candidates) {
    if (!c.empty() && fs::exists(c, ec)) {
      st.recomp_exe = c;
      st.has_recomp = true;
      break;
    }
  }
  return st;
}

bool ensure_prereqs(const fs::path& start_dir, const Log& log, std::atomic<bool>& cancel, bool force_update_source) {
  std::error_code ec;
  PrereqStatus st = check_prereqs(start_dir);
  const fs::path app = cellforge_app_dir();
  fs::create_directories(app, ec);

  // 1. CellForge source/runtime from GitHub (if not running inside a local git checkout or updating)
  if (!st.has_source || (force_update_source && find_repo_root(start_dir).empty())) {
    log("== Downloading CellForge source & runtime from GitHub...");
    const fs::path zip = app / "CellForge-master.zip";
    const fs::path src_dir = app / "src";
    fs::create_directories(src_dir, ec);
    int rc = run(L"curl.exe", {L"-L", L"-f", L"-s", L"-S", L"-o", zip.wstring(), kCellForgeZipUrl}, log, cancel);
    if (cancel) { log("cancelled"); return false; }
    if (rc != 0 || !fs::exists(zip, ec)) {
      log("error: failed to download CellForge source from GitHub (check your internet connection).");
      return false;
    }
    log("== Extracting CellForge source...");
    fs::remove_all(src_dir / "CellForge-master", ec);
    rc = run(L"tar.exe", {L"-xf", zip.wstring(), L"-C", src_dir.wstring()}, log, cancel);
    fs::remove(zip, ec);
    if (cancel) { log("cancelled"); return false; }
    if (rc != 0 || !fs::exists(src_dir / "CellForge-master" / "CMakeLists.txt", ec)) {
      log("error: failed to extract CellForge archive.");
      return false;
    }
    st = check_prereqs(start_dir);
    log("CellForge source ready: " + narrow(st.repo_dir.wstring()));
  }

  // 2. Visual Studio 2022 C++ Build Tools + CMake
  if (!st.has_msvc || !st.has_cmake) {
    log("== Installing Visual Studio 2022 C++ Build Tools & CMake (this may take a few minutes)...");
    const fs::path vs_exe = env_dir(L"TEMP", app) / "vs_BuildTools.exe";
    int rc = run(L"curl.exe", {L"-L", L"-f", L"-s", L"-S", L"-o", vs_exe.wstring(), kVsBuildToolsUrl}, log, cancel);
    if (cancel) { log("cancelled"); return false; }
    if (rc != 0 || !fs::exists(vs_exe, ec)) {
      log("error: failed to download Visual Studio 2022 Build Tools installer.");
      return false;
    }
    log("== Running Visual Studio Build Tools installer (accept the UAC prompt if shown)...");
    rc = run(vs_exe,
             {L"--wait", L"--norestart", L"--passive",
              L"--add", L"Microsoft.VisualStudio.Workload.VCTools",
              L"--includeRecommended"},
             log, cancel, /*hide_window=*/false);
    if (cancel) { log("cancelled"); return false; }
    // VS installer returns 0 (success) or 3010 (success, reboot recommended)
    if (rc != 0 && rc != 3010) {
      log("error: Visual Studio Build Tools installation exited with code " + std::to_string(rc));
      return false;
    }
    st = check_prereqs(start_dir);
    if (!st.has_msvc || !st.has_cmake) {
      log("error: Build Tools installed, but CMake/MSVC were not detected yet. Try restarting CellForge.");
      return false;
    }
    log("C++ Build Tools & CMake are ready.");
  }

  // 3. Build ps3recomp.exe from source if not present (or if source was just updated)
  if (!st.has_recomp || force_update_source) {
    log("== Building CellForge recompiler (ps3recomp.exe)...");
    const fs::path tbld = app / "tools_build";
    int rc = run(st.cmake_exe,
                 {L"-S", st.repo_dir.wstring(), L"-B", tbld.wstring(), L"-G", L"Visual Studio 17 2022", L"-A", L"x64"},
                 log, cancel);
    if (cancel) { log("cancelled"); return false; }
    if (rc != 0) { log("error: CMake configuration for ps3recomp failed."); return false; }
    rc = run(st.cmake_exe,
             {L"--build", tbld.wstring(), L"--config", L"Release", L"--target", L"ps3recomp", L"--", L"/m", L"/v:minimal"},
             log, cancel);
    if (cancel) { log("cancelled"); return false; }
    const fs::path built = tbld / "Release" / "ps3recomp.exe";
    if (rc != 0 || !fs::exists(built, ec)) {
      log("error: failed to build ps3recomp.exe.");
      return false;
    }
    fs::create_directories(app / "bin", ec);
    fs::copy_file(built, app / "bin" / "ps3recomp.exe", fs::copy_options::overwrite_existing, ec);
    log("Recompiler ready: " + narrow((app / "bin" / "ps3recomp.exe").wstring()));
  }

  return true;
}

std::string game_name(const Validation& v) {
  if (v.title_id == "BLUS30464" || v.title_id == "BLES00760") return "Skate3-VK";
  std::string name = v.title.empty() ? (v.title_id.empty() ? "CellForgeGame" : v.title_id) : v.title;
  std::string clean;
  for (char c : name) {
    if (std::isalnum(uint8_t(c)) || c == '-' || c == '_') clean += c;
    else if (c == ' ' && !clean.empty() && clean.back() != '-') clean += '-';
  }
  while (!clean.empty() && (clean.back() == '-' || clean.back() == '_')) clean.pop_back();
  return clean.empty() ? "CellForgeGame" : clean;
}

fs::path default_work_dir(const fs::path& repo, const Validation& v) {
  std::string name = v.title_id.empty() ? "game" : v.title_id;
  for (char& c : name)
    if (!(std::isalnum(uint8_t(c)) || c == '_' || c == '-')) c = '_';
  std::error_code ec;
  if (!repo.empty() && fs::is_directory(repo / ".git", ec)) return repo / "out" / name;
  if (!repo.empty() && repo.string().find("CellForge-master") == std::string::npos) return repo / "out" / name;
  return cellforge_app_dir() / "work" / name;
}

fs::path default_install_dir(const Validation& v) {
  return env_dir(L"LOCALAPPDATA", exe_dir()) / "Games" / game_name(v);
}

fs::path saves_dir(const Validation& v) {
  const fs::path base = env_dir(L"USERPROFILE", exe_dir()) / "Saved Games";
  return base / game_name(v);
}

fs::path game_exe(const Settings& s) {
  std::error_code ec;
  const fs::path sk = s.work_dir / "build" / "Release" / "Skate3-VK.exe";
  if (fs::exists(sk, ec)) return sk;
  const fs::path def = s.work_dir / "build" / "Release" / "ps3game.exe";
  if (fs::exists(def, ec)) return def;
  // Also check any *.exe in build/Release
  const fs::path rel = s.work_dir / "build" / "Release";
  if (fs::is_directory(rel, ec)) {
    for (const auto& e : fs::directory_iterator(rel, ec))
      if (e.path().extension() == ".exe") return e.path();
  }
  return def;
}

fs::path installed_exe(const Settings& s, const Validation& v) {
  const fs::path dir = s.install_dir.empty() ? default_install_dir(v) : s.install_dir;
  return dir / (game_name(v) + ".exe");
}

bool build(const Settings& s, const fs::path& repo_hint, const Log& log, std::atomic<bool>& cancel) {
  const Validation v = validate(s);
  if (!v.ok) { log("error: " + v.error); return false; }
  const std::string gname = game_name(v);
  if (!v.title.empty()) log("Game: " + v.title + (v.title_id.empty() ? "" : " [" + v.title_id + "]") + " -> " + gname);

  const fs::path start = repo_hint.empty() ? exe_dir() : repo_hint;
  if (!ensure_prereqs(start, log, cancel, false)) return false;
  const PrereqStatus st = check_prereqs(start);
  if (!st.all_ready() || !st.has_recomp) {
    log("error: prerequisites are still incomplete.");
    return false;
  }

  std::error_code ec;
  const fs::path work = s.work_dir.empty() ? default_work_dir(st.repo_dir, v) : s.work_dir;
  const fs::path gen = work / "gen", bld = work / "build";
  fs::create_directories(gen, ec);
  for (const auto& e : fs::directory_iterator(gen, ec))
    if (e.path().extension() == ".cpp") fs::remove(e.path(), ec);

  log("== 1/4  Recompiling game code (PowerPC -> C++)");
  int rc = run(st.recomp_exe, {s.elf.wstring(), L"--out-dir", gen.wstring(), L"--split", std::to_wstring(s.split)}, log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: recompilation failed"); return false; }

  // Generate .ico from PS3_GAME/ICON0.PNG so the standalone .exe has the real game icon
  const fs::path ico = work / (gname + ".ico");
  const fs::path icon0 = s.game_root / "PS3_GAME" / "ICON0.PNG";
  const bool has_ico = fs::exists(icon0, ec) && write_ico_from_png(icon0, ico);

  log("== 2/4  Configuring native x64 build");
  std::vector<std::wstring> cfg = {
      L"-S", st.repo_dir.wstring(), L"-B", bld.wstring(),
      L"-G", L"Visual Studio 17 2022", L"-A", L"x64",
      L"-DPS3RECOMP_GAME_DIR=" + gen.wstring(),
      L"-DPS3RECOMP_GAME_NAME=" + widen(gname),
  };
  if (has_ico) cfg.push_back(L"-DPS3RECOMP_GAME_ICON=" + ico.wstring());
  rc = run(st.cmake_exe, cfg, log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: CMake configuration failed"); return false; }

  log("== 3/4  Compiling native game executable (this takes a few minutes on first build)");
  rc = run(st.cmake_exe, {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3game", L"--", L"/m", L"/v:minimal"},
           log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: compilation failed"); return false; }

  const fs::path built = bld / "Release" / (gname + ".exe");
  if (!fs::exists(built, ec)) { log("error: build finished but " + gname + ".exe is missing"); return false; }

  log("== 4/4  Deploying standalone detached game folder");
  return deploy_standalone(s, v, built, log);
}

bool update_runtime(const Settings& s, const fs::path& repo_hint, const Log& log, std::atomic<bool>& cancel) {
  const Validation v = validate(s);
  if (!v.ok) { log("error: " + v.error); return false; }
  const fs::path start = repo_hint.empty() ? exe_dir() : repo_hint;
  const PrereqStatus st = check_prereqs(start);
  if (!st.has_cmake || !st.has_msvc) {
    log("error: CMake or C++ Build Tools missing — run Install Prereqs first.");
    return false;
  }
  const fs::path work = s.work_dir.empty() ? default_work_dir(st.repo_dir, v) : s.work_dir;
  const fs::path bld = work / "build";
  std::error_code ec;
  if (!fs::exists(bld / "CMakeCache.txt", ec)) {
    log("error: no existing build found in " + narrow(bld.wstring()) + " — run Build & Install first.");
    return false;
  }

  log("== 1/4  Updating SPU JIT & Vulkan presenter (ps3_spu, ps3_gfx)");
  int rc = run(st.cmake_exe,
               {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3_spu", L"ps3_gfx", L"--", L"/m", L"/v:minimal"},
               log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: ps3_spu/ps3_gfx build failed"); return false; }

  log("== 2/4  Updating CellForge runtime (ps3_runtime)");
  rc = run(st.cmake_exe,
           {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3_runtime", L"--", L"/m",
            L"/p:BuildProjectReferences=false", L"/v:minimal"},
           log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: ps3_runtime build failed"); return false; }

  log("== 3/4  Re-linking game executable (preserving generated code & saves)");
  rc = run(st.cmake_exe,
           {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3game", L"--", L"/m",
            L"/p:BuildProjectReferences=false", L"/v:minimal"},
           log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: game link failed"); return false; }

  const fs::path built = game_exe(Settings{s.game_root, s.elf, work, s.install_dir, s.split});
  if (!fs::exists(built, ec)) { log("error: built executable not found"); return false; }

  log("== 4/4  Updating standalone game folder (saves & .ini untouched)");
  return deploy_standalone(s, v, built, log);
}

bool launch(const Settings& s, std::string& error) {
  std::error_code ec;
  const Validation v = validate(s);
  const fs::path inst = installed_exe(s, v);
  if (fs::exists(inst, ec)) {
    // Launch the completely detached standalone executable with zero arguments!
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    std::wstring cmd = quote(inst.wstring());
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    const std::wstring cwd = inst.parent_path().wstring();
    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, cwd.c_str(), &si, &pi)) {
      error = "Could not start " + narrow(inst.wstring()) + " (error " + std::to_string(GetLastError()) + ").";
      return false;
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return true;
  }

  const fs::path exe = game_exe(s);
  if (!fs::exists(exe, ec)) { error = "The game has not been built yet."; return false; }
  const fs::path save_root = saves_dir(v);
  fs::create_directories(save_root, ec);
  fs::create_directories(s.work_dir / "mods", ec);
  std::wstring cmd = quote(exe.wstring()) + L" " + quote(s.elf.wstring()) + L" --game-root " + quote(s.game_root.wstring()) +
                     L" --data " + quote(save_root.wstring()) + L" --mods " + quote((s.work_dir / "mods").wstring());
  STARTUPINFOW si{sizeof(si)};
  PROCESS_INFORMATION pi{};
  std::vector<wchar_t> buf(cmd.begin(), cmd.end());
  buf.push_back(0);
  const std::wstring cwd = s.work_dir.wstring();
  if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, cwd.c_str(), &si, &pi)) {
    error = "Could not start the game (error " + std::to_string(GetLastError()) + ").";
    return false;
  }
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return true;
}

void save_settings(const fs::path& file, const Settings& s) {
  std::ofstream f(file);
  f << "game_root=" << narrow(s.game_root.wstring()) << "\n";
  f << "elf=" << narrow(s.elf.wstring()) << "\n";
  f << "work_dir=" << narrow(s.work_dir.wstring()) << "\n";
  f << "install_dir=" << narrow(s.install_dir.wstring()) << "\n";
}

Settings load_settings(const fs::path& file) {
  Settings s;
  std::ifstream f(file);
  std::string line;
  while (std::getline(f, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
    if (k == "game_root") s.game_root = widen(v);
    else if (k == "elf") s.elf = widen(v);
    else if (k == "work_dir") s.work_dir = widen(v);
    else if (k == "install_dir") s.install_dir = widen(v);
  }
  return s;
}

}  // namespace builder
