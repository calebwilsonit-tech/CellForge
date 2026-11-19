#include "builder_core.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

#include "vfs.h"

namespace builder {
namespace {

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

// Runs a program, streaming its output line by line. Returns the exit code, or -1.
int run(const fs::path& exe, const std::vector<std::wstring>& args, const Log& log, std::atomic<bool>& cancel) {
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
  const BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  CloseHandle(wr);
  if (!ok) { CloseHandle(rd); log("cannot start " + narrow(exe.wstring())); return -1; }

  std::string line;
  char chunk[4096];
  DWORD got = 0;
  // Poll so a cancel request can interrupt a long compile.
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
  const wchar_t* roots[] = {L"C:\\Program Files\\Microsoft Visual Studio\\2022", L"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022"};
  const wchar_t* editions[] = {L"BuildTools", L"Community", L"Professional", L"Enterprise"};
  std::error_code ec;
  for (auto r : roots)
    for (auto e : editions) {
      fs::path c = fs::path(r) / e / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe";
      if (fs::exists(c, ec)) return c;
    }
  return {};
}

fs::path default_work_dir(const fs::path& repo, const Validation& v) {
  std::string name = v.title_id.empty() ? "game" : v.title_id;
  for (char& c : name)
    if (!(std::isalnum(uint8_t(c)) || c == '_' || c == '-')) c = '_';
  return repo / "out" / name;
}

fs::path saves_dir(const Validation& v) {
  wchar_t prof[MAX_PATH] = {};
  const DWORD n = GetEnvironmentVariableW(L"USERPROFILE", prof, MAX_PATH);
  fs::path base = (n > 0 && n < MAX_PATH) ? (fs::path(prof) / "Saved Games") : exe_dir();
  if (v.title_id == "BLUS30464" || v.title_id == "BLES00760") return base / "Skate3-VK";
  std::string folder = v.title_id.empty() ? "CellForge" : ("CellForge\\" + v.title_id);
  return base / folder;
}

fs::path game_exe(const Settings& s) {
  std::error_code ec;
  const fs::path sk = s.work_dir / "build" / "Release" / "Skate3-VK.exe";
  if (fs::exists(sk, ec)) return sk;
  return s.work_dir / "build" / "Release" / "ps3game.exe";
}

bool update_runtime(const Settings& s, const fs::path& repo, const Log& log, std::atomic<bool>& cancel) {
  (void)repo;
  const fs::path bld = s.work_dir / "build";
  std::error_code ec;
  if (!fs::exists(bld / "CMakeCache.txt", ec)) {
    log("error: no existing build found in " + narrow(bld.wstring()) + " — run a full Build first.");
    return false;
  }
  const fs::path cmake = find_cmake();
  if (cmake.empty()) {
    log("error: CMake was not found.");
    return false;
  }
  log("== 1/3  Updating SPU JIT & Vulkan presenter (ps3_spu, ps3_gfx)");
  int rc = run(cmake, {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3_spu", L"ps3_gfx", L"--", L"/m", L"/v:minimal"},
               log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: ps3_spu build failed"); return false; }

  log("== 2/3  Updating CellForge runtime (ps3_runtime)");
  rc = run(cmake, {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3_runtime", L"--", L"/m",
                   L"/p:BuildProjectReferences=false", L"/v:minimal"},
           log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: ps3_runtime build failed"); return false; }

  log("== 3/3  Re-linking game executable (preserving generated code & saves)");
  rc = run(cmake, {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3game", L"--", L"/m",
                   L"/p:BuildProjectReferences=false", L"/v:minimal"},
           log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: game link failed"); return false; }

  log("Update complete: " + narrow(game_exe(s).wstring()));
  return true;
}

bool build(const Settings& s, const fs::path& repo, const Log& log, std::atomic<bool>& cancel) {
  const Validation v = validate(s);
  if (!v.ok) { log("error: " + v.error); return false; }
  if (!v.title.empty()) log("Game: " + v.title + (v.title_id.empty() ? "" : " [" + v.title_id + "]"));

  const fs::path cmake = find_cmake();
  if (cmake.empty()) {
    log("error: CMake was not found. Install the Visual Studio 2022 Build Tools with the "
        "'Desktop development with C++' workload (it includes CMake and the C++ compiler).");
    return false;
  }
  fs::path recomp = exe_dir() / "ps3recomp.exe";
  std::error_code ec;
  if (!fs::exists(recomp, ec)) recomp = repo / "build" / "Release" / "ps3recomp.exe";
  if (!fs::exists(recomp, ec)) { log("error: ps3recomp.exe was not found next to the builder."); return false; }

  const fs::path gen = s.work_dir / "gen", bld = s.work_dir / "build";
  fs::create_directories(gen, ec);
  for (const auto& e : fs::directory_iterator(gen, ec))
    if (e.path().extension() == ".cpp") fs::remove(e.path(), ec);

  log("== 1/3  Recompiling the game code");
  int rc = run(recomp, {s.elf.wstring(), L"--out-dir", gen.wstring(), L"--split", std::to_wstring(s.split)}, log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: recompilation failed"); return false; }

  log("== 2/3  Configuring the build");
  rc = run(cmake, {L"-S", repo.wstring(), L"-B", bld.wstring(), L"-G", L"Visual Studio 17 2022", L"-A", L"x64",
                   L"-DPS3RECOMP_GAME_DIR=" + gen.wstring()}, log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: CMake configuration failed (is the Visual Studio C++ toolchain installed?)"); return false; }

  log("== 3/3  Compiling (this can take a while for a full game)");
  rc = run(cmake, {L"--build", bld.wstring(), L"--config", L"Release", L"--target", L"ps3game", L"--", L"/m", L"/v:minimal"},
           log, cancel);
  if (cancel) { log("cancelled"); return false; }
  if (rc != 0) { log("error: compilation failed"); return false; }

  if (!fs::exists(game_exe(s), ec)) { log("error: the build finished but ps3game.exe is missing"); return false; }
  log("Done: " + narrow(game_exe(s).wstring()));
  return true;
}

bool launch(const Settings& s, std::string& error) {
  std::error_code ec;
  const fs::path exe = game_exe(s);
  if (!fs::exists(exe, ec)) { error = "The game has not been built yet."; return false; }
  const Validation v = validate(s);
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
  }
  return s;
}

}  // namespace builder
