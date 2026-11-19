// Console twin of the ps3recomp Builder (same pipeline, scriptable and testable).
//
//   ps3build --game <folder with PS3_GAME> --elf <your decrypted ELF>
//            [--work <dir>] [--split N] [--validate-only] [--play]
//
// Exit codes: 0 ok, 1 validation/build failure, 2 bad arguments.
// This program does not decrypt anything; an encrypted EBOOT.BIN is rejected.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "builder_core.h"

namespace {
std::atomic<bool> g_cancel{false};
BOOL WINAPI on_ctrl(DWORD) {
  g_cancel = true;
  return TRUE;
}
void usage() {
  std::fprintf(stderr,
               "usage: ps3build --game <folder with PS3_GAME> --elf <decrypted ELF>\n"
               "                [--work <dir>] [--split N] [--validate-only] [--play]\n");
}
}  // namespace

int wmain(int argc, wchar_t** argv) {
  builder::Settings s;
  bool validate_only = false, play = false;
  for (int i = 1; i < argc; ++i) {
    const std::wstring a = argv[i];
    auto next = [&]() -> std::wstring { return i + 1 < argc ? argv[++i] : std::wstring(); };
    if (a == L"--game") s.game_root = next();
    else if (a == L"--elf") s.elf = next();
    else if (a == L"--work") s.work_dir = next();
    else if (a == L"--split") s.split = std::max(1, _wtoi(next().c_str()));
    else if (a == L"--validate-only") validate_only = true;
    else if (a == L"--play") play = true;
    else { usage(); return 2; }
  }
  if (s.game_root.empty() || s.elf.empty()) { usage(); return 2; }

  const builder::Validation v = builder::validate(s);
  if (!v.ok) {
    std::printf("error: %s\n", v.error.c_str());
    return 1;
  }
  std::printf("valid: %s%s\n", v.title.empty() ? "(no PARAM.SFO title)" : v.title.c_str(),
              v.title_id.empty() ? "" : (" [" + v.title_id + "]").c_str());
  if (validate_only) return 0;

  wchar_t exe[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(nullptr, exe, DWORD(std::size(exe)));
  const builder::fs::path repo = builder::find_repo_root(builder::fs::path(std::wstring(exe, n)).parent_path());
  if (repo.empty()) {
    std::printf("error: the ps3recomp source folder was not found next to this program\n");
    return 1;
  }
  if (s.work_dir.empty()) s.work_dir = builder::default_work_dir(repo, v);

  SetConsoleCtrlHandler(on_ctrl, TRUE);
  const bool ok = builder::build(
      s, repo, [](const std::string& line) { std::printf("%s\n", line.c_str()); std::fflush(stdout); }, g_cancel);
  if (!ok) return 1;
  if (play) {
    std::string err;
    if (!builder::launch(s, err)) {
      std::printf("error: %s\n", err.c_str());
      return 1;
    }
    std::printf("started the game\n");
  }
  return 0;
}
