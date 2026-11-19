// Tests for the builder pipeline core (validation, settings, paths).

#include <cstdio>
#include <fstream>
#include <string>

#include "builder_core.h"

namespace {
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what) {
  ++g_checks;
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++g_fail;
}
void write_bytes(const builder::fs::path& p, const std::string& bytes) {
  std::ofstream(p, std::ios::binary).write(bytes.data(), std::streamsize(bytes.size()));
}
}  // namespace

int main(int argc, char** argv) {
  namespace fs = builder::fs;
  const fs::path tmp = fs::temp_directory_path() / "ps3recomp_builder_test";
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp / "game" / "PS3_GAME" / "USRDIR");

  builder::Settings s;
  std::printf("validation\n");
  s.game_root = tmp / "nothing";
  s.elf = tmp / "x.elf";
  check(!builder::validate(s).ok, "missing PS3_GAME folder is rejected");

  s.game_root = tmp / "game";
  check(!builder::validate(s).ok, "missing ELF is rejected");

  write_bytes(tmp / "EBOOT.BIN", std::string("SCE\0\0\0\0\2", 8) + std::string(64, '\0'));
  s.elf = tmp / "EBOOT.BIN";
  const builder::Validation enc = builder::validate(s);
  check(!enc.ok && enc.error.find("does not decrypt") != std::string::npos,
        "encrypted SELF (SCE header) is rejected with a clear message");

  write_bytes(tmp / "notelf.bin", "hello world");
  s.elf = tmp / "notelf.bin";
  check(!builder::validate(s).ok, "non-ELF file is rejected");

  // A real ELF: the test ELF built by make_test_elf, if passed on the command line.
  if (argc > 1) {
    s.elf = argv[1];
    const builder::Validation v = builder::validate(s);
    check(v.ok, "plain ELF in a valid game folder is accepted");
    check(builder::default_work_dir(tmp, v).filename() == "game", "work folder defaults to 'game' without a TITLE_ID");
  }

  std::printf("settings\n");
  builder::Settings w;
  w.game_root = tmp / "game with spaces";
  w.elf = tmp / "d\xc3\xa9" "crypted.elf";  // non-ASCII path survives the round trip
  w.work_dir = tmp / "work";
  builder::save_settings(tmp / "builder.ini", w);
  const builder::Settings r = builder::load_settings(tmp / "builder.ini");
  check(r.game_root == w.game_root && r.elf == w.elf && r.work_dir == w.work_dir, "settings round-trip through the ini");

  std::printf("environment\n");
  const fs::path cmake = builder::find_cmake();
  std::printf("  cmake: %s\n", cmake.empty() ? "(not found)" : cmake.string().c_str());
  check(!builder::find_repo_root(fs::path(argv[0]).parent_path()).empty(), "repository root found from the test binary");

  fs::remove_all(tmp, ec);
  if (g_fail) std::printf("%d of %d checks FAILED\n", g_fail, g_checks);
  else std::printf("all %d checks passed\n", g_checks);
  return g_fail ? 1 : 0;
}
