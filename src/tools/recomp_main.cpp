// ps3recomp: lifts PPU code from a plain (decrypted) PS3 ELF to C++.
//
//   ps3recomp <input.elf> (-o <out.cpp> | --out-dir <dir> [--split N])
//             [--func 0xADDR[:SIZE]]... [--func-file FILE] [--no-opd-scan]
//             [--no-hooks] [--disasm]
//
// Seeds: the entry point (an OPD on PS3), every OPD found in the image whose TOC
// matches the entry's, and any --func / --func-file addresses. FILE holds lines of
// "<name> <hex-address>[:size]" ('#' starts a comment).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "nid.h"
#include "ppu_analysis.h"
#include "ppu_lift.h"
#include "ps3_exec.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: ps3recomp <input.elf> (-o <out.cpp> | --out-dir <dir> [--split N])\n"
               "                 [--func 0xADDR[:SIZE]]... [--func-file FILE] [--no-opd-scan]\n"
               "                 [--no-hooks] [--disasm] [--imports]\n");
}

bool parse_seed(const std::string& s, ppu::FuncSeed& out) {
  char* end = nullptr;
  out.addr = std::strtoull(s.c_str(), &end, 0);
  if (end == s.c_str()) return false;
  out.size = (*end == ':') ? std::strtoull(end + 1, nullptr, 0) : 0;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, output, out_dir;
  std::vector<ppu::FuncSeed> seeds;
  bool disasm = false, opd_scan = true, list_imports = false;
  ppu::LiftOptions lopt;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (a == "-o") output = next();
    else if (a == "--out-dir") out_dir = next();
    else if (a == "--split") lopt.funcs_per_file = std::strtoull(next().c_str(), nullptr, 0);
    else if (a == "--no-hooks") lopt.hooks = false;
    else if (a == "--no-opd-scan") opd_scan = false;
    else if (a == "--disasm") disasm = true;
    else if (a == "--imports") list_imports = true;
    else if (a == "--func") {
      ppu::FuncSeed s;
      if (!parse_seed(next(), s)) { std::fprintf(stderr, "bad --func value\n"); return 2; }
      seeds.push_back(s);
    } else if (a == "--func-file") {
      std::ifstream f(next());
      if (!f) { std::fprintf(stderr, "cannot open func file\n"); return 2; }
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string name, addr;
        ppu::FuncSeed s;
        if (ls >> name >> addr && parse_seed(addr, s)) seeds.push_back(s);
      }
    } else if (!a.empty() && a[0] != '-' && input.empty()) {
      input = a;
    } else {
      usage();
      return 2;
    }
  }
  if (input.empty() || (output.empty() && out_dir.empty() && !disasm && !list_imports)) { usage(); return 2; }

  elf::LoadResult lr = elf::load_file(input);
  if (!lr.ok) { std::fprintf(stderr, "error: %s\n", lr.error.c_str()); return 1; }
  const elf::Image& img = lr.image;

  if (list_imports) {
    std::vector<elf::ImportModule> mods;
    std::string err;
    if (!elf::read_imports(img, mods, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const elf::ProcParam pp = elf::read_proc_param(img);
    if (pp.present)
      std::printf("process: sdk 0x%08X, primary prio %d, stack 0x%X, malloc page 0x%X\n", pp.sdk_version,
                  pp.primary_prio, pp.primary_stacksize, pp.malloc_pagesize);
    size_t total = 0, named = 0;
    for (const auto& m : mods) {
      std::printf("%s (%zu functions, %zu variables)\n", m.name.c_str(), m.funcs.size(), m.vars.size());
      for (const auto& f : m.funcs) {
        const std::string& n = nid::name_of(f.nid);
        std::printf("  0x%08X  %-40s slot 0x%08X\n", f.nid, n.empty() ? "?" : n.c_str(), f.slot);
        ++total;
        named += !n.empty();
      }
      for (const auto& v : m.vars) std::printf("  0x%08X  (variable) %s\n", v.nid, nid::name_of(v.nid).c_str());
    }
    std::printf("%zu modules, %zu functions (%zu named)\n", mods.size(), total, named);
    if (output.empty() && out_dir.empty() && !disasm) return 0;
  }

  ppu::AnalysisOptions aopt;
  uint64_t code = 0, toc = 0;
  if (img.read_opd(img.entry, code, toc) && img.is_executable(code)) {
    seeds.push_back({code, 0});
    aopt.toc = toc;
    std::printf("entry OPD 0x%llX -> code 0x%llX, TOC 0x%llX\n", (unsigned long long)img.entry,
                (unsigned long long)code, (unsigned long long)toc);
  } else if (img.is_executable(img.entry)) {
    seeds.push_back({img.entry, 0});
  }
  if (opd_scan && aopt.toc) {
    const auto opds = ppu::scan_opds(img, aopt.toc);
    std::printf("OPD scan: %zu function descriptor(s)\n", opds.size());
    seeds.insert(seeds.end(), opds.begin(), opds.end());
    std::set<uint64_t> tocs;
    for (const auto& s : opds)
      if (s.toc && s.toc != aopt.toc) tocs.insert(s.toc);
    aopt.alt_tocs.assign(tocs.begin(), tocs.end());
  }
  lopt.alt_tocs = aopt.alt_tocs;
  if (seeds.empty()) { std::fprintf(stderr, "error: no function seeds (use --func)\n"); return 1; }
  lopt.toc = aopt.toc;

  std::vector<std::string> warnings;
  const auto funcs = ppu::discover_functions(img, aopt, seeds, warnings);

  if (disasm) {
    for (const auto& [start, r] : funcs) {
      std::printf("%s:\n", ppu::func_name(start).c_str());
      for (uint64_t a = r.start; a < r.end; a += 4) {
        uint32_t raw = 0;
        img.read32(a, raw);
        std::printf("  %08llX  %08X  %s\n", (unsigned long long)a, raw, ppu::disassemble(ppu::decode(raw), a).c_str());
      }
    }
  }

  if (!output.empty() || !out_dir.empty()) {
    namespace fs = std::filesystem;
    fs::path dir, stem;
    if (!out_dir.empty()) {
      dir = out_dir;
      stem = "ppu_recomp";
      std::error_code ec;
      fs::create_directories(dir, ec);
    } else {
      const fs::path p(output);
      dir = p.parent_path();
      stem = p.stem();
      lopt.funcs_per_file = 0;  // -o always writes a single file
    }
    std::set<std::string> written;
    size_t unchanged = 0;
    for (const auto& f : ppu::lift_program(img, funcs, lopt, stem.string(), warnings)) {
      const fs::path path = dir / f.name;
      written.insert(f.name);
      // Leave identical files untouched so incremental builds only recompile what changed.
      {
        std::ifstream in(path, std::ios::binary);
        if (in) {
          std::string old((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
          if (old == f.content) { ++unchanged; continue; }
        }
      }
      std::ofstream out(path, std::ios::binary);
      if (!out) { std::fprintf(stderr, "error: cannot write %s\n", path.string().c_str()); return 1; }
      out << f.content;
    }
    // Remove generated files from an earlier run that this run did not produce
    // (fewer functions -> fewer split files); stale copies would define duplicates.
    if (!out_dir.empty()) {
      std::error_code ec;
      for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind(stem.string() + "_", 0) == 0 && e.path().extension() == ".cpp" && !written.count(n)) {
          fs::remove(e.path(), ec);
          std::printf("removed stale %s\n", n.c_str());
        }
      }
    }
    std::printf("%zu file(s) written, %zu unchanged\n", written.size() - unchanged, unchanged);
  }

  std::printf("%zu function(s), %zu warning(s)\n", funcs.size(), warnings.size());
  for (const auto& w : warnings) std::printf("  warning: %s\n", w.c_str());
  return 0;
}
