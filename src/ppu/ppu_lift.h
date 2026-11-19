#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "elf64.h"
#include "ppu_analysis.h"
#include "ppu_decode.h"

namespace ppu {

struct LiftOptions {
  uint64_t toc = 0;
  std::vector<uint64_t> alt_tocs;  // the program's other TOCs (see AnalysisOptions)
  // Route guest-to-guest calls through per-function slots so mods can hook any
  // function. Off = direct calls (slightly faster, not hookable).
  bool hooks = true;
  // 0 = one output file; N = split into files of N functions plus a register file.
  size_t funcs_per_file = 0;
};

struct OutputFile {
  std::string name;
  std::string content;
};

std::string func_name(uint64_t addr);

// Emits C++ for all functions. `stem` names the output files (stem.cpp, or
// stem_000.cpp ... plus stem_register.cpp when splitting).
std::vector<OutputFile> lift_program(const elf::Image& img, const std::map<uint64_t, FuncRange>& funcs,
                                     const LiftOptions& opt, const std::string& stem,
                                     std::vector<std::string>& warnings);

}  // namespace ppu
