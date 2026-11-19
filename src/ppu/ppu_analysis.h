#pragma once
// Static analysis over PPU code: function discovery and switch jump-table recovery.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "elf64.h"
#include "ppu_decode.h"

namespace ppu {

struct FuncSeed {
  uint64_t addr = 0;
  uint64_t size = 0;  // 0 = find the end heuristically
  uint64_t toc = 0;   // r2 for this function if known (large programs have several TOCs)
};

struct FuncRange {
  uint64_t start = 0;
  uint64_t end = 0;  // exclusive
  uint64_t toc = 0;  // this function's TOC (0 = unknown, use the program default)
};

struct JumpTable {
  uint64_t table = 0;
  std::vector<uint64_t> targets;  // one per table entry (may repeat)
};

struct AnalysisOptions {
  uint64_t toc = 0;        // r2 value; 0 = unknown
  bool gap_fill = true;    // treat uncovered code as functions too
  // Other TOCs in the program. When a switch can't be resolved with `toc` (a
  // function whose TOC was guessed), these are tried, accepting only a table that
  // sits inline right after the bctr, which validates the guess.
  std::vector<uint64_t> alt_tocs;
};

// Tries to recover the switch table feeding the `bctr` at `bctr_addr`, looking no
// further back than `func_start`. Recognises:
//   [cmplwi crN,rI,max ; bgt crN,default] ... base = const | ld/lwz base,off(r2)
//   lwzx/lwax/ldx rE,rBase,rScaled ; [add rT,rE,rBase] ; mtctr rT ; bctr
std::optional<JumpTable> recover_jump_table(const elf::Image& img, const AnalysisOptions& opt,
                                            uint64_t func_start, uint64_t bctr_addr);

// True when the `bctr` at bctr_addr jumps through a pointer loaded from memory
// (import stubs and other calls through a function descriptor) rather than a
// switch table: `lwz/ld rX,..; ...; mtctr rX; bctr`.
bool is_pointer_call(const elf::Image& img, uint64_t func_start, uint64_t bctr_addr);

// Seeds from OPD-shaped pairs {code, toc} anywhere in the image. Accepts the entry
// TOC plus any TOC shared by many descriptors (large programs use several TOCs).
std::vector<FuncSeed> scan_opds(const elf::Image& img, uint64_t entry_toc);

// Finds function boundaries starting from the seeds and following call targets.
std::map<uint64_t, FuncRange> discover_functions(const elf::Image& img, const AnalysisOptions& opt,
                                                 const std::vector<FuncSeed>& seeds,
                                                 std::vector<std::string>& warnings);

}  // namespace ppu
