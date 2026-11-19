#include "ppu_analysis.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <set>

namespace ppu {
namespace {

constexpr int kWindow = 64;  // instructions to look back for a jump-table pattern
constexpr uint64_t kMaxTableEntries = 4096;

bool starts_with(const char* s, const char* p) { return std::strncmp(s, p, std::strlen(p)) == 0; }
bool ends_with(const char* s, const char* p) {
  const size_t n = std::strlen(s), m = std::strlen(p);
  return n >= m && std::strcmp(s + n - m, p) == 0;
}

bool is_call(const Insn& d) {
  switch (d.op) {
    case Op::B: case Op::Bc: case Op::Bclr: case Op::Bcctr: return d.lk();
    case Op::Sc: return true;
    default: return false;
  }
}

// Bitmask of GPRs written by an instruction.
uint32_t gpr_writes(const Insn& d) {
  // Undecodable words inside a code stream are almost always inline data (e.g. a
  // preceding switch table), which writes no registers.
  if (d.op == Op::Invalid) return 0;
  if (is_call(d)) return 0x1FF9u;       // r0, r3..r12 are volatile across calls
  if (d.op == Op::Lmw) return ~0u << d.rt();
  const OpInfo& info = op_info(d.op);
  if (info.vkind != VKind::None) return 0;
  uint32_t m = 0;
  if (starts_with(info.operands, "D,") || std::strcmp(info.operands, "D") == 0) m |= 1u << d.rt();
  if (starts_with(info.operands, "A,S")) m |= 1u << d.ra();
  const bool mem = info.mnemonic[0] == 'l' || starts_with(info.mnemonic, "st");
  if (mem && (ends_with(info.mnemonic, "u") || ends_with(info.mnemonic, "ux"))) m |= 1u << d.ra();
  return m;
}

struct Ctx {
  const elf::Image& img;
  const AnalysisOptions& opt;
  uint64_t lo;
};

bool fetch(const Ctx& x, uint64_t a, Insn& d) {
  uint32_t raw = 0;
  if (!x.img.read32(a, raw)) return false;
  d = decode(raw);
  return true;
}

// Finds the closest instruction before `at` (not before x.lo) that writes `reg`.
bool find_def(const Ctx& x, uint64_t at, unsigned reg, uint64_t& def_addr, Insn& def) {
  for (int n = 0; n < kWindow && at >= x.lo + 4; ++n) {
    at -= 4;
    Insn d;
    if (!fetch(x, at, d)) return false;
    if (gpr_writes(d) & (1u << reg)) { def_addr = at; def = d; return true; }
  }
  return false;
}

// Constant value of `reg` just before the instruction at `at`.
std::optional<uint64_t> resolve(const Ctx& x, uint64_t at, unsigned reg, int depth = 0) {
  if (reg == 0 || depth > 8) return std::nullopt;
  // r2 is the function's TOC throughout its body: the ABI only writes it to restore
  // it after calls (`ld r2,40(r1)`), which yields the same value.
  if (reg == 2 && x.opt.toc) return x.opt.toc;
  uint64_t a = 0;
  Insn d;
  if (!find_def(x, at, reg, a, d)) return std::nullopt;
  auto src = [&](unsigned r) -> std::optional<uint64_t> {
    if (r == 0) return uint64_t(0);
    return resolve(x, a, r, depth + 1);
  };
  switch (d.op) {
    case Op::Addi: {
      auto b = src(d.ra());
      return b ? std::optional<uint64_t>(*b + uint64_t(d.simm())) : std::nullopt;
    }
    case Op::Addis: {
      auto b = src(d.ra());
      return b ? std::optional<uint64_t>(*b + uint64_t(d.simm() * 65536)) : std::nullopt;
    }
    case Op::Ori: case Op::Oris: case Op::Xori: case Op::Xoris: {
      if (d.ra() != reg) return std::nullopt;
      auto b = resolve(x, a, d.rt(), depth + 1);
      if (!b) return std::nullopt;
      const uint64_t imm = (d.op == Op::Oris || d.op == Op::Xoris) ? d.uimm() << 16 : d.uimm();
      return (d.op == Op::Ori || d.op == Op::Oris) ? (*b | imm) : (*b ^ imm);
    }
    case Op::Or:
      if (d.rt() == d.rb() && d.ra() == reg) return resolve(x, a, d.rt(), depth + 1);
      return std::nullopt;
    case Op::Lwz: case Op::Ld: {
      auto b = src(d.ra());
      if (!b) return std::nullopt;
      const uint64_t ea = (*b + uint64_t(d.op == Op::Ld ? d.ds() : d.simm())) & 0xFFFFFFFFull;
      if (d.op == Op::Ld) { uint64_t v; if (x.img.read64(ea, v)) return v; }
      else { uint32_t v; if (x.img.read32(ea, v)) return uint64_t(v); }
      return std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

bool is_table_load(Op op) { return op == Op::Lwzx || op == Op::Lwax || op == Op::Ldx; }

// Like find_def, but looks through value-preserving steps compilers put between a
// table load and its use (extsw/extsh/extsb, mr).
bool find_def_through_ext(const Ctx& x, uint64_t at, unsigned reg, uint64_t& def_addr, Insn& def) {
  for (int hops = 0; hops < 4; ++hops) {
    if (!find_def(x, at, reg, def_addr, def)) return false;
    const bool ext = (def.op == Op::Extsw || def.op == Op::Extsh || def.op == Op::Extsb) && def.ra() == reg;
    const bool mov = def.op == Op::Or && def.rt() == def.rb() && def.ra() == reg;
    if (!ext && !mov) return true;
    reg = def.rt();  // source register of the extend / move
    at = def_addr;
  }
  return false;
}

}  // namespace

namespace {
std::optional<JumpTable> recover_with_toc(const elf::Image& img, const AnalysisOptions& opt,
                                          uint64_t func_start, uint64_t bctr_addr);
}

std::optional<JumpTable> recover_jump_table(const elf::Image& img, const AnalysisOptions& opt,
                                            uint64_t func_start, uint64_t bctr_addr) {
  if (auto jt = recover_with_toc(img, opt, func_start, bctr_addr)) return jt;
  for (uint64_t t : opt.alt_tocs) {
    if (t == opt.toc) continue;
    AnalysisOptions o;
    o.toc = t;
    auto jt = recover_with_toc(img, o, func_start, bctr_addr);
    if (jt && jt->table == bctr_addr + 4) return jt;
  }
  return std::nullopt;
}

namespace {
std::optional<JumpTable> recover_with_toc(const elf::Image& img, const AnalysisOptions& opt,
                                          uint64_t func_start, uint64_t bctr_addr) {
  const uint64_t lo = std::max<uint64_t>(func_start, bctr_addr >= 4 * kWindow ? bctr_addr - 4 * kWindow : 0);
  Ctx x{img, opt, lo};

  // 1. mtctr rT
  uint64_t mtctr_addr = 0;
  unsigned t = 0;
  {
    uint64_t a = bctr_addr;
    bool found = false;
    while (a >= lo + 4) {
      a -= 4;
      Insn d;
      if (!fetch(x, a, d)) return std::nullopt;
      if (d.op == Op::Mtspr && d.spr() == 9) { mtctr_addr = a; t = d.rt(); found = true; break; }
      if (d.op == Op::B || d.op == Op::Bclr || d.op == Op::Bcctr) return std::nullopt;  // crossed a block edge
    }
    if (!found) return std::nullopt;
  }

  // 2. definition of rT: either the table load itself or add rT,rEntry,rBase
  uint64_t def_addr = 0;
  Insn def;
  if (!find_def_through_ext(x, mtctr_addr, t, def_addr, def)) return std::nullopt;

  bool relative = false;
  uint64_t load_addr = 0, add_addr = 0;
  Insn load;
  unsigned base_reg = 0;
  if (is_table_load(def.op)) {
    load_addr = def_addr;
    load = def;
  } else if (def.op == Op::Add) {
    relative = true;
    add_addr = def_addr;
    const unsigned regs[2] = {def.ra(), def.rb()};
    bool ok = false;
    for (int k = 0; k < 2 && !ok; ++k) {
      uint64_t la = 0;
      Insn ld;
      if (find_def_through_ext(x, add_addr, regs[k], la, ld) && is_table_load(ld.op)) {
        load_addr = la; load = ld; base_reg = regs[1 - k]; ok = true;
      }
    }
    if (!ok) return std::nullopt;
  } else {
    return std::nullopt;
  }

  // 3. table address: whichever load operand is a known constant
  std::optional<uint64_t> table = resolve(x, load_addr, load.ra());
  if (!table) table = resolve(x, load_addr, load.rb());
  if (!table) return std::nullopt;
  uint64_t base = 0;
  if (relative) {
    auto b = resolve(x, add_addr, base_reg);
    if (!b) return std::nullopt;
    base = *b;
  }

  // 4. bound: nearest cmplwi before the load, guarded by a branch on its GT bit
  uint64_t count = 0;
  {
    uint64_t a = load_addr;
    while (a >= lo + 4 && !count) {
      a -= 4;
      Insn d;
      if (!fetch(x, a, d)) return std::nullopt;
      // An unconditional branch / return ends the straight-line path to the load:
      // a compare above it belongs to another block.
      if ((d.op == Op::B && !d.lk()) || ((d.op == Op::Bclr || d.op == Op::Bcctr) && d.bo() == 20)) break;
      if (d.op != Op::Cmpli) continue;
      const unsigned gt_bit = d.crfd() * 4 + 1;
      for (uint64_t b = a + 4; b < bctr_addr; b += 4) {
        Insn br;
        if (fetch(x, b, br) && br.op == Op::Bc && br.bi() == gt_bit) { count = d.uimm() + 1; break; }
      }
      if (!count) break;
    }
  }
  // No bounds check found nearby (it can live in another block). For a relative
  // table stored right after the bctr, the size is visible in the data itself:
  // entries are small offsets, and the table ends where instruction words begin.
  if (!count && relative && (*table & 0xFFFFFFFFull) == bctr_addr + 4) {
    const uint32_t esize = load.op == Op::Ldx ? 8 : 4;
    for (uint64_t i = 0; i < kMaxTableEntries; ++i) {
      uint32_t e = 0;
      if (!img.read32(*table + i * esize, e)) break;
      const int32_t off = int32_t(e);
      if (off < -0x100000 || off > 0x100000 || (off & 3)) break;
      const uint64_t target = (base + uint64_t(int64_t(off))) & 0xFFFFFFFFull;
      if (!img.is_executable(target)) break;
      count = i + 1;
    }
  }
  if (!count || count > kMaxTableEntries) return std::nullopt;

  // 5. read the entries
  JumpTable jt;
  jt.table = *table & 0xFFFFFFFFull;
  const unsigned esize = load.op == Op::Ldx ? 8 : 4;
  for (uint64_t i = 0; i < count; ++i) {
    const uint64_t ea = jt.table + i * esize;
    uint64_t target = 0;
    if (esize == 8) {
      if (!img.read64(ea, target)) return std::nullopt;
    } else {
      uint32_t e = 0;
      if (!img.read32(ea, e)) return std::nullopt;
      target = (relative || load.op == Op::Lwax) ? uint64_t(int64_t(int32_t(e))) : e;
    }
    if (relative) target += base;
    target &= 0xFFFFFFFFull;
    if ((target & 3) || !img.is_executable(target)) return std::nullopt;
    // Switch targets are code of the same function: near the bctr. A table read
    // through a wrong TOC yields scattered addresses and is rejected here.
    constexpr uint64_t kMaxSpan = 0x80000;
    if (target + kMaxSpan < bctr_addr || target > bctr_addr + kMaxSpan) return std::nullopt;
    jt.targets.push_back(target);
  }
  return jt;
}
}  // namespace

bool is_pointer_call(const elf::Image& img, uint64_t func_start, uint64_t bctr_addr) {
  AnalysisOptions opt;
  const uint64_t lo = std::max<uint64_t>(func_start, bctr_addr >= 64 ? bctr_addr - 64 : 0);
  Ctx x{img, opt, lo};
  for (uint64_t a = bctr_addr; a >= lo + 4;) {
    a -= 4;
    Insn d;
    if (!fetch(x, a, d)) return false;
    if (d.op == Op::Mtspr && d.spr() == 9) {
      uint64_t da = 0;
      Insn def;
      if (!find_def(x, a, d.rt(), da, def)) return false;
      return (def.op == Op::Lwz || def.op == Op::Ld) && def.ra() != 0;
    }
  }
  return false;
}

std::vector<FuncSeed> scan_opds(const elf::Image& img, uint64_t entry_toc) {
  std::vector<FuncSeed> out;
  if (!entry_toc) return out;
  // Pass 1: candidate {code, toc} pairs whose toc points into a data segment.
  struct Cand { uint32_t code, toc; };
  std::vector<Cand> cands;
  std::map<uint32_t, uint32_t> toc_count;
  for (const auto& s : img.segments) {
    for (uint64_t off = 0; off + 8 <= s.memsz; off += 4) {
      const uint8_t* p = &s.data[off];
      const uint32_t code = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
      const uint32_t t = (uint32_t(p[4]) << 24) | (uint32_t(p[5]) << 16) | (uint32_t(p[6]) << 8) | p[7];
      if ((code & 3) || !img.is_executable(code)) continue;
      const elf::Segment* ts = img.find(t);
      if (!ts || ts->executable()) continue;
      cands.push_back({code, t});
      ++toc_count[t];
    }
  }
  // Pass 2: real TOCs are shared by many descriptors; random data pairs are not.
  std::set<uint32_t> tocs = {uint32_t(entry_toc)};
  for (const auto& [t, n] : toc_count)
    if (n >= 256) tocs.insert(t);
  std::set<uint64_t> seen;
  for (const Cand& c : cands)
    if (tocs.count(c.toc) && seen.insert(c.code).second) out.push_back({c.code, 0, c.toc});
  return out;
}

std::map<uint64_t, FuncRange> discover_functions(const elf::Image& img, const AnalysisOptions& opt,
                                                 const std::vector<FuncSeed>& seeds,
                                                 std::vector<std::string>& warnings) {
  std::map<uint64_t, FuncRange> out;
  std::map<uint64_t, uint64_t> explicit_size, toc_of;
  std::set<uint64_t> known;
  std::deque<uint64_t> work;
  for (const auto& s : seeds) {
    work.push_back(s.addr);
    known.insert(s.addr);
    if (s.size) explicit_size[s.addr] = s.size;
    if (s.toc) toc_of.emplace(s.addr, s.toc);
  }

  char buf[160];
  auto explore = [&]() {
  while (!work.empty()) {
    const uint64_t start = work.front();
    work.pop_front();
    if (out.count(start)) continue;
    if ((start & 3) || !img.is_executable(start)) {
      std::snprintf(buf, sizeof buf, "%08llX: function start is not executable code, skipped", (unsigned long long)start);
      warnings.push_back(buf);
      continue;
    }
    // This function's TOC: from its descriptor, inherited from its caller, or the default.
    const auto tit = toc_of.find(start);
    const uint64_t ftoc = tit != toc_of.end() ? tit->second : opt.toc;
    AnalysisOptions fopt = opt;
    fopt.toc = ftoc;
    // Same TOC candidates as the lifter's jump-table recovery (jt_opts), so both
    // passes agree on which switches exist and how far each function reaches.
    if (opt.toc) fopt.alt_tocs.push_back(opt.toc);

    const auto es = explicit_size.find(start);
    uint64_t end = 0;
    uint64_t max_target = start;
    std::vector<uint64_t> jumps;  // b/bc targets without link
    for (uint64_t addr = start;; addr += 4) {
      uint32_t raw = 0;
      if (!img.read32(addr, raw) || !img.is_executable(addr)) { end = addr; break; }
      const Insn d = decode(raw);
      bool terminator = false;

      if (d.op == Op::B || d.op == Op::Bc) {
        const uint64_t t = branch_target(d, addr) & 0xFFFFFFFFull;
        if (d.lk()) {
          toc_of.emplace(t, ftoc);  // a direct call stays within the caller's TOC group
          if (known.insert(t).second) work.push_back(t);
        } else {
          jumps.push_back(t);
          if (t > addr && t - addr < 0x2000 && !known.count(t))
            max_target = std::max(max_target, t);  // forward jump inside this function
        }
        terminator = d.op == Op::B && !d.lk();
      } else if (d.op == Op::Bclr) {
        terminator = d.bo() == 20 && !d.lk();
      } else if (d.op == Op::Bcctr && d.bo() == 20 && !d.lk()) {
        terminator = true;
        if (auto jt = recover_jump_table(img, fopt, start, addr)) {
          for (uint64_t t : jt->targets) {
            if (t > addr) max_target = std::max(max_target, t);
            jumps.push_back(t);  // entries if they end up outside this function
          }
          // A table stored right after the bctr is data, not code: skip over it.
          if (jt->table == addr + 4) addr = jt->table + 4 * jt->targets.size() - 4;
        }
      }

      if (es != explicit_size.end()) {
        if (addr + 4 >= start + es->second) { end = addr + 4; break; }
        continue;
      }
      if (terminator && addr >= max_target) { end = addr + 4; break; }
    }
    out[start] = FuncRange{start, end, ftoc};
    // Jumps that leave this function are tail calls: their targets are function
    // entries even when they lie inside another function's range.
    for (uint64_t t : jumps) {
      if (t >= start && t < end) continue;
      if (!img.is_executable(t)) continue;
      toc_of.emplace(t, ftoc);
      if (known.insert(t).second) work.push_back(t);
    }
  }
  };
  explore();

  // Gap filling: code that no discovered function covers (reached only through
  // pointers we can't see, e.g. computed addresses) becomes a function too.
  // Zero padding and undecodable words between functions are skipped.
  // Plausible code: valid instructions up to the first unconditional branch/return.
  auto plausible = [&](uint64_t a) {
    for (int n = 0; n < 64; ++n, a += 4) {
      uint32_t raw = 0;
      if (!img.read32(a, raw) || !img.is_executable(a)) return false;
      const Insn d = decode(raw);
      if (d.op == Op::Invalid) return false;
      if ((d.op == Op::B && !d.lk()) || ((d.op == Op::Bclr || d.op == Op::Bcctr) && d.bo() == 20 && !d.lk()))
        return true;
    }
    return true;
  };
  if (opt.gap_fill && !out.empty()) {
    // Only between the first and last discovered function: the code segment also
    // holds the ELF header (before) and read-only data (after).
    const uint64_t lo_bound = out.begin()->second.start;
    uint64_t hi_bound = 0;
    for (const auto& [s, r] : out) hi_bound = std::max(hi_bound, r.end);
    for (int round = 0; round < 64; ++round) {
      std::vector<std::pair<uint64_t, uint64_t>> covered;
      for (const auto& [s, r] : out) covered.push_back({r.start, r.end});
      std::sort(covered.begin(), covered.end());
      size_t added = 0;
      for (const auto& seg : img.segments) {
        if (!seg.executable()) continue;
        size_t ci = 0;
        const uint64_t seg_lo = std::max(seg.vaddr, lo_bound);
        const uint64_t seg_hi = std::min(seg.vaddr + seg.memsz, hi_bound);
        for (uint64_t a = seg_lo; a + 4 <= seg_hi;) {
          while (ci < covered.size() && covered[ci].second <= a) ++ci;
          if (ci < covered.size() && covered[ci].first <= a) { a = covered[ci].second; continue; }
          uint32_t raw = 0;
          img.read32(a, raw);
          if (raw != 0 && !known.count(a) && plausible(a)) {
            known.insert(a);
            // Code is laid out by object file, so a gap shares the TOC of the code before it.
            if (ci > 0) {
              auto prev = out.find(covered[ci - 1].first);
              if (prev != out.end() && prev->second.toc) toc_of.emplace(a, prev->second.toc);
            }
            work.push_back(a);
            ++added;
            // Skip ahead to the next covered range; the new function is explored below.
            a = ci < covered.size() ? covered[ci].first : seg_hi;
            continue;
          }
          a += 4;
        }
      }
      if (!added) break;
      explore();
    }
  }
  return out;
}

}  // namespace ppu
