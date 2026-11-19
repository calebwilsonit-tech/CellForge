#include "ppu_lift.h"

#include <cstdarg>
#include <cstdio>
#include <set>
#include <sstream>

namespace ppu {
namespace {

std::string F(const char* f, ...) {
  char buf[640];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

uint64_t mask64(unsigned mb, unsigned me) {
  const uint64_t a = ~0ull >> mb, b = ~0ull << (63 - me);
  return mb <= me ? (a & b) : (a | b);
}

std::string s64(int64_t v) { return F("(uint64_t)(%lldLL)", (long long)v); }
std::string hex64(uint64_t v) { return F("0x%llXull", (unsigned long long)v); }
std::string hex32(uint64_t v) { return F("0x%08Xu", unsigned(v)); }
std::string label(uint64_t a) { return F("L_%08llX", (unsigned long long)a); }
std::string slot_name(uint64_t a) { return F("s_%08llX", (unsigned long long)a); }

std::string R(unsigned n) { return F("c.r[%u]", n); }
std::string R0(unsigned n) { return n ? R(n) : std::string("0ull"); }

// ---- memory operations ------------------------------------------------------
enum class Mem : uint8_t { Load, Store, FLoad, FStore };
enum class EaForm : uint8_t { D, DS, X };
struct MemOp {
  Op op;
  Mem kind;
  const char* fn;
  const char* cast;  // load: wrap of the loaded value; store: cast of the stored value
  EaForm form;
  bool update;
};

constexpr const char* kSx16 = "(uint64_t)(int64_t)(int16_t)";
constexpr const char* kSx32 = "(uint64_t)(int64_t)(int32_t)";

const MemOp kMemOps[] = {
    {Op::Lbz, Mem::Load, "ld8", "", EaForm::D, false},     {Op::Lbzu, Mem::Load, "ld8", "", EaForm::D, true},
    {Op::Lbzx, Mem::Load, "ld8", "", EaForm::X, false},    {Op::Lbzux, Mem::Load, "ld8", "", EaForm::X, true},
    {Op::Lhz, Mem::Load, "ld16", "", EaForm::D, false},    {Op::Lhzu, Mem::Load, "ld16", "", EaForm::D, true},
    {Op::Lhzx, Mem::Load, "ld16", "", EaForm::X, false},   {Op::Lhzux, Mem::Load, "ld16", "", EaForm::X, true},
    {Op::Lha, Mem::Load, "ld16", kSx16, EaForm::D, false}, {Op::Lhau, Mem::Load, "ld16", kSx16, EaForm::D, true},
    {Op::Lhax, Mem::Load, "ld16", kSx16, EaForm::X, false},{Op::Lhaux, Mem::Load, "ld16", kSx16, EaForm::X, true},
    {Op::Lwz, Mem::Load, "ld32", "", EaForm::D, false},    {Op::Lwzu, Mem::Load, "ld32", "", EaForm::D, true},
    {Op::Lwzx, Mem::Load, "ld32", "", EaForm::X, false},   {Op::Lwzux, Mem::Load, "ld32", "", EaForm::X, true},
    {Op::Lwa, Mem::Load, "ld32", kSx32, EaForm::DS, false},{Op::Lwax, Mem::Load, "ld32", kSx32, EaForm::X, false},
    {Op::Lwaux, Mem::Load, "ld32", kSx32, EaForm::X, true},
    {Op::Ld, Mem::Load, "ld64", "", EaForm::DS, false},    {Op::Ldu, Mem::Load, "ld64", "", EaForm::DS, true},
    {Op::Ldx, Mem::Load, "ld64", "", EaForm::X, false},    {Op::Ldux, Mem::Load, "ld64", "", EaForm::X, true},
    {Op::Lhbrx, Mem::Load, "ld16r", "", EaForm::X, false}, {Op::Lwbrx, Mem::Load, "ld32r", "", EaForm::X, false},
    {Op::Ldbrx, Mem::Load, "ld64r", "", EaForm::X, false},
    {Op::Stb, Mem::Store, "st8", "(uint8_t)", EaForm::D, false},   {Op::Stbu, Mem::Store, "st8", "(uint8_t)", EaForm::D, true},
    {Op::Stbx, Mem::Store, "st8", "(uint8_t)", EaForm::X, false},  {Op::Stbux, Mem::Store, "st8", "(uint8_t)", EaForm::X, true},
    {Op::Sth, Mem::Store, "st16", "(uint16_t)", EaForm::D, false}, {Op::Sthu, Mem::Store, "st16", "(uint16_t)", EaForm::D, true},
    {Op::Sthx, Mem::Store, "st16", "(uint16_t)", EaForm::X, false},{Op::Sthux, Mem::Store, "st16", "(uint16_t)", EaForm::X, true},
    {Op::Stw, Mem::Store, "st32", "(uint32_t)", EaForm::D, false}, {Op::Stwu, Mem::Store, "st32", "(uint32_t)", EaForm::D, true},
    {Op::Stwx, Mem::Store, "st32", "(uint32_t)", EaForm::X, false},{Op::Stwux, Mem::Store, "st32", "(uint32_t)", EaForm::X, true},
    {Op::Std, Mem::Store, "st64", "", EaForm::DS, false},          {Op::Stdu, Mem::Store, "st64", "", EaForm::DS, true},
    {Op::Stdx, Mem::Store, "st64", "", EaForm::X, false},          {Op::Stdux, Mem::Store, "st64", "", EaForm::X, true},
    {Op::Sthbrx, Mem::Store, "st16r", "(uint16_t)", EaForm::X, false},
    {Op::Stwbrx, Mem::Store, "st32r", "(uint32_t)", EaForm::X, false},
    {Op::Stdbrx, Mem::Store, "st64r", "", EaForm::X, false},
    {Op::Lfs, Mem::FLoad, "ldf32", "", EaForm::D, false},   {Op::Lfsu, Mem::FLoad, "ldf32", "", EaForm::D, true},
    {Op::Lfsx, Mem::FLoad, "ldf32", "", EaForm::X, false},  {Op::Lfsux, Mem::FLoad, "ldf32", "", EaForm::X, true},
    {Op::Lfd, Mem::FLoad, "ldf64", "", EaForm::D, false},   {Op::Lfdu, Mem::FLoad, "ldf64", "", EaForm::D, true},
    {Op::Lfdx, Mem::FLoad, "ldf64", "", EaForm::X, false},  {Op::Lfdux, Mem::FLoad, "ldf64", "", EaForm::X, true},
    {Op::Stfs, Mem::FStore, "stf32", "", EaForm::D, false}, {Op::Stfsu, Mem::FStore, "stf32", "", EaForm::D, true},
    {Op::Stfsx, Mem::FStore, "stf32", "", EaForm::X, false},{Op::Stfsux, Mem::FStore, "stf32", "", EaForm::X, true},
    {Op::Stfd, Mem::FStore, "stf64", "", EaForm::D, false}, {Op::Stfdu, Mem::FStore, "stf64", "", EaForm::D, true},
    {Op::Stfdx, Mem::FStore, "stf64", "", EaForm::X, false},{Op::Stfdux, Mem::FStore, "stf64", "", EaForm::X, true},
};

const MemOp* find_mem(Op op) {
  for (const auto& m : kMemOps)
    if (m.op == op) return &m;
  return nullptr;
}

// Effective address (uint32_t expression) for non-update forms.
std::string ea_of(const Insn& d, EaForm form) {
  switch (form) {
    case EaForm::D: return F("(uint32_t)(%s + %s)", R0(d.ra()).c_str(), s64(d.simm()).c_str());
    case EaForm::DS: return F("(uint32_t)(%s + %s)", R0(d.ra()).c_str(), s64(d.ds()).c_str());
    case EaForm::X: return F("(uint32_t)(%s + %s)", R0(d.ra()).c_str(), R(d.rb()).c_str());
  }
  return "0";
}
std::string ea_x(const Insn& d) { return ea_of(d, EaForm::X); }

std::string lift_mem(const MemOp& m, const Insn& d) {
  const unsigned rt = d.rt();
  std::string access;
  const char* ea = m.update ? "(uint32_t)ea" : nullptr;
  const std::string eas = m.update ? std::string(ea) : ea_of(d, m.form);
  switch (m.kind) {
    case Mem::Load: access = F("c.r[%u] = %sppu::%s(c.base, %s);", rt, m.cast, m.fn, eas.c_str()); break;
    case Mem::Store: access = F("ppu::%s(c.base, %s, %sc.r[%u]);", m.fn, eas.c_str(), m.cast, rt); break;
    case Mem::FLoad: access = F("c.f[%u] = ppu::%s(c.base, %s);", rt, m.fn, eas.c_str()); break;
    case Mem::FStore: access = F("ppu::%s(c.base, %s, c.f[%u]);", m.fn, eas.c_str(), rt); break;
  }
  if (!m.update) return access;
  const std::string off = m.form == EaForm::X ? R(d.rb()) : s64(m.form == EaForm::DS ? d.ds() : d.simm());
  return F("{ const uint64_t ea = c.r[%u] + %s; %s c.r[%u] = ea; }", d.ra(), off.c_str(), access.c_str(), d.ra());
}

// ---- per-function context ---------------------------------------------------
struct FnCtx {
  const elf::Image& img;
  const std::map<uint64_t, FuncRange>& funcs;
  const LiftOptions& opt;
  FuncRange self;
  std::vector<std::string>& warnings;
  std::set<uint64_t>& referenced;  // functions called directly (need a declaration)
};

AnalysisOptions jt_opts(const FnCtx& x) {
  AnalysisOptions o;
  o.toc = x.self.toc ? x.self.toc : x.opt.toc;
  o.alt_tocs = x.opt.alt_tocs;
  if (x.opt.toc) o.alt_tocs.push_back(x.opt.toc);
  return o;
}

void warn(FnCtx& x, uint64_t addr, const Insn& d, const char* why) {
  x.warnings.push_back(F("%08llX: %s (%s)", (unsigned long long)addr, why, disassemble(d, addr).c_str()));
}
std::string trap_stmt(uint64_t addr, const Insn& d) {
  return F("ppu::unimplemented(c, %s, %s);", hex32(addr).c_str(), hex32(d.raw).c_str());
}

std::string call_stmt(FnCtx& x, uint64_t target) {
  target &= 0xFFFFFFFFull;
  if (x.funcs.count(target)) {
    x.referenced.insert(target);
    return (x.opt.hooks ? slot_name(target) : func_name(target)) + "(c);";
  }
  return F("ppu::call_indirect(c, %s);", hex64(target).c_str());
}
bool in_self(const FnCtx& x, uint64_t t) { return t >= x.self.start && t < x.self.end; }

std::string cr_bit(unsigned bi) { return F("ppu::cr_bit(c, %u)", bi); }

// "Branch taken?" test for BO/BI. `pre` gets the CTR decrement, if any.
std::string branch_cond(const Insn& d, std::string& pre) {
  std::string cond;
  auto add = [&](const std::string& t) { cond += (cond.empty() ? "" : " && ") + t; };
  const unsigned bo = d.bo();
  if (!(bo & 0x04)) {
    pre = "--c.ctr; ";
    add((bo & 0x02) ? "c.ctr == 0" : "c.ctr != 0");
  }
  if (!(bo & 0x10)) add((bo & 0x08) ? cr_bit(d.bi()) : "!" + cr_bit(d.bi()));
  return cond.empty() ? "true" : cond;
}

// Arithmetic with optional XER[CA]/[OV] and CR0 updates. a/b are expressions
// already adjusted for subtract forms (a = ~rA).
std::string arith(const Insn& d, const std::string& a, const std::string& b, const char* expr, bool ov_ok) {
  std::string s = F("{ const uint64_t a = %s, b = %s; const uint64_t r = %s;", a.c_str(), b.c_str(), expr);
  if (d.oe() && ov_ok) s += " ppu::set_ov_add(c, a, b, r);";
  s += F(" c.r[%u] = r;", d.rt());
  if (d.rc()) s += " ppu::set_cr0(c, r);";
  return s + " }";
}

std::string rec(const Insn& d, unsigned dest) { return d.rc() ? F(" ppu::set_cr0(c, c.r[%u]);", dest) : ""; }

std::string lift_vmx(FnCtx& x, const Insn& d, uint64_t addr) {
  const OpInfo& info = op_info(d.op);
  const char* n = info.mnemonic;
  const unsigned t = d.rt(), a = d.ra(), b = d.rb(), cc = d.rc5();
  switch (info.vkind) {
    case VKind::DAB: return F("ppu::vmx::%s(c.v[%u], c.v[%u], c.v[%u]);", n, t, a, b);
    case VKind::DB: return F("ppu::vmx::%s(c.v[%u], c.v[%u]);", n, t, b);
    case VKind::DABC: return F("ppu::vmx::%s(c.v[%u], c.v[%u], c.v[%u], c.v[%u]);", n, t, a, b, cc);
    case VKind::DABS: return F("ppu::vmx::%s(c.v[%u], c.v[%u], c.v[%u], %u);", n, t, a, b, d.shb());
    case VKind::DBU: return F("ppu::vmx::%s(c.v[%u], c.v[%u], %u);", n, t, b, a);
    case VKind::DS: return F("ppu::vmx::%s(c.v[%u], %d);", n, t, d.simm5());
    case VKind::CMP: {
      std::string s = F("ppu::vmx::%s(c.v[%u], c.v[%u], c.v[%u]);", n, t, a, b);
      if (d.vrc())
        s += F(" c.cr[6] = ppu::vmx::%s(c.v[%u]);", d.op == Op::Vcmpbfp ? "cr6_bounds" : "cr6_from", t);
      return s;
    }
    case VKind::MFVSCR:
      return F("{ ppu::v128 z{}; ppu::vmx::set<uint32_t>(z, 3, c.vscr); c.v[%u] = z; }", t);
    case VKind::MTVSCR: return F("c.vscr = ppu::vmx::get<uint32_t>(c.v[%u], 3);", b);
    case VKind::None: break;
  }
  warn(x, addr, d, "unhandled vector op");
  return trap_stmt(addr, d);
}

std::string lift_insn(FnCtx& x, const Insn& d, uint64_t addr) {
  const unsigned rt = d.rt(), ra = d.ra(), rb = d.rb();
  const uint64_t next = addr + 4;

  if (const MemOp* m = find_mem(d.op)) return lift_mem(*m, d);
  if (op_info(d.op).vkind != VKind::None) return lift_vmx(x, d, addr);

  const bool fp_rc = d.rc() && (d.raw >> 26 == 59 || d.raw >> 26 == 63);
  if (fp_rc) warn(x, addr, d, "FP record form: CR1 is not updated");

  switch (d.op) {
    case Op::Invalid:
    case Op::Count:
      warn(x, addr, d, "undecoded instruction, emitting runtime trap");
      return trap_stmt(addr, d);

    // ---- immediates ----
    case Op::Addi: case Op::Addis: {
      const int64_t imm = d.op == Op::Addis ? d.simm() * 65536 : d.simm();
      return ra == 0 ? F("c.r[%u] = %s;", rt, s64(imm).c_str())
                     : F("c.r[%u] = c.r[%u] + %s;", rt, ra, s64(imm).c_str());
    }
    case Op::Mulli: return F("c.r[%u] = (uint64_t)((int64_t)c.r[%u] * %lldLL);", rt, ra, (long long)d.simm());
    case Op::Subfic: return F("c.r[%u] = ppu::add3(c, ~c.r[%u], %s, 1);", rt, ra, s64(d.simm()).c_str());
    case Op::Addic: return F("c.r[%u] = ppu::add3(c, c.r[%u], %s, 0);", rt, ra, s64(d.simm()).c_str());
    case Op::AddicRc:
      return F("c.r[%u] = ppu::add3(c, c.r[%u], %s, 0); ppu::set_cr0(c, c.r[%u]);", rt, ra, s64(d.simm()).c_str(), rt);
    case Op::Ori: return F("c.r[%u] = c.r[%u] | %s;", ra, rt, hex64(d.uimm()).c_str());
    case Op::Oris: return F("c.r[%u] = c.r[%u] | %s;", ra, rt, hex64(d.uimm() << 16).c_str());
    case Op::Xori: return F("c.r[%u] = c.r[%u] ^ %s;", ra, rt, hex64(d.uimm()).c_str());
    case Op::Xoris: return F("c.r[%u] = c.r[%u] ^ %s;", ra, rt, hex64(d.uimm() << 16).c_str());
    case Op::AndiRc:
      return F("c.r[%u] = c.r[%u] & %s; ppu::set_cr0(c, c.r[%u]);", ra, rt, hex64(d.uimm()).c_str(), ra);
    case Op::AndisRc:
      return F("c.r[%u] = c.r[%u] & %s; ppu::set_cr0(c, c.r[%u]);", ra, rt, hex64(d.uimm() << 16).c_str(), ra);

    // ---- compares ----
    case Op::Cmpi:
      return d.l() ? F("ppu::cmp_s(c, %u, (int64_t)c.r[%u], %lldLL);", d.crfd(), ra, (long long)d.simm())
                   : F("ppu::cmp_s(c, %u, (int32_t)c.r[%u], %lldLL);", d.crfd(), ra, (long long)d.simm());
    case Op::Cmpli:
      return d.l() ? F("ppu::cmp_u(c, %u, c.r[%u], %s);", d.crfd(), ra, hex64(d.uimm()).c_str())
                   : F("ppu::cmp_u(c, %u, (uint32_t)c.r[%u], %s);", d.crfd(), ra, hex64(d.uimm()).c_str());
    case Op::Cmp:
      return d.l() ? F("ppu::cmp_s(c, %u, (int64_t)c.r[%u], (int64_t)c.r[%u]);", d.crfd(), ra, rb)
                   : F("ppu::cmp_s(c, %u, (int32_t)c.r[%u], (int32_t)c.r[%u]);", d.crfd(), ra, rb);
    case Op::Cmpl:
      return d.l() ? F("ppu::cmp_u(c, %u, c.r[%u], c.r[%u]);", d.crfd(), ra, rb)
                   : F("ppu::cmp_u(c, %u, (uint32_t)c.r[%u], (uint32_t)c.r[%u]);", d.crfd(), ra, rb);

    // ---- rotates ----
    case Op::Rlwinm:
      return F("c.r[%u] = ppu::rotl32x(c.r[%u], %u) & %s;%s", ra, rt, rb,
               hex64(mask64(d.rc5() + 32, d.me5() + 32)).c_str(), rec(d, ra).c_str());
    case Op::Rlwnm:
      return F("c.r[%u] = ppu::rotl32x(c.r[%u], (unsigned)c.r[%u] & 31) & %s;%s", ra, rt, rb,
               hex64(mask64(d.rc5() + 32, d.me5() + 32)).c_str(), rec(d, ra).c_str());
    case Op::Rlwimi: {
      const uint64_t m = mask64(d.rc5() + 32, d.me5() + 32);
      return F("c.r[%u] = (ppu::rotl32x(c.r[%u], %u) & %s) | (c.r[%u] & %s);%s", ra, rt, rb, hex64(m).c_str(), ra,
               hex64(~m).c_str(), rec(d, ra).c_str());
    }
    case Op::Rldicl:
      return F("c.r[%u] = ppu::rotl64(c.r[%u], %u) & %s;%s", ra, rt, d.sh6(), hex64(mask64(d.mb6(), 63)).c_str(),
               rec(d, ra).c_str());
    case Op::Rldicr:
      return F("c.r[%u] = ppu::rotl64(c.r[%u], %u) & %s;%s", ra, rt, d.sh6(), hex64(mask64(0, d.mb6())).c_str(),
               rec(d, ra).c_str());
    case Op::Rldic:
      return F("c.r[%u] = ppu::rotl64(c.r[%u], %u) & %s;%s", ra, rt, d.sh6(),
               hex64(mask64(d.mb6(), 63 - d.sh6())).c_str(), rec(d, ra).c_str());
    case Op::Rldimi: {
      const uint64_t m = mask64(d.mb6(), 63 - d.sh6());
      return F("c.r[%u] = (ppu::rotl64(c.r[%u], %u) & %s) | (c.r[%u] & %s);%s", ra, rt, d.sh6(), hex64(m).c_str(), ra,
               hex64(~m).c_str(), rec(d, ra).c_str());
    }
    case Op::Rldcl:
      return F("c.r[%u] = ppu::rotl64(c.r[%u], (unsigned)c.r[%u] & 63) & %s;%s", ra, rt, rb,
               hex64(mask64(d.mb6(), 63)).c_str(), rec(d, ra).c_str());
    case Op::Rldcr:
      return F("c.r[%u] = ppu::rotl64(c.r[%u], (unsigned)c.r[%u] & 63) & %s;%s", ra, rt, rb,
               hex64(mask64(0, d.mb6())).c_str(), rec(d, ra).c_str());

    // ---- XO arithmetic ----
    case Op::Add: return arith(d, R(ra), R(rb), "a + b", true);
    case Op::Addc: return arith(d, R(ra), R(rb), "ppu::add3(c, a, b, 0)", true);
    case Op::Adde: return arith(d, R(ra), R(rb), "ppu::add3(c, a, b, c.xer_ca)", true);
    case Op::Addze: return arith(d, R(ra), "0ull", "ppu::add3(c, a, b, c.xer_ca)", true);
    case Op::Addme: return arith(d, R(ra), "~0ull", "ppu::add3(c, a, b, c.xer_ca)", true);
    case Op::Subf: return arith(d, "~" + R(ra), R(rb), "a + b + 1", true);
    case Op::Subfc: return arith(d, "~" + R(ra), R(rb), "ppu::add3(c, a, b, 1)", true);
    case Op::Subfe: return arith(d, "~" + R(ra), R(rb), "ppu::add3(c, a, b, c.xer_ca)", true);
    case Op::Subfze: return arith(d, "~" + R(ra), "0ull", "ppu::add3(c, a, b, c.xer_ca)", true);
    case Op::Subfme: return arith(d, "~" + R(ra), "~0ull", "ppu::add3(c, a, b, c.xer_ca)", true);
    case Op::Neg: return arith(d, "~" + R(ra), "0ull", "a + b + 1", true);
    case Op::Mullw:
      if (d.oe()) warn(x, addr, d, "mullwo: XER[OV] not modelled");
      return arith(d, R(ra), R(rb), "(uint64_t)((int64_t)(int32_t)a * (int64_t)(int32_t)b)", false);
    case Op::Mulld:
      if (d.oe()) warn(x, addr, d, "mulldo: XER[OV] not modelled");
      return arith(d, R(ra), R(rb), "a * b", false);
    case Op::Mulhw: return arith(d, R(ra), R(rb), "(uint64_t)(((int64_t)(int32_t)a * (int64_t)(int32_t)b) >> 32)", false);
    case Op::Mulhwu: return arith(d, R(ra), R(rb), "(uint64_t)(uint32_t)(((uint64_t)(uint32_t)a * (uint32_t)b) >> 32)", false);
    case Op::Mulhd: return arith(d, R(ra), R(rb), "ppu::mulh64((int64_t)a, (int64_t)b)", false);
    case Op::Mulhdu: return arith(d, R(ra), R(rb), "ppu::umulh64(a, b)", false);
    case Op::Divw:
      if (d.oe()) warn(x, addr, d, "divwo: XER[OV] not modelled");
      return arith(d, R(ra), R(rb), "ppu::divw((int32_t)a, (int32_t)b)", false);
    case Op::Divwu:
      return arith(d, R(ra), R(rb), "(uint32_t)b ? (uint64_t)((uint32_t)a / (uint32_t)b) : 0", false);
    case Op::Divd:
      if (d.oe()) warn(x, addr, d, "divdo: XER[OV] not modelled");
      return arith(d, R(ra), R(rb), "ppu::divd((int64_t)a, (int64_t)b)", false);
    case Op::Divdu: return arith(d, R(ra), R(rb), "b ? a / b : 0", false);

    // ---- logical ----
    case Op::And: return F("c.r[%u] = c.r[%u] & c.r[%u];%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Andc: return F("c.r[%u] = c.r[%u] & ~c.r[%u];%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Or: return F("c.r[%u] = c.r[%u] | c.r[%u];%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Orc: return F("c.r[%u] = c.r[%u] | ~c.r[%u];%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Xor: return F("c.r[%u] = c.r[%u] ^ c.r[%u];%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Nor: return F("c.r[%u] = ~(c.r[%u] | c.r[%u]);%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Nand: return F("c.r[%u] = ~(c.r[%u] & c.r[%u]);%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Eqv: return F("c.r[%u] = ~(c.r[%u] ^ c.r[%u]);%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Extsb: return F("c.r[%u] = (uint64_t)(int64_t)(int8_t)c.r[%u];%s", ra, rt, rec(d, ra).c_str());
    case Op::Extsh: return F("c.r[%u] = (uint64_t)(int64_t)(int16_t)c.r[%u];%s", ra, rt, rec(d, ra).c_str());
    case Op::Extsw: return F("c.r[%u] = (uint64_t)(int64_t)(int32_t)c.r[%u];%s", ra, rt, rec(d, ra).c_str());
    case Op::Cntlzw: return F("c.r[%u] = (uint64_t)std::countl_zero((uint32_t)c.r[%u]);%s", ra, rt, rec(d, ra).c_str());
    case Op::Cntlzd: return F("c.r[%u] = (uint64_t)std::countl_zero(c.r[%u]);%s", ra, rt, rec(d, ra).c_str());

    // ---- shifts ----
    case Op::Slw:
      return F("{ const unsigned n = (unsigned)c.r[%u] & 63; c.r[%u] = n < 32 ? (uint64_t)(uint32_t)((uint32_t)c.r[%u] << n) : 0; }%s",
               rb, ra, rt, rec(d, ra).c_str());
    case Op::Srw:
      return F("{ const unsigned n = (unsigned)c.r[%u] & 63; c.r[%u] = n < 32 ? (uint64_t)((uint32_t)c.r[%u] >> n) : 0; }%s",
               rb, ra, rt, rec(d, ra).c_str());
    case Op::Sld:
      return F("{ const unsigned n = (unsigned)c.r[%u] & 127; c.r[%u] = n < 64 ? c.r[%u] << n : 0; }%s", rb, ra, rt,
               rec(d, ra).c_str());
    case Op::Srd:
      return F("{ const unsigned n = (unsigned)c.r[%u] & 127; c.r[%u] = n < 64 ? c.r[%u] >> n : 0; }%s", rb, ra, rt,
               rec(d, ra).c_str());
    case Op::Sraw: return F("c.r[%u] = ppu::sraw(c, c.r[%u], (unsigned)c.r[%u] & 63);%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Srawi: return F("c.r[%u] = ppu::sraw(c, c.r[%u], %u);%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Srad: return F("c.r[%u] = ppu::srad(c, c.r[%u], (unsigned)c.r[%u] & 127);%s", ra, rt, rb, rec(d, ra).c_str());
    case Op::Sradi: return F("c.r[%u] = ppu::srad(c, c.r[%u], %u);%s", ra, rt, d.sh6(), rec(d, ra).c_str());

    // ---- load/store multiple ----
    case Op::Lmw:
      return F("{ uint32_t ea = %s; for (int i = %u; i < 32; ++i, ea += 4) c.r[i] = ppu::ld32(c.base, ea); }",
               ea_of(d, EaForm::D).c_str(), rt);
    case Op::Stmw:
      return F("{ uint32_t ea = %s; for (int i = %u; i < 32; ++i, ea += 4) ppu::st32(c.base, ea, (uint32_t)c.r[i]); }",
               ea_of(d, EaForm::D).c_str(), rt);
    case Op::Stfiwx: return F("ppu::st32(c.base, %s, ppu::fp_bits_lo(c.f[%u]));", ea_x(d).c_str(), rt);

    // ---- atomics, cache, sync ----
    case Op::Lwarx: return F("c.r[%u] = ppu::lwarx(c, %s);", rt, ea_x(d).c_str());
    case Op::Ldarx: return F("c.r[%u] = ppu::ldarx(c, %s);", rt, ea_x(d).c_str());
    case Op::StwcxRc: return F("ppu::stwcx(c, %s, (uint32_t)c.r[%u]);", ea_x(d).c_str(), rt);
    case Op::StdcxRc: return F("ppu::stdcx(c, %s, c.r[%u]);", ea_x(d).c_str(), rt);
    case Op::Sync: case Op::Eieio: return "std::atomic_thread_fence(std::memory_order_seq_cst);";
    case Op::Isync: case Op::Dcbst: case Op::Dcbf: case Op::Dcbt: case Op::Dcbtst: case Op::Icbi:
    case Op::Dst: case Op::Dstst: case Op::Dss:
      return ";";
    case Op::Dcbz: return F("ppu::dcbz(c.base, %s);", ea_x(d).c_str());

    // ---- traps ----
    case Op::Tw:
      return F("if (ppu::trap_cond(%u, (int32_t)c.r[%u], (int32_t)c.r[%u])) ppu::trap(c, %s);", rt, ra, rb, hex32(addr).c_str());
    case Op::Td:
      return F("if (ppu::trap_cond(%u, (int64_t)c.r[%u], (int64_t)c.r[%u])) ppu::trap(c, %s);", rt, ra, rb, hex32(addr).c_str());
    case Op::Twi:
      return F("if (ppu::trap_cond(%u, (int32_t)c.r[%u], %lldLL)) ppu::trap(c, %s);", rt, ra, (long long)d.simm(), hex32(addr).c_str());
    case Op::Tdi:
      return F("if (ppu::trap_cond(%u, (int64_t)c.r[%u], %lldLL)) ppu::trap(c, %s);", rt, ra, (long long)d.simm(), hex32(addr).c_str());

    // ---- special registers / CR ----
    case Op::Mfspr:
      switch (d.spr()) {
        case 1: return F("c.r[%u] = ppu::get_xer(c);", rt);
        case 8: return F("c.r[%u] = c.lr;", rt);
        case 9: return F("c.r[%u] = c.ctr;", rt);
        case 256: return F("c.r[%u] = c.vrsave;", rt);
        case 268: return F("c.r[%u] = ppu::timebase();", rt);
        case 269: return F("c.r[%u] = ppu::timebase() >> 32;", rt);
        default: warn(x, addr, d, "unsupported SPR"); return trap_stmt(addr, d);
      }
    case Op::Mtspr:
      switch (d.spr()) {
        case 1: return F("ppu::set_xer(c, c.r[%u]);", rt);
        case 8: return F("c.lr = c.r[%u];", rt);
        case 9: return F("c.ctr = c.r[%u];", rt);
        case 256: return F("c.vrsave = (uint32_t)c.r[%u];", rt);
        default: warn(x, addr, d, "unsupported SPR"); return trap_stmt(addr, d);
      }
    case Op::Mftb:
      return d.spr() == 269 ? F("c.r[%u] = ppu::timebase() >> 32;", rt) : F("c.r[%u] = ppu::timebase();", rt);
    case Op::Mfcr: return F("c.r[%u] = ppu::get_cr(c);", rt);
    case Op::Mtcrf: return F("ppu::set_cr(c, (uint32_t)c.r[%u], 0x%02X);", rt, d.crm());
    case Op::Mcrf: return F("c.cr[%u] = c.cr[%u];", d.crfd(), d.crfs());
    case Op::Crand: return F("%s = %s & %s;", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());
    case Op::Cror: return F("%s = %s | %s;", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());
    case Op::Crxor: return F("%s = %s ^ %s;", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());
    case Op::Crnand: return F("%s = !(%s & %s);", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());
    case Op::Crnor: return F("%s = !(%s | %s);", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());
    case Op::Creqv: return F("%s = !(%s ^ %s);", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());
    case Op::Crandc: return F("%s = %s & !%s;", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());
    case Op::Crorc: return F("%s = %s | !%s;", cr_bit(rt).c_str(), cr_bit(ra).c_str(), cr_bit(rb).c_str());

    // ---- floating point ----
    case Op::Fadd: return F("c.f[%u] = c.f[%u] + c.f[%u];", rt, ra, rb);
    case Op::Fsub: return F("c.f[%u] = c.f[%u] - c.f[%u];", rt, ra, rb);
    case Op::Fmul: return F("c.f[%u] = c.f[%u] * c.f[%u];", rt, ra, d.rc5());
    case Op::Fdiv: return F("c.f[%u] = c.f[%u] / c.f[%u];", rt, ra, rb);
    case Op::Fadds: return F("c.f[%u] = ppu::round_single(c.f[%u] + c.f[%u]);", rt, ra, rb);
    case Op::Fsubs: return F("c.f[%u] = ppu::round_single(c.f[%u] - c.f[%u]);", rt, ra, rb);
    case Op::Fmuls: return F("c.f[%u] = ppu::round_single(c.f[%u] * c.f[%u]);", rt, ra, d.rc5());
    case Op::Fdivs: return F("c.f[%u] = ppu::round_single(c.f[%u] / c.f[%u]);", rt, ra, rb);
    case Op::Fsqrt: return F("c.f[%u] = std::sqrt(c.f[%u]);", rt, rb);
    case Op::Fsqrts: return F("c.f[%u] = ppu::round_single(std::sqrt(c.f[%u]));", rt, rb);
    case Op::Fres: return F("c.f[%u] = ppu::round_single(1.0 / c.f[%u]);", rt, rb);
    case Op::Frsqrte: return F("c.f[%u] = 1.0 / std::sqrt(c.f[%u]);", rt, rb);
    case Op::Fsel: return F("c.f[%u] = ppu::fsel(c.f[%u], c.f[%u], c.f[%u]);", rt, ra, d.rc5(), rb);
    case Op::Fmadd: return F("c.f[%u] = std::fma(c.f[%u], c.f[%u], c.f[%u]);", rt, ra, d.rc5(), rb);
    case Op::Fmsub: return F("c.f[%u] = std::fma(c.f[%u], c.f[%u], -c.f[%u]);", rt, ra, d.rc5(), rb);
    case Op::Fnmadd: return F("c.f[%u] = -std::fma(c.f[%u], c.f[%u], c.f[%u]);", rt, ra, d.rc5(), rb);
    case Op::Fnmsub: return F("c.f[%u] = -std::fma(c.f[%u], c.f[%u], -c.f[%u]);", rt, ra, d.rc5(), rb);
    case Op::Fmadds: return F("c.f[%u] = ppu::round_single(std::fma(c.f[%u], c.f[%u], c.f[%u]));", rt, ra, d.rc5(), rb);
    case Op::Fmsubs: return F("c.f[%u] = ppu::round_single(std::fma(c.f[%u], c.f[%u], -c.f[%u]));", rt, ra, d.rc5(), rb);
    case Op::Fnmadds: return F("c.f[%u] = ppu::round_single(-std::fma(c.f[%u], c.f[%u], c.f[%u]));", rt, ra, d.rc5(), rb);
    case Op::Fnmsubs: return F("c.f[%u] = ppu::round_single(-std::fma(c.f[%u], c.f[%u], -c.f[%u]));", rt, ra, d.rc5(), rb);
    case Op::Fcmpu: case Op::Fcmpo: return F("ppu::fcmp(c, %u, c.f[%u], c.f[%u]);", d.crfd(), ra, rb);
    case Op::Frsp: return F("c.f[%u] = ppu::round_single(c.f[%u]);", rt, rb);
    case Op::Fctiw: return F("c.f[%u] = ppu::fctiw(c.f[%u], false);", rt, rb);
    case Op::Fctiwz: return F("c.f[%u] = ppu::fctiw(c.f[%u], true);", rt, rb);
    case Op::Fctid: return F("c.f[%u] = ppu::fctid(c.f[%u], false);", rt, rb);
    case Op::Fctidz: return F("c.f[%u] = ppu::fctid(c.f[%u], true);", rt, rb);
    case Op::Fcfid: return F("c.f[%u] = ppu::fcfid(c.f[%u]);", rt, rb);
    case Op::Fneg: return F("c.f[%u] = -c.f[%u];", rt, rb);
    case Op::Fmr: return F("c.f[%u] = c.f[%u];", rt, rb);
    case Op::Fabs: return F("c.f[%u] = std::fabs(c.f[%u]);", rt, rb);
    case Op::Fnabs: return F("c.f[%u] = -std::fabs(c.f[%u]);", rt, rb);
    case Op::Mffs: return F("c.f[%u] = std::bit_cast<double>((uint64_t)c.fpscr);", rt);
    case Op::Mtfsf: {
      uint32_t m = 0;
      for (int i = 0; i < 8; ++i)
        if (d.fm() & (0x80u >> i)) m |= 0xFu << (4 * (7 - i));
      return F("c.fpscr = (c.fpscr & ~0x%08Xu) | (ppu::fp_bits_lo(c.f[%u]) & 0x%08Xu);", m, rb, m);
    }
    case Op::Mtfsb0: return F("c.fpscr &= ~(0x80000000u >> %u);", rt);
    case Op::Mtfsb1: return F("c.fpscr |= (0x80000000u >> %u);", rt);
    case Op::Mtfsfi: {
      const unsigned sh = 4 * (7 - d.crfd());
      return F("c.fpscr = (c.fpscr & ~(0xFu << %u)) | (%uu << %u);", sh, (d.raw >> 12) & 15, sh);
    }
    case Op::Mcrfs: {
      const unsigned sh = 4 * (7 - d.crfs());
      return F("{ const uint32_t n = (c.fpscr >> %u) & 15; c.cr[%u] = ppu::CRField{uint8_t(n >> 3), uint8_t((n >> 2) & 1), uint8_t((n >> 1) & 1), uint8_t(n & 1)}; }",
               sh, d.crfd());
    }

    // ---- vector memory ----
    case Op::Lvx: case Op::Lvxl: case Op::Lvebx: case Op::Lvehx: case Op::Lvewx:
      return F("ppu::vmx::load(c.v[%u], c.base, %s);", rt, ea_x(d).c_str());
    case Op::Stvx: case Op::Stvxl: return F("ppu::vmx::store(c.v[%u], c.base, %s);", rt, ea_x(d).c_str());
    case Op::Stvebx: return F("ppu::vmx::store_elem(c.v[%u], c.base, %s, 1);", rt, ea_x(d).c_str());
    case Op::Stvehx: return F("ppu::vmx::store_elem(c.v[%u], c.base, %s, 2);", rt, ea_x(d).c_str());
    case Op::Stvewx: return F("ppu::vmx::store_elem(c.v[%u], c.base, %s, 4);", rt, ea_x(d).c_str());
    case Op::Lvsl: return F("ppu::vmx::lvsl(c.v[%u], %s);", rt, ea_x(d).c_str());
    case Op::Lvsr: return F("ppu::vmx::lvsr(c.v[%u], %s);", rt, ea_x(d).c_str());
    case Op::Lvlx: case Op::Lvlxl: return F("ppu::vmx::lvlx(c.v[%u], c.base, %s);", rt, ea_x(d).c_str());
    case Op::Lvrx: case Op::Lvrxl: return F("ppu::vmx::lvrx(c.v[%u], c.base, %s);", rt, ea_x(d).c_str());
    case Op::Stvlx: case Op::Stvlxl: return F("ppu::vmx::stvlx(c.v[%u], c.base, %s);", rt, ea_x(d).c_str());
    case Op::Stvrx: case Op::Stvrxl: return F("ppu::vmx::stvrx(c.v[%u], c.base, %s);", rt, ea_x(d).c_str());

    // ---- branches ----
    case Op::B: {
      const uint64_t t = branch_target(d, addr) & 0xFFFFFFFFull;
      if (d.lk()) return F("c.lr = %s; %s", hex64(next).c_str(), call_stmt(x, t).c_str());
      if (in_self(x, t)) return "goto " + label(t) + ";";
      return F("{ %s return; }", call_stmt(x, t).c_str());  // tail call
    }
    case Op::Bc: {
      const uint64_t t = branch_target(d, addr) & 0xFFFFFFFFull;
      std::string pre;
      const std::string cond = branch_cond(d, pre);
      if (d.lk())
        return F("{ %sif (%s) { c.lr = %s; %s } }", pre.c_str(), cond.c_str(), hex64(next).c_str(), call_stmt(x, t).c_str());
      if (in_self(x, t)) return F("{ %sif (%s) goto %s; }", pre.c_str(), cond.c_str(), label(t).c_str());
      return F("{ %sif (%s) { %s return; } }", pre.c_str(), cond.c_str(), call_stmt(x, t).c_str());
    }
    case Op::Bclr: {
      std::string pre;
      const std::string cond = branch_cond(d, pre);
      if (d.lk())  // blrl: call through LR
        return F("{ %sif (%s) { const uint64_t t = c.lr; c.lr = %s; ppu::call_indirect(c, t); } }", pre.c_str(),
                 cond.c_str(), hex64(next).c_str());
      if (cond == "true" && pre.empty()) return "return;";
      return F("{ %sif (%s) return; }", pre.c_str(), cond.c_str());
    }
    case Op::Bcctr: {
      std::string pre;
      const std::string cond = branch_cond(d, pre);
      if (!pre.empty()) { warn(x, addr, d, "bcctr with CTR decrement is an invalid form"); return trap_stmt(addr, d); }
      if (d.lk())
        return F("if (%s) { c.lr = %s; ppu::call_indirect(c, c.ctr); }", cond.c_str(), hex64(next).c_str());
      if (cond != "true") return F("if (%s) { ppu::call_indirect(c, c.ctr); return; }", cond.c_str());

      if (auto jt = recover_jump_table(x.img, jt_opts(x), x.self.start, addr)) {
        std::set<uint64_t> cases(jt->targets.begin(), jt->targets.end());
        std::string s = "switch ((uint32_t)c.ctr) {";
        bool any_outside = false;
        for (uint64_t t : cases) {
          if (in_self(x, t)) s += F(" case %s: goto %s;", hex32(t).c_str(), label(t).c_str());
          else any_outside = true;
        }
        if (any_outside) warn(x, addr, d, "jump table has targets outside the function; dispatched indirectly");
        return s + " default: ppu::call_indirect(c, c.ctr); return; }";
      }
      if (is_pointer_call(x.img, x.self.start, addr))
        return "{ ppu::call_indirect(c, c.ctr); return; }  // call through function descriptor";
      warn(x, addr, d, "bctr: no jump table recognised, treated as indirect tail call");
      return "{ ppu::call_indirect(c, c.ctr); return; }";
    }
    case Op::Sc: return F("ppu::syscall(c, %s);", hex32(addr).c_str());

    default: break;
  }
  warn(x, addr, d, "no lifter for this instruction");
  return trap_stmt(addr, d);
}

std::string lift_function(FnCtx& x) {
  const FuncRange& range = x.self;
  std::vector<Insn> insns;
  std::set<uint64_t> labels, data;
  for (uint64_t a = range.start; a < range.end; a += 4) {
    uint32_t raw = 0;
    x.img.read32(a, raw);
    insns.push_back(decode(raw));
    const Insn& d = insns.back();
    if ((d.op == Op::B || d.op == Op::Bc) && !d.lk()) {
      const uint64_t t = branch_target(d, a) & 0xFFFFFFFFull;
      if (in_self(x, t)) labels.insert(t);
    } else if (d.op == Op::Bcctr && !d.lk() && d.bo() == 20) {
      static const char* dbg = std::getenv("PS3RECOMP_DEBUG_JT");
      if (dbg && std::strtoull(dbg, nullptr, 16) == a) {
        const AnalysisOptions o = jt_opts(x);
        auto jt = recover_jump_table(x.img, o, range.start, a);
        std::fprintf(stderr, "jt %08llX in fn %08llX toc %08llX alts %zu -> %s table %08llX n %zu\n",
                     (unsigned long long)a, (unsigned long long)range.start, (unsigned long long)o.toc, o.alt_tocs.size(),
                     jt ? "ok" : "none", jt ? (unsigned long long)jt->table : 0ull, jt ? jt->targets.size() : 0);
      }
      if (auto jt = recover_jump_table(x.img, jt_opts(x), range.start, a)) {
        for (uint64_t t : jt->targets)
          if (in_self(x, t)) labels.insert(t);
        if (jt->table == a + 4)  // inline table: its words are data
          for (size_t k = 0; k < jt->targets.size(); ++k) data.insert(jt->table + 4 * k);
      }
    }
  }

  std::ostringstream o;
  o << "void " << func_name(range.start) << "(ppu::Context& c) {\n";
  for (size_t i = 0; i < insns.size(); ++i) {
    const uint64_t a = range.start + i * 4;
    if (data.count(a)) {
      if (!data.count(a - 4)) o << "  // " << F("%08llX", (unsigned long long)a) << "  jump table data\n";
      continue;
    }
    if (labels.count(a)) o << label(a) << ":;\n";
    o << "  // " << F("%08llX", (unsigned long long)a) << "  " << disassemble(insns[i], a) << "\n";
    o << "  " << lift_insn(x, insns[i], a) << "\n";
  }
  o << "}\n\n";
  return o.str();
}

}  // namespace

std::string func_name(uint64_t addr) { return F("fn_%08llX", (unsigned long long)addr); }

std::vector<OutputFile> lift_program(const elf::Image& img, const std::map<uint64_t, FuncRange>& funcs,
                                     const LiftOptions& opt, const std::string& stem,
                                     std::vector<std::string>& warnings) {
  // Split into files of at most funcs_per_file functions and at most the average
  // amount of code per file, so no single file (dominated by a few huge functions)
  // becomes the long pole of a parallel build.
  uint64_t total_bytes = 0;
  for (const auto& [start, r] : funcs) total_bytes += r.end - r.start;
  const uint64_t byte_budget =
      opt.funcs_per_file && !funcs.empty() ? std::max<uint64_t>(total_bytes * opt.funcs_per_file / funcs.size(), 4096) : 0;
  std::vector<std::vector<FuncRange>> chunks(1);
  uint64_t chunk_bytes = 0;
  for (const auto& [start, r] : funcs) {
    if (opt.funcs_per_file && !chunks.back().empty() &&
        (chunks.back().size() >= opt.funcs_per_file || chunk_bytes + (r.end - r.start) > byte_budget)) {
      chunks.emplace_back();
      chunk_bytes = 0;
    }
    chunks.back().push_back(r);
    chunk_bytes += r.end - r.start;
  }
  const bool split = opt.funcs_per_file != 0;

  std::vector<OutputFile> out;
  for (size_t ci = 0; ci < chunks.size(); ++ci) {
    std::set<uint64_t> referenced;
    std::string body;
    for (const FuncRange& r : chunks[ci]) {
      FnCtx x{img, funcs, opt, r, warnings, referenced};
      body += lift_function(x);
    }

    std::ostringstream o;
    o << "// Generated by ps3recomp. Do not edit.\n#include \"ppu_runtime.h\"\n\n";
    // Declarations of everything this chunk calls or defines.
    std::set<uint64_t> declare = referenced;
    for (const FuncRange& r : chunks[ci]) declare.insert(r.start);
    for (uint64_t a : declare) {
      o << "void " << func_name(a) << "(ppu::Context& c);\n";
      if (opt.hooks) o << "extern ppu::Fn " << slot_name(a) << ";\n";
    }
    o << "\n" << body;
    if (opt.hooks)
      for (const FuncRange& r : chunks[ci]) o << "ppu::Fn " << slot_name(r.start) << " = " << func_name(r.start) << ";\n";

    o << "\nvoid ppu_register_chunk_" << ci << "() {\n";
    for (const FuncRange& r : chunks[ci])
      o << "  ppu::register_function(" << hex32(r.start) << ", " << func_name(r.start) << ", "
        << (opt.hooks ? "&" + slot_name(r.start) : std::string("nullptr")) << ");\n";
    o << "}\n";

    if (!split) {
      o << "\nvoid ppu_register_functions() { ppu_register_chunk_0(); }\n";
      out.push_back({stem + ".cpp", o.str()});
    } else {
      out.push_back({F("%s_%03zu.cpp", stem.c_str(), ci), o.str()});
    }
  }

  if (split) {
    std::ostringstream o;
    o << "// Generated by ps3recomp. Do not edit.\n#include \"ppu_runtime.h\"\n\n";
    for (size_t ci = 0; ci < chunks.size(); ++ci) o << "void ppu_register_chunk_" << ci << "();\n";
    o << "\nvoid ppu_register_functions() {\n";
    for (size_t ci = 0; ci < chunks.size(); ++ci) o << "  ppu_register_chunk_" << ci << "();\n";
    o << "}\n";
    out.push_back({stem + "_register.cpp", o.str()});
  }
  return out;
}

}  // namespace ppu
