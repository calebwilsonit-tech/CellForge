#include "ppu_decode.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace ppu {
namespace {

enum Form { PRIM, DS, MD, MDS, X10, X9, XS, A5, VA, VX, VC };

const OpInfo kInfo[] = {
    {".long", "", 0, VKind::None},
#define OP(e, m, f, fl, p, form, xo) {m, f, uint8_t(fl), VKind::None},
#include "ppu_ops.inc"
#undef OP
#define VOP(e, n, k, form, xo)                                                              \
  {#n,                                                                                      \
   VKind::k == VKind::DABC   ? "vD,vA,vC,vB"                                                \
   : VKind::k == VKind::DABS ? "vD,vA,vB,SHB"                                               \
   : VKind::k == VKind::DB   ? "vD,vB"                                                      \
   : VKind::k == VKind::DBU  ? "vD,vB,UIMM5"                                                \
   : VKind::k == VKind::DS   ? "vD,SIMM5"                                                   \
   : VKind::k == VKind::MFVSCR ? "vD"                                                       \
   : VKind::k == VKind::MTVSCR ? "vB"                                                       \
                               : "vD,vA,vB",                                                \
   0, VKind::k},
#include "ppu_vmx_ops.inc"
#undef VOP
};
static_assert(sizeof(kInfo) / sizeof(kInfo[0]) == size_t(Op::Count), "op table out of sync");

struct Tables {
  std::array<Op, 64> prim{};
  std::array<Op, 4> ds58{}, ds62{};
  std::array<Op, 8> md{};
  std::array<Op, 16> mds{};
  std::array<Op, 1024> x19{}, x31{}, x63{};
  std::array<Op, 512> x9{}, xs{};
  std::array<Op, 32> a59{}, a63{};
  std::array<Op, 64> va{};
  std::array<Op, 2048> vx{};
  std::array<Op, 1024> vc{};

  Tables() {
    auto put = [this](Op op, int p, int form, int xo) {
      switch (form) {
        case PRIM: prim[p] = op; break;
        case DS: (p == 58 ? ds58 : ds62)[xo] = op; break;
        case MD: md[xo] = op; break;
        case MDS: mds[xo] = op; break;
        case X10: (p == 19 ? x19 : p == 31 ? x31 : x63)[xo] = op; break;
        case X9: x9[xo] = op; break;
        case XS: xs[xo] = op; break;
        case A5: (p == 59 ? a59 : a63)[xo] = op; break;
        case VA: va[xo] = op; break;
        case VX: vx[xo] = op; break;
        case VC: vc[xo] = op; break;
      }
    };
#define OP(e, m, f, fl, p, form, xo) put(Op::e, p, form, xo);
#include "ppu_ops.inc"
#undef OP
#define VOP(e, n, k, form, xo) put(Op::e, 4, form, xo);
#include "ppu_vmx_ops.inc"
#undef VOP
  }
};

const Tables& tables() {
  static const Tables t;
  return t;
}

std::string fmt(const char* f, ...) {
  char buf[192];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

}  // namespace

const OpInfo& op_info(Op op) { return kInfo[size_t(op)]; }

Insn decode(uint32_t i) {
  const Tables& t = tables();
  Insn d;
  d.raw = i;
  const uint32_t p = i >> 26;
  switch (p) {
    case 4:
      d.op = t.va[i & 63];
      if (d.op == Op::Invalid) d.op = t.vx[i & 0x7FF];
      if (d.op == Op::Invalid) d.op = t.vc[i & 0x3FF];
      break;
    case 19: d.op = t.x19[(i >> 1) & 0x3FF]; break;
    case 30:
      d.op = t.md[(i >> 2) & 7];
      if (((i >> 2) & 7) >= 4) d.op = t.mds[(i >> 1) & 15];
      break;
    case 31:
      d.op = t.x31[(i >> 1) & 0x3FF];
      if (d.op == Op::Invalid) d.op = t.x9[(i >> 1) & 0x1FF];
      if (d.op == Op::Invalid) d.op = t.xs[(i >> 2) & 0x1FF];
      break;
    case 58: d.op = t.ds58[i & 3]; break;
    case 62: d.op = t.ds62[i & 3]; break;
    case 59:
      d.op = t.a59[(i >> 1) & 31];
      break;
    case 63:
      if (((i >> 1) & 31) >= 16) d.op = t.a63[(i >> 1) & 31];
      else d.op = t.x63[(i >> 1) & 0x3FF];
      break;
    default: d.op = t.prim[p]; break;
  }
  return d;
}

uint64_t branch_target(const Insn& d, uint64_t addr) {
  const int64_t off = d.op == Op::B ? d.li() : d.bd();
  return d.aa() ? uint64_t(off) : addr + uint64_t(off);
}

std::string disassemble(const Insn& d, uint64_t addr) {
  const OpInfo& info = op_info(d.op);
  if (d.op == Op::Invalid) return fmt(".long 0x%08X", d.raw);

  // Common simplified mnemonics.
  switch (d.op) {
    case Op::Addi: if (d.ra() == 0) return fmt("li r%u,%lld", d.rt(), (long long)d.simm()); break;
    case Op::Addis: if (d.ra() == 0) return fmt("lis r%u,%lld", d.rt(), (long long)d.simm()); break;
    case Op::Ori: if (d.raw == 0x60000000) return "nop"; break;
    case Op::Or: if (d.rt() == d.rb()) return fmt("mr%s r%u,r%u", d.rc() ? "." : "", d.ra(), d.rt()); break;
    case Op::Mfspr:
      if (d.spr() == 8) return fmt("mflr r%u", d.rt());
      if (d.spr() == 9) return fmt("mfctr r%u", d.rt());
      break;
    case Op::Mtspr:
      if (d.spr() == 8) return fmt("mtlr r%u", d.rt());
      if (d.spr() == 9) return fmt("mtctr r%u", d.rt());
      break;
    case Op::Bclr: if (d.bo() == 20) return d.lk() ? "blrl" : "blr"; break;
    case Op::Bcctr: if (d.bo() == 20) return d.lk() ? "bctrl" : "bctr"; break;
    default: break;
  }

  std::string name = info.mnemonic;
  if ((info.flags & F_OE) && d.oe()) name += "o";
  if ((info.flags & F_RC) && d.rc()) name += ".";
  if (info.vkind == VKind::CMP && d.vrc()) name += ".";
  if ((info.flags & F_LK) && d.lk()) name += "l";
  if ((info.flags & F_AA) && d.aa()) name += "a";

  std::string ops;
  const char* f = info.operands;
  while (*f) {
    const char* end = std::strchr(f, ',');
    const std::string tok = end ? std::string(f, end) : std::string(f);
    f = end ? end + 1 : f + std::strlen(f);
    if (!ops.empty()) ops += ",";
    const auto r0 = [&](unsigned r) { return r ? fmt("r%u", r) : std::string("0"); };
    if (tok == "D" || tok == "S") ops += fmt("r%u", d.rt());
    else if (tok == "A") ops += fmt("r%u", d.ra());
    else if (tok == "A0") ops += r0(d.ra());
    else if (tok == "B") ops += fmt("r%u", d.rb());
    else if (tok == "fD") ops += fmt("f%u", d.rt());
    else if (tok == "fA") ops += fmt("f%u", d.ra());
    else if (tok == "fB") ops += fmt("f%u", d.rb());
    else if (tok == "fC") ops += fmt("f%u", d.rc5());
    else if (tok == "vD") ops += fmt("v%u", d.rt());
    else if (tok == "vA") ops += fmt("v%u", d.ra());
    else if (tok == "vB") ops += fmt("v%u", d.rb());
    else if (tok == "vC") ops += fmt("v%u", d.rc5());
    else if (tok == "SI") ops += fmt("%lld", (long long)d.simm());
    else if (tok == "UI") ops += fmt("0x%llX", (unsigned long long)d.uimm());
    else if (tok == "d(A0)") ops += fmt("%lld(%s)", (long long)d.simm(), r0(d.ra()).c_str());
    else if (tok == "ds(A0)") ops += fmt("%lld(%s)", (long long)d.ds(), r0(d.ra()).c_str());
    else if (tok == "crfD") ops += fmt("cr%u", d.crfd());
    else if (tok == "crfS") ops += fmt("cr%u", d.crfs());
    else if (tok == "L") ops += fmt("%u", unsigned(d.l()));
    else if (tok == "TO" || tok == "BO" || tok == "crbD") ops += fmt("%u", d.rt());
    else if (tok == "BI" || tok == "crbA" || tok == "UIMM5") ops += fmt("%u", d.ra());
    else if (tok == "crbB" || tok == "SH") ops += fmt("%u", d.rb());
    else if (tok == "MB") ops += fmt("%u", d.rc5());
    else if (tok == "ME") ops += fmt("%u", d.me5());
    else if (tok == "SH6") ops += fmt("%u", d.sh6());
    else if (tok == "MB6") ops += fmt("%u", d.mb6());
    else if (tok == "SPR") ops += fmt("%u", d.spr());
    else if (tok == "CRM") ops += fmt("0x%02X", d.crm());
    else if (tok == "FM") ops += fmt("0x%02X", d.fm());
    else if (tok == "SHB") ops += fmt("%u", d.shb());
    else if (tok == "SIMM5") ops += fmt("%d", d.simm5());
    else if (tok == "TGT") ops += fmt("0x%llX", (unsigned long long)branch_target(d, addr));
    else ops += tok;
  }
  return ops.empty() ? name : name + " " + ops;
}

}  // namespace ppu
