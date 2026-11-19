#pragma once
// Table-driven decoder + disassembler for the Cell PPU (PowerPC 64 + AltiVec).
// The instruction set is defined in ppu_ops.inc and ppu_vmx_ops.inc.

#include <cstdint>
#include <string>

namespace ppu {

enum class Op : uint16_t {
  Invalid,
#define OP(e, m, f, fl, p, form, xo) e,
#include "ppu_ops.inc"
#undef OP
#define VOP(e, n, k, form, xo) e,
#include "ppu_vmx_ops.inc"
#undef VOP
  Count
};

enum OpFlags : uint8_t { F_RC = 1, F_OE = 2, F_LK = 4, F_AA = 8 };
enum class VKind : uint8_t { None, DAB, DB, DABC, DABS, DBU, DS, CMP, MFVSCR, MTVSCR };

struct OpInfo {
  const char* mnemonic;
  const char* operands;
  uint8_t flags;
  VKind vkind;  // VKind::None for non-VMX ops
};

const OpInfo& op_info(Op op);

// A decoded instruction: the op plus accessors for every field encoding.
struct Insn {
  Op op = Op::Invalid;
  uint32_t raw = 0;

  unsigned rt() const { return (raw >> 21) & 31; }   // also RS, FRT, VD, BO, TO, crbD
  unsigned ra() const { return (raw >> 16) & 31; }   // also FRA, VA, BI, crbA, UIMM5
  unsigned rb() const { return (raw >> 11) & 31; }   // also FRB, VB, crbB, SH (rlw*)
  unsigned rc5() const { return (raw >> 6) & 31; }   // FRC, VC, MB (rlw*)
  unsigned me5() const { return (raw >> 1) & 31; }   // ME (rlw*)
  int64_t simm() const { return int16_t(raw & 0xFFFF); }
  uint64_t uimm() const { return raw & 0xFFFF; }
  int64_t ds() const { return int16_t(raw & 0xFFFC); }
  bool rc() const { return raw & 1; }
  bool oe() const { return (raw >> 10) & 1; }
  bool lk() const { return raw & 1; }
  bool aa() const { return (raw >> 1) & 1; }
  bool vrc() const { return (raw >> 10) & 1; }       // VC-form record bit
  unsigned crfd() const { return (raw >> 23) & 7; }
  unsigned crfs() const { return (raw >> 18) & 7; }
  bool l() const { return (raw >> 21) & 1; }
  unsigned sh6() const { return ((raw >> 11) & 31) | (((raw >> 1) & 1) << 5); }
  unsigned mb6() const { return ((raw >> 6) & 31) | (raw & 0x20); }
  unsigned spr() const { return ((raw >> 16) & 31) | (((raw >> 11) & 31) << 5); }
  unsigned crm() const { return (raw >> 12) & 0xFF; }
  unsigned fm() const { return (raw >> 17) & 0xFF; }
  unsigned shb() const { return (raw >> 6) & 15; }
  int simm5() const { return int((ra() ^ 16u)) - 16; }
  unsigned bo() const { return rt(); }
  unsigned bi() const { return ra(); }
  int64_t li() const {
    const uint32_t v = raw & 0x03FFFFFC;
    return int64_t(int32_t(v << 6) >> 6);
  }
  int64_t bd() const { return int16_t(raw & 0xFFFC); }
};

Insn decode(uint32_t raw);

// Target of a B / Bc at `addr`.
uint64_t branch_target(const Insn& d, uint64_t addr);

std::string disassemble(const Insn& d, uint64_t addr);

}  // namespace ppu
