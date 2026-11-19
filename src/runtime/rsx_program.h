#pragma once
// RSX shader programs: decoders for the NV40-family vertex program (VP) and
// fragment program (FP) instruction formats, plus a reference interpreter.
//
// Field layouts follow the publicly documented RSX encodings:
//   VP: 4 x 32-bit words (D0..D3), three 17-bit source operands split across words.
//   FP: 4 x 32-bit words (OPDEST, SRC0, SRC1, SRC2), each stored with its two
//       16-bit halves swapped; an inline vec4 constant follows any instruction
//       that reads a constant.
// The decoded form is shared by this interpreter and the future GPU backend.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rsx {

using vec4 = std::array<float, 4>;

// ---------------- vertex programs ----------------
enum VpRegType : uint8_t { VP_TEMP = 1, VP_INPUT = 2, VP_CONST = 3 };

struct VpSrc {
  uint8_t type = 0, index = 0;   // index: temp register (type TEMP); input/const come from the instruction
  uint8_t swz[4] = {0, 1, 2, 3};
  bool neg = false, abs = false;
};

struct VpInsn {
  uint8_t vec_op = 0, sca_op = 0;
  VpSrc src[3];
  uint8_t input = 0;          // input attribute for INPUT sources
  uint16_t constant = 0;      // constant index for CONST sources
  bool vec_result = false;    // vector result goes to the output register
  uint8_t dst_tmp = 0x3F;     // vector temp destination (0x3F = none)
  uint8_t dst_out = 0x1F;     // output register (0x1F = none)
  uint8_t sca_dst_tmp = 0x3F; // scalar temp destination
  uint8_t vec_mask = 0, sca_mask = 0;  // bit0 = x .. bit3 = w
  bool saturate = false;
  bool cond_test = false;     // writes only where the condition holds
  bool end = false;
  // Address registers A0/A1 (relative constant addressing, e.g. bone matrices).
  uint8_t addr_reg = 0;       // which A register (ARL destination / index source)
  uint8_t addr_swz = 0;       // component of A added to the constant index
  bool index_const = false;   // constant index += A[addr_reg].addr_swz
  // Condition codes CC0/CC1.
  uint8_t cond = 0;           // bit0 = LT, bit1 = EQ, bit2 = GT (7 = always)
  uint8_t cond_swz[4] = {0, 1, 2, 3};
  uint8_t cond_reg = 0;       // CC register tested / updated
  bool cond_update = false;   // results also update the CC register
  uint16_t branch_target = 0; // BRA/BRI/CAL/CLI target instruction
};

// Decodes one instruction from four host-order words.
VpInsn decode_vp(const uint32_t w[4]);

// Vertex program outputs (VP dst register numbers).
enum VpOutput : uint8_t {
  VPO_HPOS = 0, VPO_COL0 = 1, VPO_COL1 = 2, VPO_BFC0 = 3, VPO_BFC1 = 4, VPO_FOG = 5, VPO_PSIZE = 6,
  VPO_TEX0 = 7,  // TEX0..TEX7 = 7..14, TEX8 = 15, TEX9 = 16
  VPO_COUNT = 17
};

struct VpState {
  std::vector<VpInsn> program;   // 512 slots
  std::vector<vec4> constants;   // 512 vec4
  uint32_t start = 0;
  uint64_t prog_ver = 1;         // bumped on program/start writes (for GPU VP cache)
  uint64_t const_ver = 1;        // bumped on constant writes (for GPU VP UBO cache)
  VpState() : program(512), constants(512, vec4{0, 0, 0, 0}) {}
};

// Runs the program from `start`. Returns false (and sets `error`) on an unsupported op.
bool run_vp(const VpState& vp, const vec4 inputs[16], vec4 outputs[VPO_COUNT], std::string* error);

// ---------------- fragment programs ----------------
enum FpRegType : uint8_t { FP_TEMP = 0, FP_INPUT = 1, FP_CONST = 2 };

// Fragment inputs (src_attr_reg_num).
enum FpInput : uint8_t { FPI_WPOS = 0, FPI_COL0 = 1, FPI_COL1 = 2, FPI_FOG = 3, FPI_TEX0 = 4, FPI_COUNT = 15 };

struct FpSrc {
  uint8_t type = 0, index = 0;
  bool fp16 = false;
  uint8_t swz[4] = {0, 1, 2, 3};
  bool neg = false, abs = false;
};

struct FpInsn {
  uint8_t op = 0;              // full opcode (6 bits + high bit from SRC1)
  uint8_t dst = 0;
  bool dst_fp16 = false, no_dest = false, saturate = false, end = false;
  bool exp_tex = false;        // OPDEST bit 21 (_bx2): expand texture fetch [0,1] -> [-1,1] via 2*x - 1
  uint8_t mask = 0xF;          // bit0 = x .. bit3 = w
  uint8_t input = 0;           // attribute for INPUT sources
  uint8_t tex = 0;             // texture unit
  uint8_t scale = 0;           // result scale (SRC1.scale)
  bool set_cond = false;
  uint8_t cond = 7;            // 0..7: FL, LT, EQ, LE, GT, NE, GE, TR
  uint8_t cond_swz[4] = {0, 1, 2, 3};
  FpSrc src[3];
  vec4 constant{};             // inline constant (if any source is CONST)
  bool has_constant = false;
};

// Decodes a program starting at guest memory `mem` (raw big-endian bytes).
// Stops at the instruction with `end` set or after max_insns.
std::vector<FpInsn> decode_fp(const uint8_t* mem, size_t max_insns = 4096);

// Texture sampling callback: unit, coordinates (s, t, r, q), lod bias, result.
using TexSampler = vec4 (*)(void* user, uint32_t unit, const vec4& coord);

struct FpResult {
  vec4 color{};
  bool discard = false;
};

bool run_fp(const std::vector<FpInsn>& prog, const vec4 inputs[FPI_COUNT], bool fp32_output, TexSampler tex,
            void* tex_user, FpResult& out, std::string* error);

std::string disasm_vp(const VpInsn& i);
std::string disasm_fp(const FpInsn& i);

// FP opcode numbers used by the interpreter.
enum FpOp : uint8_t {
  FP_NOP = 0, FP_MOV = 1, FP_MUL = 2, FP_ADD = 3, FP_MAD = 4, FP_DP3 = 5, FP_DP4 = 6, FP_DST = 7, FP_MIN = 8,
  FP_MAX = 9, FP_SLT = 10, FP_SGE = 11, FP_SLE = 12, FP_SGT = 13, FP_SNE = 14, FP_SEQ = 15, FP_FRC = 16,
  FP_FLR = 17, FP_KIL = 18, FP_DDX = 21, FP_DDY = 22, FP_TEX = 23, FP_TXP = 24, FP_TXD = 25, FP_RCP = 26,
  FP_RSQ = 27, FP_EX2 = 28, FP_LG2 = 29, FP_LIT = 30, FP_LRP = 31, FP_STR = 32, FP_SFL = 33, FP_COS = 34,
  FP_SIN = 35, FP_POW = 38, FP_DP2A = 46, FP_TXL = 47, FP_TXB = 49, FP_REFL = 54, FP_DP2 = 56, FP_NRM = 57,
  FP_DIV = 58, FP_DIVSQ = 59, FP_FENCT = 61, FP_FENCB = 62
};

// VP opcode numbers.
enum VpVecOp : uint8_t {
  VV_NOP = 0, VV_MOV, VV_MUL, VV_ADD, VV_MAD, VV_DP3, VV_DPH, VV_DP4, VV_DST, VV_MIN, VV_MAX, VV_SLT, VV_SGE,
  VV_ARL, VV_FRC, VV_FLR, VV_SEQ, VV_SFL, VV_SGT, VV_SLE, VV_SNE, VV_STR, VV_SSG, VV_TXL = 0x19
};
enum VpScaOp : uint8_t {
  VS_NOP = 0, VS_MOV, VS_RCP, VS_RCC, VS_RSQ, VS_EXP, VS_LOG, VS_LIT, VS_BRA, VS_BRI, VS_CAL, VS_CLI, VS_RET,
  VS_LG2, VS_EX2, VS_SIN, VS_COS, VS_BRB, VS_CLB, VS_PSH, VS_POP
};

}  // namespace rsx
