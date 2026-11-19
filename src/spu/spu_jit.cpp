// x86-64 AVX2 basic-block JIT for the Cell SPU.
//
// Register convention inside JIT blocks (set up once by the trampoline):
//   rbx   = SpuCore* self
//   rsi   = V128* gpr (&self->gpr[0])
//   rdi   = uint8_t* ls (256 KB Local Store)
//   r12   = BlockFn* blocks (65536 entries; blocks[pc>>2] == [r12 + pc*2])
//   r13   = const detail::QuadShuffleTables* (&detail::g_qshuf)
//   r14d  = block budget counter (decrements on each block exit)
//   r15   = epilogue address (jmp r15 returns eax to C++)
//   xmm15 = bswap32 shuffle mask (callee-saved in Win64 ABI -> survives C++ calls)
//   rsp   = 16-byte aligned; [rsp..rsp+31] is Win64 shadow space for C++ calls.
//
// Volatile scratch inside blocks: rax, rcx, rdx, r8, r9, r10, r11, xmm0..xmm5.

#include "spu_jit.h"
#include "spu_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
#define SPU_JIT_SUPPORTED 1
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#define SPU_JIT_SUPPORTED 0
#endif

namespace spu {

#if !SPU_JIT_SUPPORTED

SpuJit::SpuJit() = default;
SpuJit::~SpuJit() = default;
uint32_t SpuJit::run(SpuCore&) { return 0; }
void SpuJit::invalidate_ls(uint32_t, uint32_t) {}
void SpuJit::invalidate_all() {}

#else

namespace {

using detail::Op;
using detail::QuadShuffleTables;
using detail::sext;

constexpr size_t kCodeBufSize = 8 * 1024 * 1024;  // 8 MB executable buffer
constexpr size_t kBlockCount = kLsSize / 4;       // 65536
constexpr int kMaxBlockInsns = 64;
constexpr int kSliceBudget = 1024;

// x86-64 GP register numbers
enum Reg : uint8_t {
  RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4, RBP = 5, RSI = 6, RDI = 7,
  R8 = 8, R9 = 9, R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15,
};

// x86-64 XMM register numbers
enum Xmm : uint8_t {
  XMM0 = 0, XMM1 = 1, XMM2 = 2, XMM3 = 3, XMM4 = 4, XMM5 = 5, XMM15 = 15,
};

// Tiny x86-64 + VEX (AVX/AVX2) machine-code emitter.
struct Emit {
  uint8_t* buf;
  size_t pos;
  size_t cap;
  bool overflow = false;

  void b(uint8_t v) {
    if (pos < cap) buf[pos++] = v;
    else overflow = true;
  }
  void u16(uint16_t v) { b(uint8_t(v)); b(uint8_t(v >> 8)); }
  void u32(uint32_t v) {
    if (pos + 4 <= cap) { std::memcpy(buf + pos, &v, 4); pos += 4; }
    else overflow = true;
  }
  void u64(uint64_t v) {
    if (pos + 8 <= cap) { std::memcpy(buf + pos, &v, 8); pos += 8; }
    else overflow = true;
  }

  void rex(bool w, uint8_t reg, uint8_t idx, uint8_t rm) {
    const uint8_t v = 0x40 | (w ? 8 : 0) | ((reg >> 3) << 2) | ((idx >> 3) << 1) | (rm >> 3);
    if (v != 0x40) b(v);
  }

  // ModRM + optional SIB + displacement for [base + disp32]
  void modrm_disp(uint8_t reg, uint8_t base, int32_t disp) {
    const uint8_t r = reg & 7, b7 = base & 7;
    if (disp == 0 && b7 != 5) {
      b(uint8_t((0 << 6) | (r << 3) | b7));
      if (b7 == 4) b(0x24);
    } else if (disp >= -128 && disp <= 127) {
      b(uint8_t((1 << 6) | (r << 3) | b7));
      if (b7 == 4) b(0x24);
      b(uint8_t(disp));
    } else {
      b(uint8_t((2 << 6) | (r << 3) | b7));
      if (b7 == 4) b(0x24);
      u32(uint32_t(disp));
    }
  }

  // ModRM + SIB for [base + index*scale + disp] (scale_pow: 0=1,1=2,2=4,3=8)
  void modrm_sib(uint8_t reg, uint8_t base, uint8_t index, uint8_t scale_pow, int32_t disp = 0) {
    const uint8_t r = reg & 7, b7 = base & 7, i7 = index & 7;
    const uint8_t sib = uint8_t((scale_pow << 6) | (i7 << 3) | b7);
    if (disp == 0 && b7 != 5) {
      b(uint8_t((0 << 6) | (r << 3) | 4));
      b(sib);
    } else if (disp >= -128 && disp <= 127) {
      b(uint8_t((1 << 6) | (r << 3) | 4));
      b(sib);
      b(uint8_t(disp));
    } else {
      b(uint8_t((2 << 6) | (r << 3) | 4));
      b(sib);
      u32(uint32_t(disp));
    }
  }

  void modrm_rr(uint8_t reg, uint8_t rm) {
    b(uint8_t(0xC0 | ((reg & 7) << 3) | (rm & 7)));
  }

  // 3-byte VEX prefix (L=0, 128-bit)
  void vex3(uint8_t map, uint8_t pp, uint8_t reg, uint8_t vvvv, uint8_t rm, uint8_t idx = 0, bool w = false) {
    b(0xC4);
    const uint8_t R = (reg & 8) ? 0 : 0x80;
    const uint8_t X = (idx & 8) ? 0 : 0x40;
    const uint8_t B = (rm & 8) ? 0 : 0x20;
    b(uint8_t(R | X | B | (map & 0x1F)));
    b(uint8_t((w ? 0x80 : 0) | ((~vvvv & 0xF) << 3) | (pp & 3)));
  }

  // ---- GP instructions ----
  void push(Reg r) { if (r >= 8) b(0x41); b(uint8_t(0x50 + (r & 7))); }
  void pop(Reg r) { if (r >= 8) b(0x41); b(uint8_t(0x58 + (r & 7))); }
  void ret() { b(0xC3); }

  void mov_r64_r64(Reg dst, Reg src) { rex(true, src, 0, dst); b(0x89); modrm_rr(src, dst); }
  void mov_r32_r32(Reg dst, Reg src) { rex(false, src, 0, dst); b(0x89); modrm_rr(src, dst); }
  void mov_r64_imm64(Reg dst, uint64_t imm) { rex(true, 0, 0, dst); b(uint8_t(0xB8 + (dst & 7))); u64(imm); }
  void mov_r32_imm32(Reg dst, uint32_t imm) { rex(false, 0, 0, dst); b(uint8_t(0xB8 + (dst & 7))); u32(imm); }

  void mov_r32_m(Reg dst, Reg base, int32_t disp) { rex(false, dst, 0, base); b(0x8B); modrm_disp(dst, base, disp); }
  void mov_r64_m(Reg dst, Reg base, int32_t disp) { rex(true, dst, 0, base); b(0x8B); modrm_disp(dst, base, disp); }
  void mov_m_r32(Reg base, int32_t disp, Reg src) { rex(false, src, 0, base); b(0x89); modrm_disp(src, base, disp); }
  void mov_m_r64(Reg base, int32_t disp, Reg src) { rex(true, src, 0, base); b(0x89); modrm_disp(src, base, disp); }
  void mov_r64_sib(Reg dst, Reg base, Reg idx, uint8_t scale, int32_t disp = 0) {
    rex(true, dst, idx, base); b(0x8B); modrm_sib(dst, base, idx, scale, disp);
  }
  void movzx_r32_m16(Reg dst, Reg base, int32_t disp) {
    rex(false, dst, 0, base); b(0x0F); b(0xB7); modrm_disp(dst, base, disp);
  }

  void add_r64_imm8(Reg dst, int8_t imm) { rex(true, 0, 0, dst); b(0x83); modrm_rr(0, dst); b(uint8_t(imm)); }
  void sub_r64_imm8(Reg dst, int8_t imm) { rex(true, 0, 0, dst); b(0x83); modrm_rr(5, dst); b(uint8_t(imm)); }
  void add_r32_r32(Reg dst, Reg src) { rex(false, src, 0, dst); b(0x01); modrm_rr(src, dst); }
  void add_r32_imm32(Reg dst, int32_t imm) {
    rex(false, 0, 0, dst);
    if (imm >= -128 && imm <= 127) { b(0x83); modrm_rr(0, dst); b(uint8_t(imm)); }
    else { b(0x81); modrm_rr(0, dst); u32(uint32_t(imm)); }
  }
  void and_r32_imm32(Reg dst, uint32_t imm) {
    rex(false, 0, 0, dst);
    if (int32_t(imm) >= -128 && int32_t(imm) <= 127) { b(0x83); modrm_rr(4, dst); b(uint8_t(imm)); }
    else { b(0x81); modrm_rr(4, dst); u32(imm); }
  }
  void neg_r32(Reg dst) { rex(false, 0, 0, dst); b(0xF7); modrm_rr(3, dst); }
  void shr_r32_imm8(Reg dst, uint8_t s) { rex(false, 0, 0, dst); b(0xC1); modrm_rr(5, dst); b(s); }
  void shl_r32_imm8(Reg dst, uint8_t s) { rex(false, 0, 0, dst); b(0xC1); modrm_rr(4, dst); b(s); }
  void cmp_r32_imm8(Reg dst, int8_t imm) { rex(false, 0, 0, dst); b(0x83); modrm_rr(7, dst); b(uint8_t(imm)); }
  void test_r32_r32(Reg a, Reg b_reg) { rex(false, b_reg, 0, a); b(0x85); modrm_rr(b_reg, a); }
  void test_r64_r64(Reg a, Reg b_reg) { rex(true, b_reg, 0, a); b(0x85); modrm_rr(b_reg, a); }
  void dec_r32(Reg dst) { rex(false, 0, 0, dst); b(0xFF); modrm_rr(1, dst); }
  void cmove_r32_r32(Reg dst, Reg src) { rex(false, dst, 0, src); b(0x0F); b(0x44); modrm_rr(dst, src); }
  void cmovne_r32_r32(Reg dst, Reg src) { rex(false, dst, 0, src); b(0x0F); b(0x45); modrm_rr(dst, src); }
  void cmova_r32_r32(Reg dst, Reg src) { rex(false, dst, 0, src); b(0x0F); b(0x47); modrm_rr(dst, src); }

  void jmp_reg(Reg r) { rex(false, 0, 0, r); b(0xFF); modrm_rr(4, r); }
  void call_reg(Reg r) { rex(false, 0, 0, r); b(0xFF); modrm_rr(2, r); }
  void jle_rel8(int8_t rel) { b(0x7E); b(uint8_t(rel)); }
  void jz_rel8(int8_t rel) { b(0x74); b(uint8_t(rel)); }
  void jnz_rel8(int8_t rel) { b(0x75); b(uint8_t(rel)); }

  // ---- VEX 128-bit SIMD instructions ----
  void vmovdqu_x_m(Xmm dst, Reg base, int32_t disp) {
    vex3(1, 2, dst, 0, base); b(0x6F); modrm_disp(dst, base, disp);
  }
  void vmovdqu_m_x(Reg base, int32_t disp, Xmm src) {
    vex3(1, 2, src, 0, base); b(0x7F); modrm_disp(src, base, disp);
  }
  void vmovdqu_x_sib(Xmm dst, Reg base, Reg idx, uint8_t scale, int32_t disp = 0) {
    vex3(1, 2, dst, 0, base, idx); b(0x6F); modrm_sib(dst, base, idx, scale, disp);
  }
  void vmovdqu_sib_x(Reg base, Reg idx, uint8_t scale, Xmm src, int32_t disp = 0) {
    vex3(1, 2, src, 0, base, idx); b(0x7F); modrm_sib(src, base, idx, scale, disp);
  }
  void vmovdqu_x_x(Xmm dst, Xmm src) {
    vex3(1, 2, dst, 0, src); b(0x6F); modrm_rr(dst, src);
  }
  void vmovd_x_r32(Xmm dst, Reg src) {
    vex3(1, 1, dst, 0, src); b(0x6E); modrm_rr(dst, src);
  }
  void vpbroadcastd_x_x(Xmm dst, Xmm src) {
    vex3(2, 1, dst, 0, src); b(0x58); modrm_rr(dst, src);
  }
  void vpbroadcastw_x_x(Xmm dst, Xmm src) {
    vex3(2, 1, dst, 0, src); b(0x79); modrm_rr(dst, src);
  }
  void vpbroadcastb_x_x(Xmm dst, Xmm src) {
    vex3(2, 1, dst, 0, src); b(0x78); modrm_rr(dst, src);
  }

  // Generic 3-operand VEX reg-reg-reg / reg-reg-mem
  void vex_rrr(uint8_t map, uint8_t pp, uint8_t op, Xmm dst, Xmm s1, Xmm s2) {
    vex3(map, pp, dst, s1, s2); b(op); modrm_rr(dst, s2);
  }
  void vex_rrm(uint8_t map, uint8_t pp, uint8_t op, Xmm dst, Xmm s1, Reg base, int32_t disp) {
    vex3(map, pp, dst, s1, base); b(op); modrm_disp(dst, base, disp);
  }
  void vex_rr_sib(uint8_t map, uint8_t pp, uint8_t op, Xmm dst, Xmm s1, Reg base, Reg idx, uint8_t scale, int32_t disp = 0) {
    vex3(map, pp, dst, s1, base, idx); b(op); modrm_sib(dst, base, idx, scale, disp);
  }

  void vpxor(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xEF, d, s1, s2); }
  void vpxor_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xEF, d, s1, b_, disp); }
  void vpor(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xEB, d, s1, s2); }
  void vpor_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xEB, d, s1, b_, disp); }
  void vpand(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xDB, d, s1, s2); }
  void vpand_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xDB, d, s1, b_, disp); }
  void vpandn(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xDF, d, s1, s2); }
  void vpandn_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xDF, d, s1, b_, disp); }

  void vpaddd(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xFE, d, s1, s2); }
  void vpaddd_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xFE, d, s1, b_, disp); }
  void vpaddw(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xFD, d, s1, s2); }
  void vpaddw_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xFD, d, s1, b_, disp); }
  void vpsubd(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xFA, d, s1, s2); }
  void vpsubd_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xFA, d, s1, b_, disp); }
  void vpsubw(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xF9, d, s1, s2); }
  void vpsubw_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xF9, d, s1, b_, disp); }
  void vpmulld(Xmm d, Xmm s1, Xmm s2) { vex_rrr(2, 1, 0x40, d, s1, s2); }
  void vpavgb_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xE0, d, s1, b_, disp); }
  void vpsubusb(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0xD8, d, s1, s2); }
  void vpsubusb_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0xD8, d, s1, b_, disp); }

  void vpcmpeqd(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0x76, d, s1, s2); }
  void vpcmpeqd_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0x76, d, s1, b_, disp); }
  void vpcmpeqw(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0x75, d, s1, s2); }
  void vpcmpeqw_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0x75, d, s1, b_, disp); }
  void vpcmpeqb(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0x74, d, s1, s2); }
  void vpcmpeqb_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0x74, d, s1, b_, disp); }
  void vpcmpgtd(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0x66, d, s1, s2); }
  void vpcmpgtd_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0x66, d, s1, b_, disp); }
  void vpcmpgtw(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0x65, d, s1, s2); }
  void vpcmpgtw_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0x65, d, s1, b_, disp); }
  void vpcmpgtb(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 1, 0x64, d, s1, s2); }
  void vpcmpgtb_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 1, 0x64, d, s1, b_, disp); }

  void vpshufb(Xmm d, Xmm s1, Xmm s2) { vex_rrr(2, 1, 0x00, d, s1, s2); }
  void vpshufb_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(2, 1, 0x00, d, s1, b_, disp); }
  void vpshufb_sib(Xmm d, Xmm s1, Reg b_, Reg idx, uint8_t sc, int32_t disp) {
    vex_rr_sib(2, 1, 0x00, d, s1, b_, idx, sc, disp);
  }
  void vpshufd(Xmm d, Xmm s, uint8_t imm) {
    vex3(1, 1, d, 0, s); b(0x70); modrm_rr(d, s); b(imm);
  }
  void vpblendvb(Xmm d, Xmm s1, Xmm s2, Xmm mask) {
    vex3(3, 1, d, s1, s2); b(0x4C); modrm_rr(d, s2); b(uint8_t(mask << 4));
  }

  // Immediate shifts: vvvv = dst, ModRM.reg = ext, ModRM.rm = src
  void vshift_imm(uint8_t op, uint8_t ext, Xmm d, Xmm s, uint8_t imm) {
    vex3(1, 1, ext, d, s); b(op); modrm_rr(ext, s); b(imm);
  }
  void vpslld(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x72, 6, d, s, imm); }
  void vpsrld(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x72, 2, d, s, imm); }
  void vpsrad(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x72, 4, d, s, imm); }
  void vpsllw(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x71, 6, d, s, imm); }
  void vpsrlw(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x71, 2, d, s, imm); }
  void vpsraw(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x71, 4, d, s, imm); }
  void vpslldq(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x73, 7, d, s, imm); }
  void vpsrldq(Xmm d, Xmm s, uint8_t imm) { vshift_imm(0x73, 3, d, s, imm); }

  void vpsllvd(Xmm d, Xmm s1, Xmm s2) { vex_rrr(2, 1, 0x47, d, s1, s2); }
  void vpsrlvd(Xmm d, Xmm s1, Xmm s2) { vex_rrr(2, 1, 0x45, d, s1, s2); }
  void vpsravd(Xmm d, Xmm s1, Xmm s2) { vex_rrr(2, 1, 0x46, d, s1, s2); }
  void vpminud_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(2, 1, 0x3B, d, s1, b_, disp); }

  void vpmovmskb(Reg dst, Xmm src) { vex3(1, 1, dst, 0, src); b(0xD7); modrm_rr(dst, src); }

  void vaddps_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 0, 0x58, d, s1, b_, disp); }
  void vsubps_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 0, 0x5C, d, s1, b_, disp); }
  void vmulps_m(Xmm d, Xmm s1, Reg b_, int32_t disp) { vex_rrm(1, 0, 0x59, d, s1, b_, disp); }
  void vaddps(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 0, 0x58, d, s1, s2); }
  void vsubps(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 0, 0x5C, d, s1, s2); }
  void vmulps(Xmm d, Xmm s1, Xmm s2) { vex_rrr(1, 0, 0x59, d, s1, s2); }
  void vcvtdq2ps_m(Xmm d, Reg b_, int32_t disp) { vex3(1, 0, d, 0, b_); b(0x5B); modrm_disp(d, b_, disp); }

  // ---- Convenience helpers ----
  void load_gpr(Xmm dst, unsigned r) { vmovdqu_x_m(dst, RSI, int32_t(r * 16)); }
  void store_gpr(unsigned r, Xmm src) { vmovdqu_m_x(RSI, int32_t(r * 16), src); }
  void splat_imm32(Xmm dst, uint32_t imm) {
    if (imm == 0) {
      vpxor(dst, dst, dst);
    } else {
      mov_r32_imm32(RAX, imm);
      vmovd_x_r32(dst, RAX);
      vpbroadcastd_x_x(dst, dst);
    }
  }
  void splat_imm16(Xmm dst, uint16_t imm) {
    splat_imm32(dst, uint32_t(imm) | (uint32_t(imm) << 16));
  }
  void splat_imm8(Xmm dst, uint8_t imm) {
    const uint32_t w = uint32_t(imm) * 0x01010101u;
    splat_imm32(dst, w);
  }
  void store_preferred_zero(unsigned rt, Reg val) {
    vmovd_x_r32(XMM0, val);  // vmovd zero-extends bits 32..127!
    store_gpr(rt, XMM0);
  }
  void emit_link(unsigned rt, uint32_t next_pc) {
    mov_r32_imm32(RDX, next_pc);
    vmovd_x_r32(XMM0, RDX);
    store_gpr(rt, XMM0);
  }

  // Chain-jump tail: eax holds next_pc (masked to 0x3FFFC).
  void emit_chain_exit() {
    dec_r32(R14);
    jle_rel8(12);                           // skip to jmp r15
    mov_r64_sib(RCX, R12, RAX, 1, 0);       // 4 bytes: 49 8B 0C 44  (rcx = blocks[eax >> 2])
    test_r64_r64(RCX, RCX);                 // 3 bytes: 48 85 C9
    jz_rel8(2);                             // 2 bytes: 74 02
    jmp_reg(RCX);                           // 2 bytes: FF E1
    // Wait: count exact bytes above: 4 + 3 + 2 + 2 = 11 bytes!
    // Let's patch jle_rel8 cleanly instead of hardcoding offsets!
  }

  void emit_chain_exit_safe() {
    dec_r32(R14);
    b(0x7E);  // jle rel8
    const size_t jle_patch = pos;
    b(0);
    mov_r64_sib(RCX, R12, RAX, 1, 0);       // rcx = [r12 + rax*2]
    test_r64_r64(RCX, RCX);
    b(0x74);  // jz rel8
    const size_t jz_patch = pos;
    b(0);
    jmp_reg(RCX);
    const size_t exit_pos = pos;
    if (!overflow) {
      buf[jle_patch] = uint8_t(exit_pos - (jle_patch + 1));
      buf[jz_patch] = uint8_t(exit_pos - (jz_patch + 1));
    }
    jmp_reg(R15);
  }

  void emit_fallback_call(uint32_t cur_pc, bool terminate_block) {
    mov_r64_r64(RCX, RBX);
    mov_r32_imm32(RDX, cur_pc);
    mov_r64_imm64(RAX, reinterpret_cast<uint64_t>(&detail::jit_step_fallback));
    call_reg(RAX);
    if (terminate_block) {
      // Return next_pc (or 0xFFFFFFFF) directly to C++ so stop/DMA invalidation is handled.
      jmp_reg(R15);
    }
  }
};

#define QOFF(field) int32_t(offsetof(QuadShuffleTables, field))

}  // namespace

SpuJit::SpuJit() {
  code_ = static_cast<uint8_t*>(
      VirtualAlloc(nullptr, kCodeBufSize, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
  if (!code_) return;
  code_cap_ = kCodeBufSize;
  blocks_ = new BlockFn[kBlockCount]();
  build_trampoline();
}

SpuJit::~SpuJit() {
  delete[] blocks_;
  if (code_) VirtualFree(code_, 0, MEM_RELEASE);
}

void SpuJit::build_trampoline() {
  Emit e{code_, 0, code_cap_};
  // enter(rcx=self, rdx=gpr, r8=ls, r9=block)
  enter_ = reinterpret_cast<EnterFn>(code_);
  e.push(RBX);
  e.push(RSI);
  e.push(RDI);
  e.push(R12);
  e.push(R13);
  e.push(R14);
  e.push(R15);
  e.sub_r64_imm8(RSP, 48);  // 32B shadow space + 16B saved xmm15; rsp is 16-byte aligned!
  e.vmovdqu_m_x(RSP, 32, XMM15);
  e.mov_r64_r64(RBX, RCX);
  e.mov_r64_r64(RSI, RDX);
  e.mov_r64_r64(RDI, R8);
  e.mov_r64_imm64(R12, reinterpret_cast<uint64_t>(blocks_));
  e.mov_r64_imm64(R13, reinterpret_cast<uint64_t>(&detail::g_qshuf));
  e.mov_r32_imm32(R14, kSliceBudget);
  e.vmovdqu_x_m(XMM15, R13, QOFF(bswap32_mask));
  // Patch R15 with epilogue address once emitted:
  const size_t r15_mov_pos = e.pos;
  e.mov_r64_imm64(R15, 0);
  e.jmp_reg(R9);

  // Align epilogue to 16 bytes
  while (e.pos & 15) e.b(0x90);
  epilogue_ = code_ + e.pos;
  const uint64_t epi_addr = reinterpret_cast<uint64_t>(epilogue_);
  std::memcpy(code_ + r15_mov_pos + 2, &epi_addr, 8);

  e.vmovdqu_x_m(XMM15, RSP, 32);
  e.add_r64_imm8(RSP, 48);
  e.pop(R15);
  e.pop(R14);
  e.pop(R13);
  e.pop(R12);
  e.pop(RDI);
  e.pop(RSI);
  e.pop(RBX);
  e.ret();

  while (e.pos & 15) e.b(0x90);
  code_pos_ = e.pos;
  code_reset_pos_ = e.pos;
}

void SpuJit::invalidate_all() {
  if (!code_ || !code_pages_) return;
  std::memset(blocks_, 0, kBlockCount * sizeof(BlockFn));
  code_pos_ = code_reset_pos_;
  code_pages_ = 0;
}

void SpuJit::invalidate_ls(uint32_t addr, uint32_t len) {
  if (!code_pages_ || len == 0) return;
  const uint32_t p0 = (addr & kLsMask) >> 12;
  const uint32_t end = std::min<uint32_t>(kLsSize - 1, (addr & kLsMask) + len - 1);
  const uint32_t p1 = end >> 12;
  const uint32_t count = p1 - p0 + 1;
  const uint64_t mask = (count >= 64) ? ~0ull : (((1ull << count) - 1ull) << p0);
  if (code_pages_ & mask) invalidate_all();
}

uint32_t SpuJit::run(SpuCore& core) {
  _mm_setcsr(_mm_getcsr() | 0x8040);  // FTZ | DAZ
  uint32_t local_pc = core.pc & kLsMask & ~3u;
  for (;;) {
    BlockFn fn = blocks_[local_pc >> 2];
    if (!fn) fn = compile_block(core, local_pc);
    const uint32_t next = enter_(&core, core.gpr, core.ls(), fn);
    if (next == 0xFFFFFFFFu) {
      return core.jit_stop_code_;
    }
    if (next == 0xFFFFFFFEu) {
      std::exception_ptr ex = core.jit_exception_;
      core.jit_exception_ = nullptr;
      if (ex) std::rethrow_exception(ex);
      return 0;
    }
    local_pc = next & kLsMask & ~3u;
    core.pc = local_pc;
    if (core.exit_requested()) return 0;
  }
}

SpuJit::BlockFn SpuJit::compile_block(SpuCore& core, uint32_t start_pc) {
  // Leave generous headroom so a single block never overflows mid-emit.
  if (code_pos_ + 16384 > code_cap_) invalidate_all();
  while (code_pos_ & 15) code_[code_pos_++] = 0x90;

  uint8_t* const block_start = code_ + code_pos_;
  Emit e{code_, code_pos_, code_cap_};

  const uint8_t* const ls = core.ls();
  uint32_t cur_pc = start_pc & kLsMask & ~3u;
  bool ended = false;

  for (int count = 0; count < kMaxBlockInsns && !ended; ++count) {
    code_pages_ |= (1ull << (cur_pc >> 12));
    uint32_t raw_be;
    std::memcpy(&raw_be, ls + cur_pc, 4);
    const uint32_t raw = bswap_u32(raw_be);
    const uint32_t next = (cur_pc + 4) & kLsMask;
    const unsigned rt = raw & 0x7F;
    const unsigned ra = (raw >> 7) & 0x7F;
    const unsigned rb = (raw >> 14) & 0x7F;
    const uint32_t i16 = (raw >> 7) & 0xFFFF;
    const int32_t i10 = sext((raw >> 14) & 0x3FF, 10);
    const uint32_t i7 = (raw >> 14) & 0x7F;
    const auto rel = [&] { return (cur_pc + (uint32_t(sext(i16, 16)) << 2)) & kLsMask & ~3u; };
    const auto abs = [&] { return (uint32_t(sext(i16, 16)) << 2) & kLsMask & ~3u; };
    const Op op = detail::g_table.op[raw >> 21];

    switch (op) {
      // ---- NOP / hints ----
      case Op::Nop: case Op::Hbra: case Op::Hbrr: case Op::Mtspr: case Op::Fscrwr:
        break;
      case Op::Mfspr: case Op::Fscrrd:
        e.vpxor(XMM0, XMM0, XMM0);
        e.store_gpr(rt, XMM0);
        break;

      // ---- RRR ----
      case Op::Selb: {
        const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
        e.load_gpr(XMM2, rc);
        e.vpand_m(XMM0, XMM2, RSI, int32_t(rb * 16));
        e.vpandn_m(XMM1, XMM2, RSI, int32_t(ra * 16));
        e.vpor(XMM0, XMM0, XMM1);
        e.store_gpr(t, XMM0);
        break;
      }
      case Op::Shufb: {
        const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
        e.load_gpr(XMM2, rc);
        e.vpand_m(XMM3, XMM2, R13, QOFF(shufb_8f));
        e.vpxor_m(XMM3, XMM3, R13, QOFF(shufb_03));
        e.load_gpr(XMM0, ra);
        e.vpshufb(XMM0, XMM0, XMM3);
        e.load_gpr(XMM1, rb);
        e.vpshufb(XMM1, XMM1, XMM3);
        e.vpsllw(XMM4, XMM2, 3);
        e.vpblendvb(XMM0, XMM0, XMM1, XMM4);
        e.vpand_m(XMM4, XMM2, R13, QOFF(shufb_e0));
        e.vpcmpeqb_m(XMM1, XMM4, R13, QOFF(shufb_c0));
        e.vpcmpeqb_m(XMM4, XMM4, R13, QOFF(shufb_e0));
        e.vpand_m(XMM4, XMM4, R13, QOFF(shufb_80));
        e.vpor(XMM1, XMM1, XMM4);
        e.vpor(XMM0, XMM0, XMM1);
        e.store_gpr(t, XMM0);
        break;
      }
      case Op::Mpya: {
        const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
        e.load_gpr(XMM0, ra);
        e.vpslld(XMM0, XMM0, 16);
        e.vpsrad(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpslld(XMM1, XMM1, 16);
        e.vpsrad(XMM1, XMM1, 16);
        e.vpmulld(XMM0, XMM0, XMM1);
        e.vpaddd_m(XMM0, XMM0, RSI, int32_t(rc * 16));
        e.store_gpr(t, XMM0);
        break;
      }
      case Op::Fma: {
        const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
        e.load_gpr(XMM0, ra);
        e.vmulps_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.vaddps_m(XMM0, XMM0, RSI, int32_t(rc * 16));
        e.store_gpr(t, XMM0);
        break;
      }
      case Op::Fms: {
        const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
        e.load_gpr(XMM0, ra);
        e.vmulps_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.vsubps_m(XMM0, XMM0, RSI, int32_t(rc * 16));
        e.store_gpr(t, XMM0);
        break;
      }
      case Op::Fnms: {
        const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
        e.load_gpr(XMM0, ra);
        e.vmulps_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.load_gpr(XMM1, rc);
        e.vsubps(XMM0, XMM1, XMM0);
        e.store_gpr(t, XMM0);
        break;
      }

      // ---- Constants ----
      case Op::Ila:
        e.splat_imm32(XMM0, (raw >> 7) & 0x3FFFF);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Il:
        e.splat_imm32(XMM0, uint32_t(sext(i16, 16)));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ilh:
        e.splat_imm32(XMM0, (i16 << 16) | i16);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ilhu:
        e.splat_imm32(XMM0, i16 << 16);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Iohl:
        e.splat_imm32(XMM1, i16);
        e.vpor_m(XMM0, XMM1, RSI, int32_t(rt * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Fsmbi: {
        e.mov_r64_m(RAX, R13, QOFF(fsmb8) + int32_t(((i16 >> 8) & 0xFF) * 8));
        e.mov_m_r64(RSI, int32_t(rt * 16), RAX);
        e.mov_r64_m(RAX, R13, QOFF(fsmb8) + int32_t((i16 & 0xFF) * 8));
        e.mov_m_r64(RSI, int32_t(rt * 16 + 8), RAX);
        break;
      }

      // ---- Memory ----
      case Op::Lqd: {
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        const int32_t off = int32_t(uint32_t(i10) << 4);
        if (off != 0) e.add_r32_imm32(RAX, off);
        e.and_r32_imm32(RAX, kLsMask & ~15u);
        e.vmovdqu_x_sib(XMM0, RDI, RAX, 0);
        e.vpshufb(XMM0, XMM0, XMM15);
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Stqd: {
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        const int32_t off = int32_t(uint32_t(i10) << 4);
        if (off != 0) e.add_r32_imm32(RAX, off);
        e.and_r32_imm32(RAX, kLsMask & ~15u);
        e.load_gpr(XMM0, rt);
        e.vpshufb(XMM0, XMM0, XMM15);
        e.vmovdqu_sib_x(RDI, RAX, 0, XMM0);
        break;
      }
      case Op::Lqx: {
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.mov_r32_m(RCX, RSI, int32_t(rb * 16));
        e.add_r32_r32(RAX, RCX);
        e.and_r32_imm32(RAX, kLsMask & ~15u);
        e.vmovdqu_x_sib(XMM0, RDI, RAX, 0);
        e.vpshufb(XMM0, XMM0, XMM15);
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Stqx: {
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.mov_r32_m(RCX, RSI, int32_t(rb * 16));
        e.add_r32_r32(RAX, RCX);
        e.and_r32_imm32(RAX, kLsMask & ~15u);
        e.load_gpr(XMM0, rt);
        e.vpshufb(XMM0, XMM0, XMM15);
        e.vmovdqu_sib_x(RDI, RAX, 0, XMM0);
        break;
      }
      case Op::Lqa: {
        const uint32_t a = abs() & ~15u;
        e.vmovdqu_x_m(XMM0, RDI, int32_t(a));
        e.vpshufb(XMM0, XMM0, XMM15);
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Lqr: {
        const uint32_t a = rel() & ~15u;
        e.vmovdqu_x_m(XMM0, RDI, int32_t(a));
        e.vpshufb(XMM0, XMM0, XMM15);
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Stqa: {
        const uint32_t a = abs() & ~15u;
        e.load_gpr(XMM0, rt);
        e.vpshufb(XMM0, XMM0, XMM15);
        e.vmovdqu_m_x(RDI, int32_t(a), XMM0);
        break;
      }
      case Op::Stqr: {
        const uint32_t a = rel() & ~15u;
        e.load_gpr(XMM0, rt);
        e.vpshufb(XMM0, XMM0, XMM15);
        e.vmovdqu_m_x(RDI, int32_t(a), XMM0);
        break;
      }

      // ---- RI10 Integer / Logical / Compare ----
      case Op::Ai:
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpaddd_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ahi:
        e.splat_imm16(XMM1, uint16_t(i10));
        e.vpaddw_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Sfi:
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpsubd_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Sfhi:
        e.splat_imm16(XMM1, uint16_t(i10));
        e.vpsubw_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Andi:
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpand_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Andhi:
        e.splat_imm16(XMM1, uint16_t(i10));
        e.vpand_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Andbi:
        e.splat_imm8(XMM1, uint8_t(i10));
        e.vpand_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ori:
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpor_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Orhi:
        e.splat_imm16(XMM1, uint16_t(i10));
        e.vpor_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Orbi:
        e.splat_imm8(XMM1, uint8_t(i10));
        e.vpor_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Xori:
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpxor_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Xorhi:
        e.splat_imm16(XMM1, uint16_t(i10));
        e.vpxor_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Xorbi:
        e.splat_imm8(XMM1, uint8_t(i10));
        e.vpxor_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyi:
        e.load_gpr(XMM0, ra);
        e.vpslld(XMM0, XMM0, 16);
        e.vpsrad(XMM0, XMM0, 16);
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpmulld(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyui:
        e.load_gpr(XMM0, ra);
        e.vpand_m(XMM0, XMM0, R13, QOFF(mask_ffff));
        e.splat_imm32(XMM1, uint32_t(uint16_t(i10)));
        e.vpmulld(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Cgti:
        e.load_gpr(XMM0, ra);
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpcmpgtd(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Cgthi:
        e.load_gpr(XMM0, ra);
        e.splat_imm16(XMM1, uint16_t(i10));
        e.vpcmpgtw(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Cgtbi:
        e.load_gpr(XMM0, ra);
        e.splat_imm8(XMM1, uint8_t(i10));
        e.vpcmpgtb(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Clgti:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, R13, QOFF(sign_w));
        e.splat_imm32(XMM1, uint32_t(i10) ^ 0x80000000u);
        e.vpcmpgtd(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Clgthi:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, R13, QOFF(sign_h));
        e.splat_imm16(XMM1, uint16_t(i10) ^ 0x8000u);
        e.vpcmpgtw(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Clgtbi:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, R13, QOFF(sign_b));
        e.splat_imm8(XMM1, uint8_t(i10) ^ 0x80u);
        e.vpcmpgtb(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ceqi:
        e.splat_imm32(XMM1, uint32_t(i10));
        e.vpcmpeqd_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ceqhi:
        e.splat_imm16(XMM1, uint16_t(i10));
        e.vpcmpeqw_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ceqbi:
        e.splat_imm8(XMM1, uint8_t(i10));
        e.vpcmpeqb_m(XMM0, XMM1, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;

      // ---- Integer Arithmetic (RR) ----
      case Op::A:
        e.load_gpr(XMM0, ra);
        e.vpaddd_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ah:
        e.load_gpr(XMM0, ra);
        e.vpaddw_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Sf:
        e.load_gpr(XMM0, rb);
        e.vpsubd_m(XMM0, XMM0, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Sfh:
        e.load_gpr(XMM0, rb);
        e.vpsubw_m(XMM0, XMM0, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpy:
        e.load_gpr(XMM0, ra);
        e.vpslld(XMM0, XMM0, 16);
        e.vpsrad(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpslld(XMM1, XMM1, 16);
        e.vpsrad(XMM1, XMM1, 16);
        e.vpmulld(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyu:
        e.load_gpr(XMM0, ra);
        e.vpand_m(XMM0, XMM0, R13, QOFF(mask_ffff));
        e.load_gpr(XMM1, rb);
        e.vpand_m(XMM1, XMM1, R13, QOFF(mask_ffff));
        e.vpmulld(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyh:
        e.load_gpr(XMM0, ra);
        e.vpsrld(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpand_m(XMM1, XMM1, R13, QOFF(mask_ffff));
        e.vpmulld(XMM0, XMM0, XMM1);
        e.vpslld(XMM0, XMM0, 16);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpys:
        e.load_gpr(XMM0, ra);
        e.vpslld(XMM0, XMM0, 16);
        e.vpsrad(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpslld(XMM1, XMM1, 16);
        e.vpsrad(XMM1, XMM1, 16);
        e.vpmulld(XMM0, XMM0, XMM1);
        e.vpsrad(XMM0, XMM0, 16);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyhh:
        e.load_gpr(XMM0, ra);
        e.vpsrad(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpsrad(XMM1, XMM1, 16);
        e.vpmulld(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyhha:
        e.load_gpr(XMM0, ra);
        e.vpsrad(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpsrad(XMM1, XMM1, 16);
        e.vpmulld(XMM0, XMM0, XMM1);
        e.vpaddd_m(XMM0, XMM0, RSI, int32_t(rt * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyhhu:
        e.load_gpr(XMM0, ra);
        e.vpsrld(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpsrld(XMM1, XMM1, 16);
        e.vpmulld(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Mpyhhau:
        e.load_gpr(XMM0, ra);
        e.vpsrld(XMM0, XMM0, 16);
        e.load_gpr(XMM1, rb);
        e.vpsrld(XMM1, XMM1, 16);
        e.vpmulld(XMM0, XMM0, XMM1);
        e.vpaddd_m(XMM0, XMM0, RSI, int32_t(rt * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Avgb:
        e.load_gpr(XMM0, ra);
        e.vpavgb_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Absdb:
        e.load_gpr(XMM0, ra);
        e.load_gpr(XMM1, rb);
        e.vpsubusb(XMM2, XMM0, XMM1);
        e.vpsubusb(XMM1, XMM1, XMM0);
        e.vpor(XMM0, XMM2, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Xsbh:
        e.load_gpr(XMM0, ra);
        e.vpsllw(XMM0, XMM0, 8);
        e.vpsraw(XMM0, XMM0, 8);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Xshw:
        e.load_gpr(XMM0, ra);
        e.vpslld(XMM0, XMM0, 16);
        e.vpsrad(XMM0, XMM0, 16);
        e.store_gpr(rt, XMM0);
        break;

      // ---- Logical (RR) ----
      case Op::And:
        e.load_gpr(XMM0, ra);
        e.vpand_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Andc:
        e.load_gpr(XMM0, rb);
        e.vpandn_m(XMM0, XMM0, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Or:
        e.load_gpr(XMM0, ra);
        e.vpor_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Orc:
        e.load_gpr(XMM0, rb);
        e.vpxor_m(XMM0, XMM0, R13, QOFF(all_ones));
        e.vpor_m(XMM0, XMM0, RSI, int32_t(ra * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Xor:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Nand:
        e.load_gpr(XMM0, ra);
        e.vpand_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.vpxor_m(XMM0, XMM0, R13, QOFF(all_ones));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Nor:
        e.load_gpr(XMM0, ra);
        e.vpor_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.vpxor_m(XMM0, XMM0, R13, QOFF(all_ones));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Eqv:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.vpxor_m(XMM0, XMM0, R13, QOFF(all_ones));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Orx:
        e.load_gpr(XMM0, ra);
        e.vpshufd(XMM1, XMM0, 0x4E);  // swap hi/lo 64
        e.vpor(XMM0, XMM0, XMM1);
        e.vpshufd(XMM1, XMM0, 0x01);  // lane 1 -> lane 0
        e.vpor(XMM0, XMM0, XMM1);
        // Zero lanes 1..3:
        e.vpxor(XMM1, XMM1, XMM1);
        e.store_gpr(rt, XMM1);
        e.vex3(1, 1, XMM0, 0, RSI); e.b(0x7E); e.modrm_disp(XMM0, RSI, int32_t(rt * 16));  // vmovd [rsi+rt*16], xmm0
        break;

      // ---- Compares (RR) ----
      case Op::Ceq:
        e.load_gpr(XMM0, ra);
        e.vpcmpeqd_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ceqh:
        e.load_gpr(XMM0, ra);
        e.vpcmpeqw_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Ceqb:
        e.load_gpr(XMM0, ra);
        e.vpcmpeqb_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Cgt:
        e.load_gpr(XMM0, ra);
        e.vpcmpgtd_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Cgth:
        e.load_gpr(XMM0, ra);
        e.vpcmpgtw_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Cgtb:
        e.load_gpr(XMM0, ra);
        e.vpcmpgtb_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Clgt:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, R13, QOFF(sign_w));
        e.load_gpr(XMM1, rb);
        e.vpxor_m(XMM1, XMM1, R13, QOFF(sign_w));
        e.vpcmpgtd(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Clgth:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, R13, QOFF(sign_h));
        e.load_gpr(XMM1, rb);
        e.vpxor_m(XMM1, XMM1, R13, QOFF(sign_h));
        e.vpcmpgtw(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Clgtb:
        e.load_gpr(XMM0, ra);
        e.vpxor_m(XMM0, XMM0, R13, QOFF(sign_b));
        e.load_gpr(XMM1, rb);
        e.vpxor_m(XMM1, XMM1, R13, QOFF(sign_b));
        e.vpcmpgtb(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;

      // ---- Element shifts & rotates ----
      case Op::Shlhi: {
        const unsigned s = i7 & 0x1F;
        if (s > 15) {
          e.vpxor(XMM0, XMM0, XMM0);
        } else {
          e.load_gpr(XMM0, ra);
          if (s) e.vpsllw(XMM0, XMM0, uint8_t(s));
        }
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Shli: {
        const unsigned s = i7 & 0x3F;
        if (s > 31) {
          e.vpxor(XMM0, XMM0, XMM0);
        } else {
          e.load_gpr(XMM0, ra);
          if (s) e.vpslld(XMM0, XMM0, uint8_t(s));
        }
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Shl:
        e.load_gpr(XMM0, ra);
        e.load_gpr(XMM1, rb);
        e.vpand_m(XMM1, XMM1, R13, QOFF(mask_3f));
        e.vpsllvd(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Roti: {
        const unsigned s = i7 & 0x1F;
        e.load_gpr(XMM0, ra);
        if (s) {
          e.vpslld(XMM1, XMM0, uint8_t(s));
          e.vpsrld(XMM0, XMM0, uint8_t(32 - s));
          e.vpor(XMM0, XMM1, XMM0);
        }
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rot:
        e.load_gpr(XMM0, ra);
        e.load_gpr(XMM1, rb);
        e.vpand_m(XMM1, XMM1, R13, QOFF(mask_1f));
        e.vpsllvd(XMM2, XMM0, XMM1);
        e.vmovdqu_x_m(XMM3, R13, QOFF(const_32));
        e.vpsubd(XMM3, XMM3, XMM1);
        e.vpsrlvd(XMM0, XMM0, XMM3);
        e.vpor(XMM0, XMM2, XMM0);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Rothmi: {
        const unsigned s = (0u - i7) & 0x1F;
        if (s > 15) {
          e.vpxor(XMM0, XMM0, XMM0);
        } else {
          e.load_gpr(XMM0, ra);
          if (s) e.vpsrlw(XMM0, XMM0, uint8_t(s));
        }
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rotmi: {
        const unsigned s = (0u - i7) & 0x3F;
        if (s > 31) {
          e.vpxor(XMM0, XMM0, XMM0);
        } else {
          e.load_gpr(XMM0, ra);
          if (s) e.vpsrld(XMM0, XMM0, uint8_t(s));
        }
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rotm:
        e.vpxor(XMM1, XMM1, XMM1);
        e.vpsubd_m(XMM1, XMM1, RSI, int32_t(rb * 16));
        e.vpand_m(XMM1, XMM1, R13, QOFF(mask_3f));
        e.load_gpr(XMM0, ra);
        e.vpsrlvd(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Rotmahi: {
        const unsigned s = std::min<unsigned>((0u - i7) & 0x1F, 15);
        e.load_gpr(XMM0, ra);
        if (s) e.vpsraw(XMM0, XMM0, uint8_t(s));
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rotmai: {
        const unsigned s = std::min<unsigned>((0u - i7) & 0x3F, 31);
        e.load_gpr(XMM0, ra);
        if (s) e.vpsrad(XMM0, XMM0, uint8_t(s));
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rotma:
        e.vpxor(XMM1, XMM1, XMM1);
        e.vpsubd_m(XMM1, XMM1, RSI, int32_t(rb * 16));
        e.vpand_m(XMM1, XMM1, R13, QOFF(mask_3f));
        e.vpminud_m(XMM1, XMM1, R13, QOFF(const_31));
        e.load_gpr(XMM0, ra);
        e.vpsravd(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;

      // ---- Quadword shifts & rotates ----
      case Op::Rotqbyi:
        e.load_gpr(XMM0, ra);
        e.vpshufb_m(XMM0, XMM0, R13, QOFF(rotqby) + int32_t((i7 & 0xF) * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Rotqby:
        e.mov_r32_m(RAX, RSI, int32_t(rb * 16));
        e.and_r32_imm32(RAX, 0xF);
        e.shl_r32_imm8(RAX, 4);
        e.load_gpr(XMM0, ra);
        e.vpshufb_sib(XMM0, XMM0, R13, RAX, 0, QOFF(rotqby));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Rotqbybi:
        e.mov_r32_m(RAX, RSI, int32_t(rb * 16));
        e.shr_r32_imm8(RAX, 3);
        e.and_r32_imm32(RAX, 0xF);
        e.shl_r32_imm8(RAX, 4);
        e.load_gpr(XMM0, ra);
        e.vpshufb_sib(XMM0, XMM0, R13, RAX, 0, QOFF(rotqby));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Shlqbyi: {
        const unsigned n = std::min<unsigned>(i7 & 0x1F, 16);
        e.load_gpr(XMM0, ra);
        e.vpshufb_m(XMM0, XMM0, R13, QOFF(shlqby) + int32_t(n * 16));
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Shlqby:
        e.mov_r32_m(RAX, RSI, int32_t(rb * 16));
        e.and_r32_imm32(RAX, 0x1F);
        e.mov_r32_imm32(RCX, 16);
        e.cmp_r32_imm8(RAX, 16);
        e.cmova_r32_r32(RAX, RCX);
        e.shl_r32_imm8(RAX, 4);
        e.load_gpr(XMM0, ra);
        e.vpshufb_sib(XMM0, XMM0, R13, RAX, 0, QOFF(shlqby));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Shlqbybi:
        e.mov_r32_m(RAX, RSI, int32_t(rb * 16));
        e.shr_r32_imm8(RAX, 3);
        e.and_r32_imm32(RAX, 0x1F);
        e.mov_r32_imm32(RCX, 16);
        e.cmp_r32_imm8(RAX, 16);
        e.cmova_r32_r32(RAX, RCX);
        e.shl_r32_imm8(RAX, 4);
        e.load_gpr(XMM0, ra);
        e.vpshufb_sib(XMM0, XMM0, R13, RAX, 0, QOFF(shlqby));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Rotqmbyi: {
        const unsigned n = std::min<unsigned>((0u - i7) & 0x1F, 16);
        e.load_gpr(XMM0, ra);
        e.vpshufb_m(XMM0, XMM0, R13, QOFF(rotqmby) + int32_t(n * 16));
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rotqmby:
        e.mov_r32_m(RAX, RSI, int32_t(rb * 16));
        e.neg_r32(RAX);
        e.and_r32_imm32(RAX, 0x1F);
        e.mov_r32_imm32(RCX, 16);
        e.cmp_r32_imm8(RAX, 16);
        e.cmova_r32_r32(RAX, RCX);
        e.shl_r32_imm8(RAX, 4);
        e.load_gpr(XMM0, ra);
        e.vpshufb_sib(XMM0, XMM0, R13, RAX, 0, QOFF(rotqmby));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Rotqmbybi:
        e.mov_r32_m(RAX, RSI, int32_t(rb * 16));
        e.shr_r32_imm8(RAX, 3);
        e.neg_r32(RAX);
        e.and_r32_imm32(RAX, 0x1F);
        e.mov_r32_imm32(RCX, 16);
        e.cmp_r32_imm8(RAX, 16);
        e.cmova_r32_r32(RAX, RCX);
        e.shl_r32_imm8(RAX, 4);
        e.load_gpr(XMM0, ra);
        e.vpshufb_sib(XMM0, XMM0, R13, RAX, 0, QOFF(rotqmby));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Shlqbii: {
        const int s = int(i7 & 7);
        e.load_gpr(XMM0, ra);
        if (s) {
          e.vpslld(XMM1, XMM0, uint8_t(s));
          e.vpsrldq(XMM2, XMM0, 4);
          e.vpsrld(XMM2, XMM2, uint8_t(32 - s));
          e.vpor(XMM0, XMM1, XMM2);
        }
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rotqbii: {
        const int s = int(i7 & 7);
        e.load_gpr(XMM0, ra);
        if (s) {
          e.vpslld(XMM1, XMM0, uint8_t(s));
          e.vpshufd(XMM2, XMM0, 0x39);  // _MM_SHUFFLE(0, 3, 2, 1)
          e.vpsrld(XMM2, XMM2, uint8_t(32 - s));
          e.vpor(XMM0, XMM1, XMM2);
        }
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Rotqmbii: {
        const int s = int((0u - i7) & 7);
        e.load_gpr(XMM0, ra);
        if (s) {
          e.vpsrld(XMM1, XMM0, uint8_t(s));
          e.vpslldq(XMM2, XMM0, 4);
          e.vpslld(XMM2, XMM2, uint8_t(32 - s));
          e.vpor(XMM0, XMM1, XMM2);
        }
        e.store_gpr(rt, XMM0);
        break;
      }

      // ---- Insertion controls & masks ----
      case Op::Cbd: case Op::Chd: case Op::Cwd: case Op::Cdd: {
        const int si = int(op) - int(Op::Cbd);
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        const int32_t off = sext(i7, 7);
        if (off != 0) e.add_r32_imm32(RAX, off);
        e.and_r32_imm32(RAX, 15);
        e.shl_r32_imm8(RAX, 4);
        e.vmovdqu_x_sib(XMM0, R13, RAX, 0, QOFF(insert) + si * 256);
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Cbx: case Op::Chx: case Op::Cwx: case Op::Cdx: {
        const int si = int(op) - int(Op::Cbx);
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.mov_r32_m(RCX, RSI, int32_t(rb * 16));
        e.add_r32_r32(RAX, RCX);
        e.and_r32_imm32(RAX, 15);
        e.shl_r32_imm8(RAX, 4);
        e.vmovdqu_x_sib(XMM0, R13, RAX, 0, QOFF(insert) + si * 256);
        e.store_gpr(rt, XMM0);
        break;
      }
      case Op::Fsm:
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.and_r32_imm32(RAX, 0xF);
        e.shl_r32_imm8(RAX, 4);
        e.vmovdqu_x_sib(XMM0, R13, RAX, 0, QOFF(fsm));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Fsmh:
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.and_r32_imm32(RAX, 0xFF);
        e.shl_r32_imm8(RAX, 4);
        e.vmovdqu_x_sib(XMM0, R13, RAX, 0, QOFF(fsmh));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Fsmb:
        e.mov_r32_m(RCX, RSI, int32_t(ra * 16));
        e.mov_r32_r32(RAX, RCX);
        e.shr_r32_imm8(RAX, 8);
        e.and_r32_imm32(RAX, 0xFF);
        e.mov_r64_sib(RDX, R13, RAX, 3, QOFF(fsmb8));
        e.mov_m_r64(RSI, int32_t(rt * 16), RDX);
        e.and_r32_imm32(RCX, 0xFF);
        e.mov_r64_sib(RDX, R13, RCX, 3, QOFF(fsmb8));
        e.mov_m_r64(RSI, int32_t(rt * 16 + 8), RDX);
        break;
      case Op::Gbh:
        e.load_gpr(XMM0, ra);
        e.vpshufb_m(XMM0, XMM0, R13, QOFF(gbh_shuf));
        e.vpsllw(XMM0, XMM0, 7);
        e.vpmovmskb(RAX, XMM0);
        e.store_preferred_zero(rt, RAX);
        break;
      case Op::Gbb:
        e.load_gpr(XMM0, ra);
        e.vpshufb_m(XMM0, XMM0, R13, QOFF(gbb_shuf));
        e.vpsllw(XMM0, XMM0, 7);
        e.vpmovmskb(RAX, XMM0);
        e.store_preferred_zero(rt, RAX);
        break;

      // ---- Single-precision float ----
      case Op::Fa:
        e.load_gpr(XMM0, ra);
        e.vaddps_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Fs:
        e.load_gpr(XMM0, ra);
        e.vsubps_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Fm:
        e.load_gpr(XMM0, ra);
        e.vmulps_m(XMM0, XMM0, RSI, int32_t(rb * 16));
        e.store_gpr(rt, XMM0);
        break;
      case Op::Fi:
        e.load_gpr(XMM0, rb);
        e.store_gpr(rt, XMM0);
        break;
      case Op::Csflt: {
        const uint32_t i8 = (raw >> 14) & 0xFF;
        e.vcvtdq2ps_m(XMM0, RSI, int32_t(ra * 16));
        e.mov_r32_m(RAX, R13, QOFF(scale_155) + int32_t(i8 * 4));
        e.vmovd_x_r32(XMM1, RAX);
        e.vpbroadcastd_x_x(XMM1, XMM1);
        e.vmulps(XMM0, XMM0, XMM1);
        e.store_gpr(rt, XMM0);
        break;
      }

      // ---- Branches (terminate basic block) ----
      case Op::Br:
        e.mov_r32_imm32(RAX, rel());
        e.emit_chain_exit_safe();
        ended = true;
        break;
      case Op::Bra:
        e.mov_r32_imm32(RAX, abs());
        e.emit_chain_exit_safe();
        ended = true;
        break;
      case Op::Brsl:
        e.emit_link(rt, next);
        e.mov_r32_imm32(RAX, rel());
        e.emit_chain_exit_safe();
        ended = true;
        break;
      case Op::Brasl:
        e.emit_link(rt, next);
        e.mov_r32_imm32(RAX, abs());
        e.emit_chain_exit_safe();
        ended = true;
        break;
      case Op::Brz:
      case Op::Brnz: {
        const uint32_t t = rel();
        e.mov_r32_m(RDX, RSI, int32_t(rt * 16));
        e.test_r32_r32(RDX, RDX);
        e.mov_r32_imm32(RAX, t);
        e.mov_r32_imm32(RCX, next);
        if (op == Op::Brz) e.cmovne_r32_r32(RAX, RCX);
        else e.cmove_r32_r32(RAX, RCX);
        e.emit_chain_exit_safe();
        ended = true;
        break;
      }
      case Op::Brhz:
      case Op::Brhnz: {
        const uint32_t t = rel();
        e.movzx_r32_m16(RDX, RSI, int32_t(rt * 16));
        e.test_r32_r32(RDX, RDX);
        e.mov_r32_imm32(RAX, t);
        e.mov_r32_imm32(RCX, next);
        if (op == Op::Brhz) e.cmovne_r32_r32(RAX, RCX);
        else e.cmove_r32_r32(RAX, RCX);
        e.emit_chain_exit_safe();
        ended = true;
        break;
      }
      case Op::Bi:
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.and_r32_imm32(RAX, kLsMask & ~3u);
        e.emit_chain_exit_safe();
        ended = true;
        break;
      case Op::Bisl:
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.and_r32_imm32(RAX, kLsMask & ~3u);
        e.emit_link(rt, next);
        e.emit_chain_exit_safe();
        ended = true;
        break;
      case Op::Biz:
      case Op::Binz:
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.and_r32_imm32(RAX, kLsMask & ~3u);
        e.mov_r32_m(RDX, RSI, int32_t(rt * 16));
        e.test_r32_r32(RDX, RDX);
        e.mov_r32_imm32(RCX, next);
        if (op == Op::Biz) e.cmovne_r32_r32(RAX, RCX);
        else e.cmove_r32_r32(RAX, RCX);
        e.emit_chain_exit_safe();
        ended = true;
        break;
      case Op::Bihz:
      case Op::Bihnz:
        e.mov_r32_m(RAX, RSI, int32_t(ra * 16));
        e.and_r32_imm32(RAX, kLsMask & ~3u);
        e.movzx_r32_m16(RDX, RSI, int32_t(rt * 16));
        e.test_r32_r32(RDX, RDX);
        e.mov_r32_imm32(RCX, next);
        if (op == Op::Bihz) e.cmovne_r32_r32(RAX, RCX);
        else e.cmove_r32_r32(RAX, RCX);
        e.emit_chain_exit_safe();
        ended = true;
        break;

      // ---- Control/channel/stop ops: run via fallback and exit block ----
      case Op::Stop:
      case Op::Wrch:
      case Op::Rdch:
      case Op::Rchcnt:
      case Op::Iret:
      case Op::Heq: case Op::Hgt: case Op::Hlgt:
      case Op::Heqi: case Op::Hgti: case Op::Hlgti:
      case Op::Invalid:
        e.emit_fallback_call(cur_pc, true);
        ended = true;
        break;

      // ---- All other rare ALU ops: inline fallback call, continue block ----
      default:
        e.emit_fallback_call(cur_pc, false);
        break;
    }

    cur_pc = next;
  }

  if (!ended) {
    e.mov_r32_imm32(RAX, cur_pc);
    e.emit_chain_exit_safe();
  }

  if (e.overflow) {
    invalidate_all();
    return compile_block(core, start_pc);
  }

  code_pos_ = e.pos;
  blocks_[start_pc >> 2] = block_start;
  return block_start;
}

#endif  // SPU_JIT_SUPPORTED

}  // namespace spu
