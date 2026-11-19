#pragma once
// Shared SPU decode tables and SIMD constant tables used by both the
// interpreter (spu_interp.cpp) and the x86-64 AVX2 JIT (spu_jit.cpp).

#include "spu_interp.h"

#include <immintrin.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace spu::detail {

enum class Op : uint8_t {
  Invalid,
  // RRR
  Selb, Shufb, Mpya, Fnms, Fma, Fms,
  // RI18
  Ila, Hbra, Hbrr,
  // RI10
  Lqd, Stqd, Ai, Ahi, Sfi, Sfhi, Andi, Andhi, Andbi, Ori, Orhi, Orbi, Xori, Xorhi, Xorbi, Mpyi, Mpyui,
  Cgti, Cgthi, Cgtbi, Clgti, Clgthi, Clgtbi, Ceqi, Ceqhi, Ceqbi, Hgti, Hlgti, Heqi,
  // RI16
  Br, Bra, Brsl, Brasl, Brnz, Brz, Brhnz, Brhz, Lqa, Lqr, Stqa, Stqr, Il, Ilh, Ilhu, Iohl, Fsmbi,
  // RI8
  Cflts, Cfltu, Csflt, Cuflt,
  // RR / RI7
  Stop, Nop, Mfspr, Mtspr, Rdch, Rchcnt, Wrch, Bi, Bisl, Iret, Biz, Binz, Bihz, Bihnz, Lqx, Stqx, Heq, Hgt, Hlgt,
  A, Ah, Sf, Sfh, Cg, Bg, Addx, Sfx, Cgx, Bgx, Mpy, Mpyu, Mpyh, Mpys, Mpyhh, Mpyhha, Mpyhhu, Mpyhhau,
  Clz, Cntb, Fsm, Fsmh, Fsmb, Gb, Gbh, Gbb, Avgb, Absdb, Sumb, Xsbh, Xshw, Xswd,
  And, Andc, Or, Orc, Xor, Nand, Nor, Eqv, Orx,
  Ceq, Ceqh, Ceqb, Cgt, Cgth, Cgtb, Clgt, Clgth, Clgtb,
  Shlh, Shlhi, Shl, Shli, Roth, Rothi, Rot, Roti, Rothm, Rothmi, Rotm, Rotmi, Rotmah, Rotmahi, Rotma, Rotmai,
  Shlqbi, Shlqbii, Shlqby, Shlqbyi, Shlqbybi, Rotqbi, Rotqbii, Rotqby, Rotqbyi, Rotqbybi,
  Rotqmbi, Rotqmbii, Rotqmby, Rotqmbyi, Rotqmbybi,
  Cbd, Chd, Cwd, Cdd, Cbx, Chx, Cwx, Cdx,
  Fa, Fs, Fm, Fceq, Fcgt, Fcmeq, Fcmgt, Frest, Frsqest, Fi, Fscrrd, Fscrwr,
  Dfa, Dfs, Dfm, Dfma, Dfms, Dfnms, Dfnma, Dfceq, Dfcgt, Dfcmeq, Dfcmgt, Dftsv, Fesd, Frds,
};

struct Table {
  std::array<Op, 2048> op{};
  void add(uint32_t value, int width, Op o) {
    const int free_bits = 11 - width;
    const uint32_t base = value << free_bits;
    for (uint32_t k = 0; k < (1u << free_bits); ++k) op[base | k] = o;
  }
  Table();
};

struct QuadShuffleTables {
  alignas(16) __m128i rotqby[16];
  alignas(16) __m128i shlqby[17];
  alignas(16) __m128i rotqmby[17];
  alignas(16) V128 insert[4][16];  // 0:b(1), 1:h(2), 2:w(4), 3:d(8)
  alignas(16) V128 fsm[16];
  alignas(16) V128 fsmh[256];
  alignas(16) uint64_t fsmb8[256];
  alignas(16) __m128i bswap32_mask;
  alignas(16) __m128i sign_w;
  alignas(16) __m128i sign_h;
  alignas(16) __m128i sign_b;
  alignas(16) __m128i shufb_8f;
  alignas(16) __m128i shufb_03;
  alignas(16) __m128i shufb_e0;
  alignas(16) __m128i shufb_c0;
  alignas(16) __m128i shufb_80;
  alignas(16) __m128i all_ones;
  alignas(16) __m128i mask_ffff;
  alignas(16) __m128i mask_3f;
  alignas(16) __m128i mask_1f;
  alignas(16) __m128i const_32;
  alignas(16) __m128i const_31;
  alignas(16) __m128i gbh_shuf;
  alignas(16) __m128i gbb_shuf;
  float scale_173[256];
  double scale_173_d[256];
  float scale_155[256];
  QuadShuffleTables();
};

extern const Table g_table;
extern const QuadShuffleTables g_qshuf;

inline int32_t sext(uint32_t v, int bits) { return int32_t(v << (32 - bits)) >> (32 - bits); }

// Executes a single instruction at cur_pc via the interpreter.
// Returns next_pc (masked to kLsMask & ~3u), or 0xFFFFFFFFu if a final `stop`
// ended execution (with self->jit_stop_code_ populated).
uint32_t jit_step_fallback(SpuCore* self, uint32_t cur_pc);

}  // namespace spu::detail
