// SPU interpreter. Instruction encodings and semantics follow the public
// "Synergistic Processor Unit Instruction Set Architecture" (v1.2) document.
//
// Formats (bit 0 = MSB):
//   RRR  op4  rt7 rb7 ra7 rc7      RR   op11 rb7 ra7 rt7
//   RI7  op11 i7  ra7 rt7          RI8  op10 i8  ra7 rt7
//   RI10 op8  i10 ra7 rt7          RI16 op9  i16 rt7
//   RI18 op7  i18 rt7
// The opcode space is a prefix code: a 2048-entry table indexed by the top 11
// bits maps every instruction word to its operation in one lookup.
//
// Registers are four native words (see V128); element-wise operations use SSE on
// that layout directly. Single-precision maths runs with FTZ/DAZ set on the SPU's
// host thread, which matches the SPU's lack of denormals.

#define _CRT_SECURE_NO_WARNINGS
#include "spu_interp.h"
#include "spu_internal.h"
#include "spu_jit.h"

#include <immintrin.h>

#if defined(_MSC_VER)
#pragma warning(disable : 4310)  // byte constants such as char(0xE0) for _mm_set1_epi8
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace spu {

SpuCore::SpuCore(SpuHost* h) : host(h), ls_(new uint8_t[kLsSize]) {
  std::memset(ls_.get(), 0, kLsSize);
  for (auto& r : gpr) r = V128::zero();
  static const bool no_jit = std::getenv("PS3RECOMP_SPU_NO_JIT") != nullptr;
  if (!no_jit) {
    auto j = std::make_unique<SpuJit>();
    if (j->available()) jit_ = std::move(j);
  }
}

SpuCore::~SpuCore() = default;

void SpuCore::invalidate_ls(uint32_t addr, uint32_t len) {
  if (jit_) jit_->invalidate_ls(addr, len);
}

void SpuCore::invalidate_all() {
  if (jit_) jit_->invalidate_all();
}

namespace detail {

Table::Table() {
  // RRR (4 bits)
  add(0x8, 4, Op::Selb); add(0xB, 4, Op::Shufb); add(0xC, 4, Op::Mpya);
  add(0xD, 4, Op::Fnms); add(0xE, 4, Op::Fma); add(0xF, 4, Op::Fms);
  // RI18 (7 bits)
  add(0x21, 7, Op::Ila); add(0x08, 7, Op::Hbra); add(0x09, 7, Op::Hbrr);
  // RI10 (8 bits)
  add(0x34, 8, Op::Lqd); add(0x24, 8, Op::Stqd); add(0x1C, 8, Op::Ai); add(0x1D, 8, Op::Ahi);
  add(0x0C, 8, Op::Sfi); add(0x0D, 8, Op::Sfhi); add(0x14, 8, Op::Andi); add(0x15, 8, Op::Andhi);
  add(0x16, 8, Op::Andbi); add(0x04, 8, Op::Ori); add(0x05, 8, Op::Orhi); add(0x06, 8, Op::Orbi);
  add(0x44, 8, Op::Xori); add(0x45, 8, Op::Xorhi); add(0x46, 8, Op::Xorbi); add(0x74, 8, Op::Mpyi);
  add(0x75, 8, Op::Mpyui); add(0x4C, 8, Op::Cgti); add(0x4D, 8, Op::Cgthi); add(0x4E, 8, Op::Cgtbi);
  add(0x5C, 8, Op::Clgti); add(0x5D, 8, Op::Clgthi); add(0x5E, 8, Op::Clgtbi); add(0x7C, 8, Op::Ceqi);
  add(0x7D, 8, Op::Ceqhi); add(0x7E, 8, Op::Ceqbi); add(0x4F, 8, Op::Hgti); add(0x5F, 8, Op::Hlgti);
  add(0x7F, 8, Op::Heqi);
  // RI16 (9 bits)
  add(0x064, 9, Op::Br); add(0x060, 9, Op::Bra); add(0x066, 9, Op::Brsl); add(0x062, 9, Op::Brasl);
  add(0x042, 9, Op::Brnz); add(0x040, 9, Op::Brz); add(0x046, 9, Op::Brhnz); add(0x044, 9, Op::Brhz);
  add(0x061, 9, Op::Lqa); add(0x067, 9, Op::Lqr); add(0x041, 9, Op::Stqa); add(0x047, 9, Op::Stqr);
  add(0x081, 9, Op::Il); add(0x083, 9, Op::Ilh); add(0x082, 9, Op::Ilhu); add(0x0C1, 9, Op::Iohl);
  add(0x065, 9, Op::Fsmbi);
  // RI8 (10 bits)
  add(0x1D8, 10, Op::Cflts); add(0x1D9, 10, Op::Cfltu); add(0x1DA, 10, Op::Csflt); add(0x1DB, 10, Op::Cuflt);
  // RR / RI7 (11 bits)
  const struct { uint32_t v; Op o; } rr[] = {
      {0x000, Op::Stop}, {0x140, Op::Stop}, {0x001, Op::Nop}, {0x201, Op::Nop}, {0x002, Op::Nop}, {0x003, Op::Nop},
      {0x1AC, Op::Nop}, {0x00C, Op::Mfspr}, {0x10C, Op::Mtspr}, {0x00D, Op::Rdch}, {0x00F, Op::Rchcnt},
      {0x10D, Op::Wrch}, {0x1A8, Op::Bi}, {0x1A9, Op::Bisl}, {0x1AB, Op::Bisl}, {0x1AA, Op::Iret},
      {0x128, Op::Biz}, {0x129, Op::Binz}, {0x12A, Op::Bihz}, {0x12B, Op::Bihnz}, {0x1C4, Op::Lqx},
      {0x144, Op::Stqx}, {0x3D8, Op::Heq}, {0x258, Op::Hgt}, {0x2D8, Op::Hlgt},
      {0x0C0, Op::A}, {0x0C8, Op::Ah}, {0x040, Op::Sf}, {0x048, Op::Sfh}, {0x0C2, Op::Cg}, {0x042, Op::Bg},
      {0x340, Op::Addx}, {0x341, Op::Sfx}, {0x342, Op::Cgx}, {0x343, Op::Bgx}, {0x3C4, Op::Mpy},
      {0x3CC, Op::Mpyu}, {0x3C5, Op::Mpyh}, {0x3C7, Op::Mpys}, {0x3C6, Op::Mpyhh}, {0x346, Op::Mpyhha},
      {0x3CE, Op::Mpyhhu}, {0x34E, Op::Mpyhhau}, {0x2A5, Op::Clz}, {0x2B4, Op::Cntb}, {0x1B4, Op::Fsm},
      {0x1B5, Op::Fsmh}, {0x1B6, Op::Fsmb}, {0x1B0, Op::Gb}, {0x1B1, Op::Gbh}, {0x1B2, Op::Gbb},
      {0x0D3, Op::Avgb}, {0x053, Op::Absdb}, {0x253, Op::Sumb}, {0x2B6, Op::Xsbh}, {0x2AE, Op::Xshw},
      {0x2A6, Op::Xswd}, {0x0C1, Op::And}, {0x2C1, Op::Andc}, {0x041, Op::Or}, {0x2C9, Op::Orc},
      {0x241, Op::Xor}, {0x0C9, Op::Nand}, {0x049, Op::Nor}, {0x249, Op::Eqv}, {0x1F0, Op::Orx},
      {0x3C0, Op::Ceq}, {0x3C8, Op::Ceqh}, {0x3D0, Op::Ceqb}, {0x240, Op::Cgt}, {0x248, Op::Cgth},
      {0x250, Op::Cgtb}, {0x2C0, Op::Clgt}, {0x2C8, Op::Clgth}, {0x2D0, Op::Clgtb},
      {0x05F, Op::Shlh}, {0x07F, Op::Shlhi}, {0x05B, Op::Shl}, {0x07B, Op::Shli}, {0x05C, Op::Roth},
      {0x07C, Op::Rothi}, {0x058, Op::Rot}, {0x078, Op::Roti}, {0x05D, Op::Rothm}, {0x07D, Op::Rothmi},
      {0x059, Op::Rotm}, {0x079, Op::Rotmi}, {0x05E, Op::Rotmah}, {0x07E, Op::Rotmahi}, {0x05A, Op::Rotma},
      {0x07A, Op::Rotmai}, {0x1DB, Op::Shlqbi}, {0x1FB, Op::Shlqbii}, {0x1DF, Op::Shlqby},
      {0x1FF, Op::Shlqbyi}, {0x1CF, Op::Shlqbybi}, {0x1D8, Op::Rotqbi}, {0x1F8, Op::Rotqbii},
      {0x1DC, Op::Rotqby}, {0x1FC, Op::Rotqbyi}, {0x1CC, Op::Rotqbybi}, {0x1D9, Op::Rotqmbi},
      {0x1F9, Op::Rotqmbii}, {0x1DD, Op::Rotqmby}, {0x1FD, Op::Rotqmbyi}, {0x1CD, Op::Rotqmbybi},
      {0x1F4, Op::Cbd}, {0x1F5, Op::Chd}, {0x1F6, Op::Cwd}, {0x1F7, Op::Cdd}, {0x1D4, Op::Cbx},
      {0x1D5, Op::Chx}, {0x1D6, Op::Cwx}, {0x1D7, Op::Cdx}, {0x2C4, Op::Fa}, {0x2C5, Op::Fs},
      {0x2C6, Op::Fm}, {0x3C2, Op::Fceq}, {0x2C2, Op::Fcgt}, {0x3CA, Op::Fcmeq}, {0x2CA, Op::Fcmgt},
      {0x1B8, Op::Frest}, {0x1B9, Op::Frsqest}, {0x3D4, Op::Fi}, {0x398, Op::Fscrrd}, {0x3BA, Op::Fscrwr},
      {0x2CC, Op::Dfa}, {0x2CD, Op::Dfs}, {0x2CE, Op::Dfm}, {0x35C, Op::Dfma}, {0x35D, Op::Dfms},
      {0x35E, Op::Dfnms}, {0x35F, Op::Dfnma}, {0x3C3, Op::Dfceq}, {0x2C3, Op::Dfcgt}, {0x3CB, Op::Dfcmeq},
      {0x2CB, Op::Dfcmgt}, {0x3BF, Op::Dftsv}, {0x3B8, Op::Fesd}, {0x3B9, Op::Frds},
  };
  for (const auto& e : rr) add(e.v, 11, e.o);
}

QuadShuffleTables::QuadShuffleTables() {
  for (unsigned s = 0; s < 16; ++s) {
    alignas(16) uint8_t m[16];
    for (unsigned k = 0; k < 16; ++k) {
      const unsigned g = k ^ 3;
      m[k] = uint8_t(((g + s) & 15) ^ 3);
    }
    rotqby[s] = _mm_load_si128(reinterpret_cast<const __m128i*>(m));
  }
  for (unsigned s = 0; s <= 16; ++s) {
    alignas(16) uint8_t ms[16], mr[16];
    for (unsigned k = 0; k < 16; ++k) {
      const unsigned g = k ^ 3;
      ms[k] = (g + s < 16) ? uint8_t((g + s) ^ 3) : 0x80u;
      mr[k] = (g >= s) ? uint8_t((g - s) ^ 3) : 0x80u;
    }
    shlqby[s] = _mm_load_si128(reinterpret_cast<const __m128i*>(ms));
    rotqmby[s] = _mm_load_si128(reinterpret_cast<const __m128i*>(mr));
  }
  const unsigned sizes[4] = {1, 2, 4, 8};
  for (int si = 0; si < 4; ++si) {
    const unsigned size = sizes[si];
    for (uint32_t t0 = 0; t0 < 16; ++t0) {
      V128 r;
      for (int i = 0; i < 16; ++i) r.set_byte(i, uint8_t(0x10 + i));
      const uint32_t t = t0 & (16 - size);
      for (unsigned i = 0; i < size; ++i)
        r.set_byte(int(t + i), uint8_t(i + (size == 1 ? 3 : size == 2 ? 2 : 0)));
      insert[si][t0] = r;
    }
  }
  for (uint32_t m = 0; m < 16; ++m) {
    V128 r;
    for (int i = 0; i < 4; ++i) r.u[i] = (m >> (3 - i)) & 1 ? ~0u : 0;
    fsm[m] = r;
  }
  for (uint32_t m = 0; m < 256; ++m) {
    V128 rh = V128::zero();
    for (int i = 0; i < 8; ++i) rh.set_h(i, (m >> (7 - i)) & 1 ? 0xFFFF : 0);
    fsmh[m] = rh;

    V128 rb = V128::zero();
    for (int i = 0; i < 8; ++i) rb.set_byte(i, (m >> (7 - i)) & 1 ? 0xFF : 0);
    std::memcpy(&fsmb8[m], rb.u, 8);

    scale_173_d[m] = std::ldexp(1.0, int(173 - m));
    scale_173[m] = float(scale_173_d[m]);
    scale_155[m] = float(std::ldexp(1.0, -int(155 - m)));
  }
  bswap32_mask = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
  sign_w = _mm_set1_epi32(int(0x80000000));
  sign_h = _mm_set1_epi16(short(0x8000));
  sign_b = _mm_set1_epi8(char(0x80));
  shufb_8f = _mm_set1_epi8(char(0x8F));
  shufb_03 = _mm_set1_epi8(0x03);
  shufb_e0 = _mm_set1_epi8(char(0xE0));
  shufb_c0 = _mm_set1_epi8(char(0xC0));
  shufb_80 = _mm_set1_epi8(char(0x80));
  all_ones = _mm_set1_epi32(-1);
  mask_ffff = _mm_set1_epi32(0xFFFF);
  mask_3f = _mm_set1_epi32(0x3F);
  mask_1f = _mm_set1_epi32(0x1F);
  const_32 = _mm_set1_epi32(32);
  const_31 = _mm_set1_epi32(31);
  gbh_shuf = _mm_setr_epi8(12, 14, 8, 10, 4, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1);
  gbb_shuf = _mm_setr_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
}

const Table g_table;
const QuadShuffleTables g_qshuf;

}  // namespace detail

namespace {

using detail::Op;
using detail::g_table;
using detail::g_qshuf;
using detail::sext;

// ---- SSE helpers ----
inline __m128i vec(const V128& r) { return _mm_load_si128(reinterpret_cast<const __m128i*>(r.u)); }
inline V128 put(__m128i v) { V128 r; _mm_store_si128(reinterpret_cast<__m128i*>(r.u), v); return r; }
inline __m128 vecf(const V128& r) { return _mm_load_ps(reinterpret_cast<const float*>(r.u)); }
inline V128 putf(__m128 v) { V128 r; _mm_store_ps(reinterpret_cast<float*>(r.u), v); return r; }
// Native word layout <-> guest byte order (byte-swap every 32-bit word).
inline __m128i bswap32(__m128i v) {
  return _mm_shuffle_epi8(v, _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
}
inline __m128i not128(__m128i v) { return _mm_xor_si128(v, _mm_set1_epi32(-1)); }

inline V128 load_q(const uint8_t* ls, uint32_t a) {
  return put(bswap32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ls + (a & kLsMask & ~15u)))));
}
inline void store_q(uint8_t* ls, uint32_t a, const V128& v) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(ls + (a & kLsMask & ~15u)), bswap32(vec(v)));
}

#if defined(_MSC_VER)
#define SPU_FORCEINLINE __forceinline
#else
#define SPU_FORCEINLINE inline __attribute__((always_inline))
#endif

// Insertion controls for cbd/chd/cwd/cdd (and the x forms).
SPU_FORCEINLINE const V128& insert_control_idx(uint32_t t, int size_idx) {
  return g_qshuf.insert[size_idx][t & 15];
}

#if defined(_MSC_VER)
[[noreturn]] __declspec(noinline) void spu_halt(uint32_t pc, const char* why) {
#else
[[noreturn]] __attribute__((noinline)) void spu_halt(uint32_t pc, const char* why) {
#endif
  throw SpuHalt{pc, why};
}

int32_t sat_s32(double v) {
  if (std::isnan(v)) return 0;
  if (v >= 2147483647.0) return 0x7FFFFFFF;
  if (v <= -2147483648.0) return int32_t(0x80000000u);
  return int32_t(v);
}
uint32_t sat_u32(double v) {
  if (std::isnan(v) || v <= 0) return 0;
  if (v >= 4294967295.0) return 0xFFFFFFFFu;
  return uint32_t(v);
}

int64_t spu_now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
uint32_t read_dec(SpuCore& self) {
  const int64_t now = spu_now_ns();
  if (self.dec_base_ns == 0) {
    self.dec_base_ns = now;
    return self.dec;
  }
  const uint64_t elapsed_ns = uint64_t(std::max<int64_t>(0, now - self.dec_base_ns));
  // 79.8 MHz PS3 timebase / SPU decrementer clock.
  const uint64_t ticks = (elapsed_ns * 798ull) / 10000ull;
  return uint32_t(self.dec - uint32_t(ticks));
}
void write_dec(SpuCore& self, uint32_t v) {
  self.dec = v;
  self.dec_base_ns = spu_now_ns();
}

// SPU single-precision floats have no NaN/Inf: exp==0x00 flushes to +0.0, and
// exp==0x01..0xFF are all finite sign-magnitude values (0x7FFFFFFF = +SPU_MAX).
SPU_FORCEINLINE __m128i spu_abs_norm(__m128i x) {
  const __m128i mag = _mm_and_si128(x, _mm_set1_epi32(0x7FFFFFFF));
  const __m128i is_norm = _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x007FFFFF));
  return _mm_and_si128(mag, is_norm);
}
SPU_FORCEINLINE __m128i spu_signed_ord(__m128i x) {
  const __m128i mag = _mm_and_si128(x, _mm_set1_epi32(0x7FFFFFFF));
  const __m128i is_norm = _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x007FFFFF));
  const __m128i nmag = _mm_and_si128(mag, is_norm);
  const __m128i sign = _mm_srai_epi32(_mm_and_si128(x, is_norm), 31);
  return _mm_sub_epi32(_mm_xor_si128(nmag, sign), sign);  // +nmag or -nmag
}

template <class F> V128 each_w(const V128& a, F f) { V128 r; for (int i = 0; i < 4; ++i) r.u[i] = f(a.u[i], i); return r; }
template <class F> V128 each_h(const V128& a, F f) { V128 r; for (int i = 0; i < 8; ++i) r.set_h(i, uint16_t(f(a.h(i), i))); return r; }

}  // namespace

SPU_FORCEINLINE bool step_impl(SpuCore& self, uint8_t* const ls, uint32_t& pc, uint32_t* stop_code);

namespace detail {

uint32_t jit_step_fallback(SpuCore* self, uint32_t cur_pc) {
  uint32_t pc = cur_pc;
  uint32_t code = 0;
  try {
    if (!step_impl(*self, self->ls(), pc, &code)) {
      self->pc = pc;
      self->jit_stop_code_ = code;
      return 0xFFFFFFFFu;
    }
    return pc;
  } catch (...) {
    self->jit_exception_ = std::current_exception();
    return 0xFFFFFFFEu;
  }
}

}  // namespace detail

uint32_t SpuCore::run() {
  if (jit_) return jit_->run(*this);
  // SPU single precision has no denormals: flush them in hardware for this thread.
  _mm_setcsr(_mm_getcsr() | 0x8040);  // FTZ | DAZ
  uint32_t code = 0;
  uint8_t* const ls = ls_.get();
  uint32_t local_pc = pc;
  for (;;) {
    for (int i = 0; i < 1024; ++i) {
      if (!step_impl(*this, ls, local_pc, &code)) {
        pc = local_pc;
        return code;
      }
    }
    pc = local_pc;  // keep the published pc roughly current (profiler, diagnostics)
    if (exit_requested_) {
      return 0;
    }
  }
}

bool SpuCore::step(uint32_t* stop_code) {
  uint32_t local_pc = pc;
  const bool ok = step_impl(*this, ls_.get(), local_pc, stop_code);
  pc = local_pc;
  return ok;
}

SPU_FORCEINLINE bool step_impl(SpuCore& self, uint8_t* const ls, uint32_t& pc, uint32_t* stop_code) {
  const uint32_t cur_pc = pc;
  uint32_t raw_be;
  std::memcpy(&raw_be, ls + (cur_pc & kLsMask & ~3u), 4);
  const uint32_t raw = bswap_u32(raw_be);
  const uint32_t next = (cur_pc + 4) & kLsMask;
  V128* const R = self.gpr;
  const unsigned rt = raw & 0x7F, ra = (raw >> 7) & 0x7F, rb = (raw >> 14) & 0x7F;
  const uint32_t i16 = (raw >> 7) & 0xFFFF;
  const int32_t i10 = sext((raw >> 14) & 0x3FF, 10);
  const uint32_t i7 = (raw >> 14) & 0x7F;
  const auto rel = [&] { return (cur_pc + (uint32_t(sext(i16, 16)) << 2)) & kLsMask & ~3u; };
  const auto abs = [&] { return (uint32_t(sext(i16, 16)) << 2) & kLsMask & ~3u; };
  const auto link = [&] { R[rt] = V128::zero(); R[rt].u[0] = next; };
  const auto bi_target = [&] { return R[ra].u[0] & kLsMask & ~3u; };
#define VA vec(R[ra])
#define VB vec(R[rb])
#define SET(x) (R[rt] = put(x))

  switch (g_table.op[raw >> 21]) {
    // ---- RRR ----
    case Op::Selb: {
      const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
      const __m128i c = vec(R[rc]);
      R[t] = put(_mm_or_si128(_mm_and_si128(c, VB), _mm_andnot_si128(c, VA)));
      break;
    }
    case Op::Shufb: {
      const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
      const __m128i c = vec(R[rc]);
      const __m128i idx = _mm_xor_si128(_mm_and_si128(c, _mm_set1_epi8(char(0x8F))), _mm_set1_epi8(0x03));
      __m128i r = _mm_blendv_epi8(_mm_shuffle_epi8(VA, idx), _mm_shuffle_epi8(VB, idx), _mm_slli_epi16(c, 3));
      if (_mm_movemask_epi8(_mm_and_si128(c, _mm_slli_epi16(c, 1)))) {
        const __m128i hi3 = _mm_and_si128(c, _mm_set1_epi8(char(0xE0)));
        const __m128i mC0 = _mm_cmpeq_epi8(hi3, _mm_set1_epi8(char(0xC0)));
        const __m128i mE0 = _mm_and_si128(_mm_cmpeq_epi8(hi3, _mm_set1_epi8(char(0xE0))), _mm_set1_epi8(char(0x80)));
        r = _mm_or_si128(r, _mm_or_si128(mC0, mE0));
      }
      R[t] = put(r);
      break;
    }
    case Op::Mpya: {
      const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
      const __m128i p = _mm_mullo_epi32(_mm_srai_epi32(_mm_slli_epi32(VA, 16), 16), _mm_srai_epi32(_mm_slli_epi32(VB, 16), 16));
      R[t] = put(_mm_add_epi32(p, vec(R[rc])));
      break;
    }
    case Op::Fnms: {
      const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
      R[t] = putf(_mm_sub_ps(vecf(R[rc]), _mm_mul_ps(vecf(R[ra]), vecf(R[rb]))));
      break;
    }
    case Op::Fma: {
      const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
      R[t] = putf(_mm_add_ps(_mm_mul_ps(vecf(R[ra]), vecf(R[rb])), vecf(R[rc])));
      break;
    }
    case Op::Fms: {
      const unsigned t = (raw >> 21) & 0x7F, rc = raw & 0x7F;
      R[t] = putf(_mm_sub_ps(_mm_mul_ps(vecf(R[ra]), vecf(R[rb])), vecf(R[rc])));
      break;
    }

    // ---- RI18 ----
    case Op::Ila: R[rt] = V128::splat_w((raw >> 7) & 0x3FFFF); break;
    case Op::Hbra: case Op::Hbrr: case Op::Nop: break;

    // ---- RI10 ----
    case Op::Lqd: R[rt] = load_q(ls, R[ra].u[0] + (uint32_t(i10) << 4)); break;
    case Op::Stqd: store_q(ls, R[ra].u[0] + (uint32_t(i10) << 4), R[rt]); break;
    case Op::Ai: SET(_mm_add_epi32(VA, _mm_set1_epi32(i10))); break;
    case Op::Ahi: SET(_mm_add_epi16(VA, _mm_set1_epi16(short(i10)))); break;
    case Op::Sfi: SET(_mm_sub_epi32(_mm_set1_epi32(i10), VA)); break;
    case Op::Sfhi: SET(_mm_sub_epi16(_mm_set1_epi16(short(i10)), VA)); break;
    case Op::Andi: SET(_mm_and_si128(VA, _mm_set1_epi32(i10))); break;
    case Op::Andhi: SET(_mm_and_si128(VA, _mm_set1_epi16(short(i10)))); break;
    case Op::Andbi: SET(_mm_and_si128(VA, _mm_set1_epi8(char(i10)))); break;
    case Op::Ori: SET(_mm_or_si128(VA, _mm_set1_epi32(i10))); break;
    case Op::Orhi: SET(_mm_or_si128(VA, _mm_set1_epi16(short(i10)))); break;
    case Op::Orbi: SET(_mm_or_si128(VA, _mm_set1_epi8(char(i10)))); break;
    case Op::Xori: SET(_mm_xor_si128(VA, _mm_set1_epi32(i10))); break;
    case Op::Xorhi: SET(_mm_xor_si128(VA, _mm_set1_epi16(short(i10)))); break;
    case Op::Xorbi: SET(_mm_xor_si128(VA, _mm_set1_epi8(char(i10)))); break;
    case Op::Mpyi: SET(_mm_mullo_epi32(_mm_srai_epi32(_mm_slli_epi32(VA, 16), 16), _mm_set1_epi32(i10))); break;
    case Op::Mpyui: SET(_mm_mullo_epi32(_mm_and_si128(VA, _mm_set1_epi32(0xFFFF)), _mm_set1_epi32(int(uint16_t(i10))))); break;
    case Op::Cgti: SET(_mm_cmpgt_epi32(VA, _mm_set1_epi32(i10))); break;
    case Op::Cgthi: SET(_mm_cmpgt_epi16(VA, _mm_set1_epi16(short(i10)))); break;
    case Op::Cgtbi: SET(_mm_cmpgt_epi8(VA, _mm_set1_epi8(char(i10)))); break;
    case Op::Clgti: {
      const __m128i k = _mm_set1_epi32(int(0x80000000));
      SET(_mm_cmpgt_epi32(_mm_xor_si128(VA, k), _mm_xor_si128(_mm_set1_epi32(i10), k)));
      break;
    }
    case Op::Clgthi: {
      const __m128i k = _mm_set1_epi16(short(0x8000));
      SET(_mm_cmpgt_epi16(_mm_xor_si128(VA, k), _mm_xor_si128(_mm_set1_epi16(short(i10)), k)));
      break;
    }
    case Op::Clgtbi: {
      const __m128i k = _mm_set1_epi8(char(0x80));
      SET(_mm_cmpgt_epi8(_mm_xor_si128(VA, k), _mm_xor_si128(_mm_set1_epi8(char(i10)), k)));
      break;
    }
    case Op::Ceqi: SET(_mm_cmpeq_epi32(VA, _mm_set1_epi32(i10))); break;
    case Op::Ceqhi: SET(_mm_cmpeq_epi16(VA, _mm_set1_epi16(short(i10)))); break;
    case Op::Ceqbi: SET(_mm_cmpeq_epi8(VA, _mm_set1_epi8(char(i10)))); break;
    case Op::Hgti: if (int32_t(R[ra].u[0]) > i10) { self.pc = cur_pc; spu_halt(cur_pc, "hgti"); } break;
    case Op::Hlgti: if (R[ra].u[0] > uint32_t(i10)) { self.pc = cur_pc; spu_halt(cur_pc, "hlgti"); } break;
    case Op::Heqi: if (R[ra].u[0] == uint32_t(i10)) { self.pc = cur_pc; spu_halt(cur_pc, "heqi"); } break;

    // ---- RI16 ----
    case Op::Br: pc = rel(); return true;
    case Op::Bra: pc = abs(); return true;
    case Op::Brsl: { const uint32_t t = rel(); link(); pc = t; return true; }
    case Op::Brasl: { const uint32_t t = abs(); link(); pc = t; return true; }
    case Op::Brnz: pc = R[rt].u[0] != 0 ? rel() : next; return true;
    case Op::Brz: pc = R[rt].u[0] == 0 ? rel() : next; return true;
    case Op::Brhnz: pc = (R[rt].u[0] & 0xFFFF) != 0 ? rel() : next; return true;
    case Op::Brhz: pc = (R[rt].u[0] & 0xFFFF) == 0 ? rel() : next; return true;
    case Op::Lqa: R[rt] = load_q(ls, abs()); break;
    case Op::Lqr: R[rt] = load_q(ls, rel()); break;
    case Op::Stqa: store_q(ls, abs(), R[rt]); break;
    case Op::Stqr: store_q(ls, rel(), R[rt]); break;
    case Op::Il: R[rt] = V128::splat_w(uint32_t(sext(i16, 16))); break;
    case Op::Ilh: R[rt] = V128::splat_w(i16 << 16 | i16); break;
    case Op::Ilhu: R[rt] = V128::splat_w(i16 << 16); break;
    case Op::Iohl: SET(_mm_or_si128(vec(R[rt]), _mm_set1_epi32(int(i16)))); break;
    case Op::Fsmbi: {
      std::memcpy(&R[rt].u[0], &g_qshuf.fsmb8[(i16 >> 8) & 0xFF], 8);
      std::memcpy(&R[rt].u[2], &g_qshuf.fsmb8[i16 & 0xFF], 8);
      break;
    }

    // ---- RI8 ----
    case Op::Cflts: {
      const uint32_t i8 = (raw >> 14) & 0xFF;
      const __m128i exp_ff = _mm_cmpeq_epi32(_mm_and_si128(VA, _mm_set1_epi32(0x7F800000)), _mm_set1_epi32(0x7F800000));
      if (i8 >= 46 && !_mm_movemask_epi8(exp_ff)) {
        const __m128 v = _mm_mul_ps(vecf(R[ra]), _mm_set1_ps(g_qshuf.scale_173[i8]));
        const __m128i res = _mm_cvttps_epi32(v);
        const __m128i ge = _mm_castps_si128(_mm_cmpge_ps(v, _mm_set1_ps(2147483648.0f)));
        const __m128i ord = _mm_castps_si128(_mm_cmpord_ps(v, v));
        SET(_mm_and_si128(_mm_xor_si128(res, ge), ord));
      } else {
        const V128 a = R[ra];
        const double sc = g_qshuf.scale_173_d[i8];
        V128 r;
        for (int i = 0; i < 4; ++i) {
          if ((a.u[i] & 0x7F800000u) == 0x7F800000u)
            r.u[i] = (a.u[i] & 0x80000000u) ? 0x80000000u : 0x7FFFFFFFu;
          else
            r.u[i] = uint32_t(sat_s32(double(a.f(i)) * sc));
        }
        R[rt] = r;
      }
      break;
    }
    case Op::Cfltu: {
      const uint32_t i8 = (raw >> 14) & 0xFF;
      const V128 a = R[ra];
      const double sc = g_qshuf.scale_173_d[i8];
      V128 r;
      for (int i = 0; i < 4; ++i) {
        if ((a.u[i] & 0x7F800000u) == 0x7F800000u)
          r.u[i] = (a.u[i] & 0x80000000u) ? 0u : 0xFFFFFFFFu;
        else
          r.u[i] = sat_u32(double(a.f(i)) * sc);
      }
      R[rt] = r;
      break;
    }
    case Op::Csflt: {
      const uint32_t i8 = (raw >> 14) & 0xFF;
      R[rt] = putf(_mm_mul_ps(_mm_cvtepi32_ps(VA), _mm_set1_ps(g_qshuf.scale_155[i8])));
      break;
    }
    case Op::Cuflt: {
      const uint32_t i8 = (raw >> 14) & 0xFF;
      const __m128 lo = _mm_cvtepi32_ps(_mm_and_si128(VA, _mm_set1_epi32(0xFFFF)));
      const __m128 hi = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srli_epi32(VA, 16)), _mm_set1_ps(65536.0f));
      R[rt] = putf(_mm_mul_ps(_mm_add_ps(hi, lo), _mm_set1_ps(g_qshuf.scale_155[i8])));
      break;
    }

    // ---- control ----
    case Op::Stop: {
      const uint32_t code = raw & 0x3FFF;
      self.pc = cur_pc;
      const bool cont = self.host && self.host->stop(self, code);
      pc = next;
      self.pc = next;
      if (cont) return true;
      if (stop_code) *stop_code = code;
      return false;
    }
    case Op::Mfspr: R[rt] = V128::zero(); break;
    case Op::Mtspr: break;
    case Op::Rdch: {
      self.pc = cur_pc;
      const uint32_t v = ra == SPU_RdSRR0 ? self.srr0 : ra == SPU_RdDec ? read_dec(self) : self.host->read_channel(self, ra);
      R[rt] = V128::zero();
      R[rt].u[0] = v;
      break;
    }
    case Op::Rchcnt: {
      self.pc = cur_pc;
      const uint32_t v = self.host->channel_count(self, ra);
      R[rt] = V128::zero();
      R[rt].u[0] = v;
      break;
    }
    case Op::Wrch:
      if (ra == SPU_WrSRR0) self.srr0 = R[rt].u[0] & kLsMask & ~3u;
      else if (ra == SPU_WrDec) write_dec(self, R[rt].u[0]);
      else { self.pc = cur_pc; self.host->write_channel(self, ra, R[rt].u[0]); }
      break;

    // ---- branches ----
    case Op::Bi: pc = bi_target(); return true;
    case Op::Bisl: { const uint32_t t = bi_target(); link(); pc = t; return true; }
    case Op::Iret: pc = self.srr0; return true;
    case Op::Biz: pc = R[rt].u[0] == 0 ? bi_target() : next; return true;
    case Op::Binz: pc = R[rt].u[0] != 0 ? bi_target() : next; return true;
    case Op::Bihz: pc = (R[rt].u[0] & 0xFFFF) == 0 ? bi_target() : next; return true;
    case Op::Bihnz: pc = (R[rt].u[0] & 0xFFFF) != 0 ? bi_target() : next; return true;

    // ---- memory ----
    case Op::Lqx: R[rt] = load_q(ls, R[ra].u[0] + R[rb].u[0]); break;
    case Op::Stqx: store_q(ls, R[ra].u[0] + R[rb].u[0], R[rt]); break;

    // ---- halts ----
    case Op::Heq: if (R[ra].u[0] == R[rb].u[0]) { self.pc = cur_pc; spu_halt(cur_pc, "heq"); } break;
    case Op::Hgt: if (int32_t(R[ra].u[0]) > int32_t(R[rb].u[0])) { self.pc = cur_pc; spu_halt(cur_pc, "hgt"); } break;
    case Op::Hlgt: if (R[ra].u[0] > R[rb].u[0]) { self.pc = cur_pc; spu_halt(cur_pc, "hlgt"); } break;

    // ---- integer arithmetic ----
    case Op::A: SET(_mm_add_epi32(VA, VB)); break;
    case Op::Ah: SET(_mm_add_epi16(VA, VB)); break;
    case Op::Sf: SET(_mm_sub_epi32(VB, VA)); break;
    case Op::Sfh: SET(_mm_sub_epi16(VB, VA)); break;
    case Op::Cg: {  // carry out of a + b
      const __m128i s = _mm_add_epi32(VA, VB), k = _mm_set1_epi32(int(0x80000000));
      SET(_mm_srli_epi32(_mm_cmpgt_epi32(_mm_xor_si128(VA, k), _mm_xor_si128(s, k)), 31));
      break;
    }
    case Op::Bg: {  // 1 when b >= a (no borrow from b - a)
      const __m128i k = _mm_set1_epi32(int(0x80000000));
      SET(_mm_srli_epi32(not128(_mm_cmpgt_epi32(_mm_xor_si128(VA, k), _mm_xor_si128(VB, k))), 31));
      break;
    }
    case Op::Addx: { const V128 a = R[ra], b = R[rb]; R[rt] = each_w(R[rt], [&](uint32_t t, int i) { return a.u[i] + b.u[i] + (t & 1); }); break; }
    case Op::Sfx: { const V128 a = R[ra], b = R[rb]; R[rt] = each_w(R[rt], [&](uint32_t t, int i) { return b.u[i] + ~a.u[i] + (t & 1); }); break; }
    case Op::Cgx: { const V128 a = R[ra], b = R[rb]; R[rt] = each_w(R[rt], [&](uint32_t t, int i) { return uint32_t((uint64_t(a.u[i]) + b.u[i] + (t & 1)) >> 32); }); break; }
    case Op::Bgx: { const V128 a = R[ra], b = R[rb]; R[rt] = each_w(R[rt], [&](uint32_t t, int i) { return uint32_t((uint64_t(b.u[i]) + uint32_t(~a.u[i]) + (t & 1)) >> 32); }); break; }
    case Op::Mpy: SET(_mm_mullo_epi32(_mm_srai_epi32(_mm_slli_epi32(VA, 16), 16), _mm_srai_epi32(_mm_slli_epi32(VB, 16), 16))); break;
    case Op::Mpyu: { const __m128i m = _mm_set1_epi32(0xFFFF); SET(_mm_mullo_epi32(_mm_and_si128(VA, m), _mm_and_si128(VB, m))); break; }
    case Op::Mpyh: SET(_mm_slli_epi32(_mm_mullo_epi32(_mm_srli_epi32(VA, 16), _mm_and_si128(VB, _mm_set1_epi32(0xFFFF))), 16)); break;
    case Op::Mpys: SET(_mm_srai_epi32(_mm_mullo_epi32(_mm_srai_epi32(_mm_slli_epi32(VA, 16), 16), _mm_srai_epi32(_mm_slli_epi32(VB, 16), 16)), 16)); break;
    case Op::Mpyhh: SET(_mm_mullo_epi32(_mm_srai_epi32(VA, 16), _mm_srai_epi32(VB, 16))); break;
    case Op::Mpyhha: SET(_mm_add_epi32(vec(R[rt]), _mm_mullo_epi32(_mm_srai_epi32(VA, 16), _mm_srai_epi32(VB, 16)))); break;
    case Op::Mpyhhu: SET(_mm_mullo_epi32(_mm_srli_epi32(VA, 16), _mm_srli_epi32(VB, 16))); break;
    case Op::Mpyhhau: SET(_mm_add_epi32(vec(R[rt]), _mm_mullo_epi32(_mm_srli_epi32(VA, 16), _mm_srli_epi32(VB, 16)))); break;
    case Op::Clz: R[rt] = each_w(R[ra], [](uint32_t x, int) { return uint32_t(std::countl_zero(x)); }); break;
    case Op::Cntb: { const V128 a = R[ra]; V128 r; for (int i = 0; i < 16; ++i) r.set_byte(i, uint8_t(std::popcount(a.byte(i)))); R[rt] = r; break; }
    case Op::Fsm: R[rt] = g_qshuf.fsm[R[ra].u[0] & 0xF]; break;
    case Op::Fsmh: R[rt] = g_qshuf.fsmh[R[ra].u[0] & 0xFF]; break;
    case Op::Fsmb: {
      const uint32_t m = R[ra].u[0];
      std::memcpy(&R[rt].u[0], &g_qshuf.fsmb8[(m >> 8) & 0xFF], 8);
      std::memcpy(&R[rt].u[2], &g_qshuf.fsmb8[m & 0xFF], 8);
      break;
    }
    case Op::Gb: {
      const V128& a = R[ra];
      const uint32_t v = ((a.u[0] & 1) << 3) | ((a.u[1] & 1) << 2) | ((a.u[2] & 1) << 1) | (a.u[3] & 1);
      R[rt] = V128::zero();
      R[rt].u[0] = v;
      break;
    }
    case Op::Gbh: {
      const __m128i shuf = _mm_shuffle_epi8(VA, _mm_setr_epi8(12, 14, 8, 10, 4, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1));
      const uint32_t v = uint32_t(_mm_movemask_epi8(_mm_slli_epi16(shuf, 7)));
      R[rt] = V128::zero();
      R[rt].u[0] = v;
      break;
    }
    case Op::Gbb: {
      const __m128i shuf = _mm_shuffle_epi8(VA, _mm_setr_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));
      const uint32_t v = uint32_t(_mm_movemask_epi8(_mm_slli_epi16(shuf, 7)));
      R[rt] = V128::zero();
      R[rt].u[0] = v;
      break;
    }
    case Op::Avgb: SET(_mm_avg_epu8(VA, VB)); break;
    case Op::Absdb: SET(_mm_or_si128(_mm_subs_epu8(VA, VB), _mm_subs_epu8(VB, VA))); break;
    case Op::Sumb: {
      const V128 a = R[ra], b = R[rb];
      V128 r;
      for (int i = 0; i < 4; ++i) {
        const uint32_t x = a.u[i], y = b.u[i];
        const uint32_t sa = (x >> 24) + ((x >> 16) & 0xFF) + ((x >> 8) & 0xFF) + (x & 0xFF);
        const uint32_t sb = (y >> 24) + ((y >> 16) & 0xFF) + ((y >> 8) & 0xFF) + (y & 0xFF);
        r.u[i] = sb << 16 | sa;
      }
      R[rt] = r;
      break;
    }
    case Op::Xsbh: SET(_mm_srai_epi16(_mm_slli_epi16(VA, 8), 8)); break;
    case Op::Xshw: SET(_mm_srai_epi32(_mm_slli_epi32(VA, 16), 16)); break;
    case Op::Xswd: { const V128 a = R[ra]; V128 r; for (int i = 0; i < 2; ++i) r.set_d(i, uint64_t(int64_t(int32_t(a.u[2 * i + 1])))); R[rt] = r; break; }

    // ---- logical ----
    case Op::And: SET(_mm_and_si128(VA, VB)); break;
    case Op::Andc: SET(_mm_andnot_si128(VB, VA)); break;
    case Op::Or: SET(_mm_or_si128(VA, VB)); break;
    case Op::Orc: SET(_mm_or_si128(VA, not128(VB))); break;
    case Op::Xor: SET(_mm_xor_si128(VA, VB)); break;
    case Op::Nand: SET(not128(_mm_and_si128(VA, VB))); break;
    case Op::Nor: SET(not128(_mm_or_si128(VA, VB))); break;
    case Op::Eqv: SET(not128(_mm_xor_si128(VA, VB))); break;
    case Op::Orx: { const V128 a = R[ra]; R[rt] = V128::zero(); R[rt].u[0] = a.u[0] | a.u[1] | a.u[2] | a.u[3]; break; }

    // ---- compares ----
    case Op::Ceq: SET(_mm_cmpeq_epi32(VA, VB)); break;
    case Op::Ceqh: SET(_mm_cmpeq_epi16(VA, VB)); break;
    case Op::Ceqb: SET(_mm_cmpeq_epi8(VA, VB)); break;
    case Op::Cgt: SET(_mm_cmpgt_epi32(VA, VB)); break;
    case Op::Cgth: SET(_mm_cmpgt_epi16(VA, VB)); break;
    case Op::Cgtb: SET(_mm_cmpgt_epi8(VA, VB)); break;
    case Op::Clgt: { const __m128i k = _mm_set1_epi32(int(0x80000000)); SET(_mm_cmpgt_epi32(_mm_xor_si128(VA, k), _mm_xor_si128(VB, k))); break; }
    case Op::Clgth: { const __m128i k = _mm_set1_epi16(short(0x8000)); SET(_mm_cmpgt_epi16(_mm_xor_si128(VA, k), _mm_xor_si128(VB, k))); break; }
    case Op::Clgtb: { const __m128i k = _mm_set1_epi8(char(0x80)); SET(_mm_cmpgt_epi8(_mm_xor_si128(VA, k), _mm_xor_si128(VB, k))); break; }

    // ---- element shifts and rotates ----
    case Op::Shlh: { const V128 b = R[rb]; R[rt] = each_h(R[ra], [&](uint16_t x, int i) { unsigned s = b.h(i) & 0x1F; return s > 15 ? 0 : x << s; }); break; }
    case Op::Shlhi: { const unsigned s = i7 & 0x1F; SET(s > 15 ? _mm_setzero_si128() : _mm_slli_epi16(VA, int(s))); break; }
    case Op::Shl: SET(_mm_sllv_epi32(VA, _mm_and_si128(VB, _mm_set1_epi32(0x3F)))); break;  // counts >= 32 give 0
    case Op::Shli: { const unsigned s = i7 & 0x3F; SET(s > 31 ? _mm_setzero_si128() : _mm_slli_epi32(VA, int(s))); break; }
    case Op::Roth: { const V128 b = R[rb]; R[rt] = each_h(R[ra], [&](uint16_t x, int i) { unsigned s = b.h(i) & 0xF; return uint16_t(x << s | x >> ((16 - s) & 15)); }); break; }
    case Op::Rothi: { const unsigned s = i7 & 0xF; R[rt] = each_h(R[ra], [&](uint16_t x, int) { return uint16_t(x << s | x >> ((16 - s) & 15)); }); break; }
    case Op::Rot: {
      const __m128i s = _mm_and_si128(VB, _mm_set1_epi32(0x1F));
      SET(_mm_or_si128(_mm_sllv_epi32(VA, s), _mm_srlv_epi32(VA, _mm_sub_epi32(_mm_set1_epi32(32), s))));
      break;
    }
    case Op::Roti: { const int s = int(i7 & 0x1F); SET(s ? _mm_or_si128(_mm_slli_epi32(VA, s), _mm_srli_epi32(VA, 32 - s)) : VA); break; }
    case Op::Rothm: { const V128 b = R[rb]; R[rt] = each_h(R[ra], [&](uint16_t x, int i) { unsigned s = (0u - b.h(i)) & 0x1F; return s > 15 ? 0 : x >> s; }); break; }
    case Op::Rothmi: { const unsigned s = (0u - i7) & 0x1F; SET(s > 15 ? _mm_setzero_si128() : _mm_srli_epi16(VA, int(s))); break; }
    case Op::Rotm: SET(_mm_srlv_epi32(VA, _mm_and_si128(_mm_sub_epi32(_mm_setzero_si128(), VB), _mm_set1_epi32(0x3F)))); break;
    case Op::Rotmi: { const unsigned s = (0u - i7) & 0x3F; SET(s > 31 ? _mm_setzero_si128() : _mm_srli_epi32(VA, int(s))); break; }
    case Op::Rotmah: { const V128 b = R[rb]; R[rt] = each_h(R[ra], [&](uint16_t x, int i) { unsigned s = (0u - b.h(i)) & 0x1F; return uint16_t(int16_t(x) >> (s > 15 ? 15 : s)); }); break; }
    case Op::Rotmahi: { const unsigned s = (0u - i7) & 0x1F; SET(_mm_srai_epi16(VA, int(s > 15 ? 15 : s))); break; }
    case Op::Rotma: {
      const __m128i s = _mm_min_epu32(_mm_and_si128(_mm_sub_epi32(_mm_setzero_si128(), VB), _mm_set1_epi32(0x3F)), _mm_set1_epi32(31));
      SET(_mm_srav_epi32(VA, s));
      break;
    }
    case Op::Rotmai: { const unsigned s = (0u - i7) & 0x3F; SET(_mm_srai_epi32(VA, int(s > 31 ? 31 : s))); break; }

    // ---- whole-quadword shifts and rotates ----
    case Op::Shlqbi: {
      const int s = int(R[rb].u[0] & 7);
      const __m128i a = VA;
      SET(s ? _mm_or_si128(_mm_slli_epi32(a, s), _mm_srli_epi32(_mm_srli_si128(a, 4), 32 - s)) : a);
      break;
    }
    case Op::Shlqbii: {
      const int s = int(i7 & 7);
      const __m128i a = VA;
      SET(s ? _mm_or_si128(_mm_slli_epi32(a, s), _mm_srli_epi32(_mm_srli_si128(a, 4), 32 - s)) : a);
      break;
    }
    case Op::Shlqby: { const unsigned n = std::min<unsigned>(R[rb].u[0] & 0x1F, 16); SET(_mm_shuffle_epi8(VA, g_qshuf.shlqby[n])); break; }
    case Op::Shlqbyi: { const unsigned n = std::min<unsigned>(i7 & 0x1F, 16); SET(_mm_shuffle_epi8(VA, g_qshuf.shlqby[n])); break; }
    case Op::Shlqbybi: { const unsigned n = std::min<unsigned>((R[rb].u[0] >> 3) & 0x1F, 16); SET(_mm_shuffle_epi8(VA, g_qshuf.shlqby[n])); break; }
    case Op::Rotqbi: {
      const int s = int(R[rb].u[0] & 7);
      const __m128i a = VA;
      SET(s ? _mm_or_si128(_mm_slli_epi32(a, s), _mm_srli_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 3, 2, 1)), 32 - s)) : a);
      break;
    }
    case Op::Rotqbii: {
      const int s = int(i7 & 7);
      const __m128i a = VA;
      SET(s ? _mm_or_si128(_mm_slli_epi32(a, s), _mm_srli_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(0, 3, 2, 1)), 32 - s)) : a);
      break;
    }
    case Op::Rotqby: SET(_mm_shuffle_epi8(VA, g_qshuf.rotqby[R[rb].u[0] & 0xF])); break;
    case Op::Rotqbyi: SET(_mm_shuffle_epi8(VA, g_qshuf.rotqby[i7 & 0xF])); break;
    case Op::Rotqbybi: SET(_mm_shuffle_epi8(VA, g_qshuf.rotqby[(R[rb].u[0] >> 3) & 0xF])); break;
    case Op::Rotqmbi: {
      const int s = int((0u - R[rb].u[0]) & 7);
      const __m128i a = VA;
      SET(s ? _mm_or_si128(_mm_srli_epi32(a, s), _mm_slli_epi32(_mm_slli_si128(a, 4), 32 - s)) : a);
      break;
    }
    case Op::Rotqmbii: {
      const int s = int((0u - i7) & 7);
      const __m128i a = VA;
      SET(s ? _mm_or_si128(_mm_srli_epi32(a, s), _mm_slli_epi32(_mm_slli_si128(a, 4), 32 - s)) : a);
      break;
    }
    case Op::Rotqmby: { const unsigned n = std::min<unsigned>((0u - R[rb].u[0]) & 0x1F, 16); SET(_mm_shuffle_epi8(VA, g_qshuf.rotqmby[n])); break; }
    case Op::Rotqmbyi: { const unsigned n = std::min<unsigned>((0u - i7) & 0x1F, 16); SET(_mm_shuffle_epi8(VA, g_qshuf.rotqmby[n])); break; }
    case Op::Rotqmbybi: { const unsigned n = std::min<unsigned>((0u - (R[rb].u[0] >> 3)) & 0x1F, 16); SET(_mm_shuffle_epi8(VA, g_qshuf.rotqmby[n])); break; }

    // ---- insertion controls ----
    case Op::Cbd: R[rt] = insert_control_idx(R[ra].u[0] + uint32_t(sext(i7, 7)), 0); break;
    case Op::Chd: R[rt] = insert_control_idx(R[ra].u[0] + uint32_t(sext(i7, 7)), 1); break;
    case Op::Cwd: R[rt] = insert_control_idx(R[ra].u[0] + uint32_t(sext(i7, 7)), 2); break;
    case Op::Cdd: R[rt] = insert_control_idx(R[ra].u[0] + uint32_t(sext(i7, 7)), 3); break;
    case Op::Cbx: R[rt] = insert_control_idx(R[ra].u[0] + R[rb].u[0], 0); break;
    case Op::Chx: R[rt] = insert_control_idx(R[ra].u[0] + R[rb].u[0], 1); break;
    case Op::Cwx: R[rt] = insert_control_idx(R[ra].u[0] + R[rb].u[0], 2); break;
    case Op::Cdx: R[rt] = insert_control_idx(R[ra].u[0] + R[rb].u[0], 3); break;

    // ---- single precision (FTZ/DAZ via MXCSR) ----
    case Op::Fa: R[rt] = putf(_mm_add_ps(vecf(R[ra]), vecf(R[rb]))); break;
    case Op::Fs: R[rt] = putf(_mm_sub_ps(vecf(R[ra]), vecf(R[rb]))); break;
    case Op::Fm: R[rt] = putf(_mm_mul_ps(vecf(R[ra]), vecf(R[rb]))); break;
    case Op::Fceq: SET(_mm_cmpeq_epi32(spu_signed_ord(VA), spu_signed_ord(VB))); break;
    case Op::Fcgt: SET(_mm_cmpgt_epi32(spu_signed_ord(VA), spu_signed_ord(VB))); break;
    case Op::Fcmeq: SET(_mm_cmpeq_epi32(spu_abs_norm(VA), spu_abs_norm(VB))); break;
    case Op::Fcmgt: SET(_mm_cmpgt_epi32(spu_abs_norm(VA), spu_abs_norm(VB))); break;
    // Estimates are computed exactly; `fi` passes the estimate through, so the
    // usual frest+fi (+Newton step) sequences produce correctly rounded results.
    // SPU single precision has no Inf/NaN: clamp frest to ±3e37 (finite even for
    // tiny inputs) and frsqest to 1e18 so a Newton step squaring frsqest(0) stays
    // finite (1e36 < FLT_MAX) and never turns into 0 * Inf = NaN.
    case Op::Frest: {
      const __m128 q = _mm_div_ps(_mm_set1_ps(1.0f), vecf(R[ra]));
      const __m128 hi = _mm_set1_ps(3e37f);
      const __m128 lo = _mm_set1_ps(-3e37f);
      R[rt] = putf(_mm_max_ps(_mm_min_ps(q, hi), lo));
      break;
    }
    case Op::Frsqest: {
      const __m128 abs_a = _mm_and_ps(vecf(R[ra]), _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
      const __m128 q = _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(abs_a));
      R[rt] = putf(_mm_min_ps(q, _mm_set1_ps(1e18f)));
      break;
    }
    case Op::Fi: R[rt] = R[rb]; break;
    case Op::Fscrrd: R[rt] = V128::zero(); break;
    case Op::Fscrwr: break;

    // ---- double precision ----
    case Op::Dfa: case Op::Dfs: case Op::Dfm: case Op::Dfma: case Op::Dfms: case Op::Dfnms: case Op::Dfnma:
    case Op::Dfceq: case Op::Dfcgt: case Op::Dfcmeq: case Op::Dfcmgt: {
      const V128 a = R[ra], b = R[rb], t = R[rt];
      V128 r;
      const Op o = g_table.op[raw >> 21];
      for (int i = 0; i < 2; ++i) {
        const double x = a.df(i), y = b.df(i), z = t.df(i);
        switch (o) {
          case Op::Dfa: r.set_df(i, x + y); break;
          case Op::Dfs: r.set_df(i, x - y); break;
          case Op::Dfm: r.set_df(i, x * y); break;
          case Op::Dfma: r.set_df(i, x * y + z); break;
          case Op::Dfms: r.set_df(i, x * y - z); break;
          case Op::Dfnms: r.set_df(i, -(x * y - z)); break;
          case Op::Dfnma: r.set_df(i, -(x * y + z)); break;
          case Op::Dfceq: r.set_d(i, x == y ? ~0ull : 0); break;
          case Op::Dfcgt: r.set_d(i, x > y ? ~0ull : 0); break;
          case Op::Dfcmeq: r.set_d(i, std::fabs(x) == std::fabs(y) ? ~0ull : 0); break;
          default: r.set_d(i, std::fabs(x) > std::fabs(y) ? ~0ull : 0); break;
        }
      }
      R[rt] = r;
      break;
    }
    case Op::Dftsv: {
      const V128 a = R[ra];
      V128 r;
      for (int i = 0; i < 2; ++i) {
        const uint64_t bits = a.d(i);
        const bool neg = (bits >> 63) != 0;
        const uint64_t exp = (bits >> 52) & 0x7FFu;
        const uint64_t frac = bits & 0x000FFFFFFFFFFFFFull;
        uint32_t flags = 0;
        if (exp == 0x7FFu) flags = frac ? 0x40u : (neg ? 0x10u : 0x20u);
        else if (exp == 0u) flags = frac ? (neg ? 0x01u : 0x02u) : (neg ? 0x04u : 0x08u);
        r.set_d(i, (flags & i7) ? ~0ull : 0ull);
      }
      R[rt] = r;
      break;
    }
    case Op::Fesd: { const V128 a = R[ra]; V128 r; for (int i = 0; i < 2; ++i) r.set_df(i, double(a.f(2 * i))); R[rt] = r; break; }
    case Op::Frds: { const V128 a = R[ra]; V128 r = V128::zero(); for (int i = 0; i < 2; ++i) r.set_f(2 * i, float(a.df(i))); R[rt] = r; break; }

    case Op::Invalid:
    default:
      self.pc = cur_pc;
      spu_halt(cur_pc, "invalid opcode");
  }
#undef VA
#undef VB
#undef SET
  pc = next;
  return true;
}

}  // namespace spu
