#pragma once
// AltiVec/VMX helpers. One function per instruction, named after the mnemonic.
// Element indices are guest (big-endian) element numbers: element 0 is leftmost.
//
// These are straightforward scalar loops; the compiler vectorizes most of them.
// Not modelled: VSCR[SAT] (saturation is applied, the sticky bit is not set) and
// the Cell's non-Java mode denormal flushing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace ppu {
struct Context;
struct CRField;
}

namespace ppu::vmx {

template <class T> constexpr int lanes = int(16 / sizeof(T));

template <class T> inline T get(const v128& v, int i) {
  T t; std::memcpy(&t, v.b + 16 - (i + 1) * int(sizeof(T)), sizeof(T)); return t;
}
template <class T> inline void set(v128& v, int i, T t) {
  std::memcpy(v.b + 16 - (i + 1) * int(sizeof(T)), &t, sizeof(T));
}
template <class T> inline T sat(int64_t v) {
  return T(std::clamp<int64_t>(v, int64_t(std::numeric_limits<T>::min()), int64_t(std::numeric_limits<T>::max())));
}
template <class T> inline T satf(double v) {
  if (std::isnan(v)) return T(0);
  if (v >= double(std::numeric_limits<T>::max())) return std::numeric_limits<T>::max();
  if (v <= double(std::numeric_limits<T>::min())) return std::numeric_limits<T>::min();
  return T(v);
}

// ---- loads / stores -------------------------------------------------------
inline void load(v128& d, const uint8_t* mem, uint32_t ea) {
  const uint8_t* p = mem + (ea & ~15u);
  for (int k = 0; k < 16; ++k) d.b[k] = p[15 - k];
}
inline void store(const v128& s, uint8_t* mem, uint32_t ea) {
  uint8_t* p = mem + (ea & ~15u);
  for (int k = 0; k < 16; ++k) p[15 - k] = s.b[k];
}
// stvebx/stvehx/stvewx: store only the element addressed by ea.
inline void store_elem(const v128& s, uint8_t* mem, uint32_t ea, int size) {
  ea &= ~uint32_t(size - 1);
  const int first = int(ea & 15);
  for (int k = 0; k < size; ++k) mem[ea + k] = get<uint8_t>(s, first + k);
}
inline void lvsl(v128& d, uint32_t ea) { for (int i = 0; i < 16; ++i) set<uint8_t>(d, i, uint8_t((ea & 15) + i)); }
inline void lvsr(v128& d, uint32_t ea) { for (int i = 0; i < 16; ++i) set<uint8_t>(d, i, uint8_t(16 - (ea & 15) + i)); }
// Cell-specific unaligned helpers.
inline void lvlx(v128& d, const uint8_t* mem, uint32_t ea) {
  const int sh = int(ea & 15);
  v128 r{};
  for (int i = 0; i < 16 - sh; ++i) set<uint8_t>(r, i, mem[ea + i]);
  d = r;
}
inline void lvrx(v128& d, const uint8_t* mem, uint32_t ea) {
  const int sh = int(ea & 15);
  const uint32_t base = ea & ~15u;
  v128 r{};
  for (int i = 0; i < sh; ++i) set<uint8_t>(r, 16 - sh + i, mem[base + i]);
  d = r;
}
inline void stvlx(const v128& s, uint8_t* mem, uint32_t ea) {
  const int sh = int(ea & 15);
  for (int i = 0; i < 16 - sh; ++i) mem[ea + i] = get<uint8_t>(s, i);
}
inline void stvrx(const v128& s, uint8_t* mem, uint32_t ea) {
  const int sh = int(ea & 15);
  const uint32_t base = ea & ~15u;
  for (int i = 0; i < sh; ++i) mem[base + i] = get<uint8_t>(s, 16 - sh + i);
}

// ---- generic lane-wise builders --------------------------------------------
#define VMX_BIN(name, T, expr)                                       \
  inline void name(v128& d, const v128& va, const v128& vb) {        \
    v128 r;                                                          \
    for (int i = 0; i < lanes<T>; ++i) {                             \
      const T a = get<T>(va, i), b = get<T>(vb, i);                  \
      (void)a; (void)b;                                              \
      set<T>(r, i, T(expr));                                         \
    }                                                                \
    d = r;                                                           \
  }
#define VMX_UN(name, T, expr)                                        \
  inline void name(v128& d, const v128& vb) {                        \
    v128 r;                                                          \
    for (int i = 0; i < lanes<T>; ++i) {                             \
      const T b = get<T>(vb, i);                                     \
      set<T>(r, i, T(expr));                                         \
    }                                                                \
    d = r;                                                           \
  }
#define VMX_CMP(name, T, expr)                                       \
  inline void name(v128& d, const v128& va, const v128& vb) {        \
    using U = std::make_unsigned_t<std::conditional_t<std::is_same_v<T, float>, int32_t, T>>; \
    v128 r;                                                          \
    for (int i = 0; i < lanes<T>; ++i) {                             \
      const T a = get<T>(va, i), b = get<T>(vb, i);                  \
      set<U>(r, i, (expr) ? U(~U(0)) : U(0));                        \
    }                                                                \
    d = r;                                                           \
  }

// ---- integer add / subtract -----------------------------------------------
VMX_BIN(vaddubm, uint8_t, a + b)
VMX_BIN(vadduhm, uint16_t, a + b)
VMX_BIN(vadduwm, uint32_t, a + b)
VMX_BIN(vaddcuw, uint32_t, (uint64_t(a) + b) >> 32)
VMX_BIN(vaddubs, uint8_t, sat<uint8_t>(int64_t(a) + b))
VMX_BIN(vadduhs, uint16_t, sat<uint16_t>(int64_t(a) + b))
VMX_BIN(vadduws, uint32_t, sat<uint32_t>(int64_t(a) + b))
VMX_BIN(vaddsbs, int8_t, sat<int8_t>(int64_t(a) + b))
VMX_BIN(vaddshs, int16_t, sat<int16_t>(int64_t(a) + b))
VMX_BIN(vaddsws, int32_t, sat<int32_t>(int64_t(a) + b))
VMX_BIN(vsububm, uint8_t, a - b)
VMX_BIN(vsubuhm, uint16_t, a - b)
VMX_BIN(vsubuwm, uint32_t, a - b)
VMX_BIN(vsubcuw, uint32_t, a >= b ? 1 : 0)
VMX_BIN(vsububs, uint8_t, sat<uint8_t>(int64_t(a) - b))
VMX_BIN(vsubuhs, uint16_t, sat<uint16_t>(int64_t(a) - b))
VMX_BIN(vsubuws, uint32_t, sat<uint32_t>(int64_t(a) - b))
VMX_BIN(vsubsbs, int8_t, sat<int8_t>(int64_t(a) - b))
VMX_BIN(vsubshs, int16_t, sat<int16_t>(int64_t(a) - b))
VMX_BIN(vsubsws, int32_t, sat<int32_t>(int64_t(a) - b))

// ---- min / max / average ----------------------------------------------------
VMX_BIN(vmaxub, uint8_t, std::max(a, b))
VMX_BIN(vmaxuh, uint16_t, std::max(a, b))
VMX_BIN(vmaxuw, uint32_t, std::max(a, b))
VMX_BIN(vmaxsb, int8_t, std::max(a, b))
VMX_BIN(vmaxsh, int16_t, std::max(a, b))
VMX_BIN(vmaxsw, int32_t, std::max(a, b))
VMX_BIN(vminub, uint8_t, std::min(a, b))
VMX_BIN(vminuh, uint16_t, std::min(a, b))
VMX_BIN(vminuw, uint32_t, std::min(a, b))
VMX_BIN(vminsb, int8_t, std::min(a, b))
VMX_BIN(vminsh, int16_t, std::min(a, b))
VMX_BIN(vminsw, int32_t, std::min(a, b))
VMX_BIN(vavgub, uint8_t, (int64_t(a) + b + 1) >> 1)
VMX_BIN(vavguh, uint16_t, (int64_t(a) + b + 1) >> 1)
VMX_BIN(vavguw, uint32_t, (int64_t(a) + b + 1) >> 1)
VMX_BIN(vavgsb, int8_t, (int64_t(a) + b + 1) >> 1)
VMX_BIN(vavgsh, int16_t, (int64_t(a) + b + 1) >> 1)
VMX_BIN(vavgsw, int32_t, (int64_t(a) + b + 1) >> 1)

// ---- shifts / rotates -------------------------------------------------------
template <class T> inline T rotl_t(T v, unsigned n) {
  constexpr unsigned bits = sizeof(T) * 8;
  n %= bits;
  return n ? T((v << n) | (v >> (bits - n))) : v;
}
VMX_BIN(vrlb, uint8_t, rotl_t<uint8_t>(a, b))
VMX_BIN(vrlh, uint16_t, rotl_t<uint16_t>(a, b))
VMX_BIN(vrlw, uint32_t, rotl_t<uint32_t>(a, b))
VMX_BIN(vslb, uint8_t, uint32_t(a) << (b & 7))
VMX_BIN(vslh, uint16_t, uint32_t(a) << (b & 15))
VMX_BIN(vslw, uint32_t, a << (b & 31))
VMX_BIN(vsrb, uint8_t, a >> (b & 7))
VMX_BIN(vsrh, uint16_t, a >> (b & 15))
VMX_BIN(vsrw, uint32_t, a >> (b & 31))
VMX_BIN(vsrab, int8_t, a >> (b & 7))
VMX_BIN(vsrah, int16_t, a >> (b & 15))
VMX_BIN(vsraw, int32_t, a >> (b & 31))

// Whole-register shifts. Host bytes form a little-endian 128-bit integer.
inline void shl128(v128& d, const v128& a, unsigned bits) {
  uint64_t lo, hi;
  std::memcpy(&lo, a.b, 8); std::memcpy(&hi, a.b + 8, 8);
  if (bits >= 128) { lo = hi = 0; }
  else if (bits >= 64) { hi = lo << (bits - 64); lo = 0; }
  else if (bits) { hi = (hi << bits) | (lo >> (64 - bits)); lo <<= bits; }
  std::memcpy(d.b, &lo, 8); std::memcpy(d.b + 8, &hi, 8);
}
inline void shr128(v128& d, const v128& a, unsigned bits) {
  uint64_t lo, hi;
  std::memcpy(&lo, a.b, 8); std::memcpy(&hi, a.b + 8, 8);
  if (bits >= 128) { lo = hi = 0; }
  else if (bits >= 64) { lo = hi >> (bits - 64); hi = 0; }
  else if (bits) { lo = (lo >> bits) | (hi << (64 - bits)); hi >>= bits; }
  std::memcpy(d.b, &lo, 8); std::memcpy(d.b + 8, &hi, 8);
}
inline void vsl(v128& d, const v128& a, const v128& b) { shl128(d, a, get<uint8_t>(b, 15) & 7); }
inline void vsr(v128& d, const v128& a, const v128& b) { shr128(d, a, get<uint8_t>(b, 15) & 7); }
inline void vslo(v128& d, const v128& a, const v128& b) { shl128(d, a, ((get<uint8_t>(b, 15) >> 3) & 15) * 8); }
inline void vsro(v128& d, const v128& a, const v128& b) { shr128(d, a, ((get<uint8_t>(b, 15) >> 3) & 15) * 8); }

// ---- logical ----------------------------------------------------------------
VMX_BIN(vand, uint32_t, a & b)
VMX_BIN(vandc, uint32_t, a & ~b)
VMX_BIN(vor, uint32_t, a | b)
VMX_BIN(vxor, uint32_t, a ^ b)
VMX_BIN(vnor, uint32_t, ~(a | b))

// ---- merge / splat ----------------------------------------------------------
template <class T> inline void merge(v128& d, const v128& a, const v128& b, int off) {
  v128 r;
  for (int i = 0; i < lanes<T> / 2; ++i) {
    set<T>(r, 2 * i, get<T>(a, i + off));
    set<T>(r, 2 * i + 1, get<T>(b, i + off));
  }
  d = r;
}
inline void vmrghb(v128& d, const v128& a, const v128& b) { merge<uint8_t>(d, a, b, 0); }
inline void vmrghh(v128& d, const v128& a, const v128& b) { merge<uint16_t>(d, a, b, 0); }
inline void vmrghw(v128& d, const v128& a, const v128& b) { merge<uint32_t>(d, a, b, 0); }
inline void vmrglb(v128& d, const v128& a, const v128& b) { merge<uint8_t>(d, a, b, 8); }
inline void vmrglh(v128& d, const v128& a, const v128& b) { merge<uint16_t>(d, a, b, 4); }
inline void vmrglw(v128& d, const v128& a, const v128& b) { merge<uint32_t>(d, a, b, 2); }

template <class T> inline void splat(v128& d, T v) { for (int i = 0; i < lanes<T>; ++i) set<T>(d, i, v); }
inline void vspltb(v128& d, const v128& b, unsigned u) { splat<uint8_t>(d, get<uint8_t>(b, int(u & 15))); }
inline void vsplth(v128& d, const v128& b, unsigned u) { splat<uint16_t>(d, get<uint16_t>(b, int(u & 7))); }
inline void vspltw(v128& d, const v128& b, unsigned u) { splat<uint32_t>(d, get<uint32_t>(b, int(u & 3))); }
inline void vspltisb(v128& d, int s) { splat<int8_t>(d, int8_t(s)); }
inline void vspltish(v128& d, int s) { splat<int16_t>(d, int16_t(s)); }
inline void vspltisw(v128& d, int s) { splat<int32_t>(d, int32_t(s)); }

// ---- pack / unpack ----------------------------------------------------------
template <class Src, class Dst, class F> inline void pack(v128& d, const v128& a, const v128& b, F f) {
  v128 r;
  constexpr int n = lanes<Src>;
  for (int i = 0; i < n; ++i) {
    set<Dst>(r, i, f(get<Src>(a, i)));
    set<Dst>(r, i + n, f(get<Src>(b, i)));
  }
  d = r;
}
inline void vpkuhum(v128& d, const v128& a, const v128& b) { pack<uint16_t, uint8_t>(d, a, b, [](uint16_t v) { return uint8_t(v); }); }
inline void vpkuwum(v128& d, const v128& a, const v128& b) { pack<uint32_t, uint16_t>(d, a, b, [](uint32_t v) { return uint16_t(v); }); }
inline void vpkuhus(v128& d, const v128& a, const v128& b) { pack<uint16_t, uint8_t>(d, a, b, [](uint16_t v) { return sat<uint8_t>(v); }); }
inline void vpkuwus(v128& d, const v128& a, const v128& b) { pack<uint32_t, uint16_t>(d, a, b, [](uint32_t v) { return sat<uint16_t>(v); }); }
inline void vpkshus(v128& d, const v128& a, const v128& b) { pack<int16_t, uint8_t>(d, a, b, [](int16_t v) { return sat<uint8_t>(v); }); }
inline void vpkswus(v128& d, const v128& a, const v128& b) { pack<int32_t, uint16_t>(d, a, b, [](int32_t v) { return sat<uint16_t>(v); }); }
inline void vpkshss(v128& d, const v128& a, const v128& b) { pack<int16_t, int8_t>(d, a, b, [](int16_t v) { return sat<int8_t>(v); }); }
inline void vpkswss(v128& d, const v128& a, const v128& b) { pack<int32_t, int16_t>(d, a, b, [](int32_t v) { return sat<int16_t>(v); }); }

template <class Src, class Dst> inline void unpack(v128& d, const v128& b, int off) {
  v128 r;
  for (int i = 0; i < lanes<Dst>; ++i) set<Dst>(r, i, Dst(get<Src>(b, i + off)));
  d = r;
}
inline void vupkhsb(v128& d, const v128& b) { unpack<int8_t, int16_t>(d, b, 0); }
inline void vupklsb(v128& d, const v128& b) { unpack<int8_t, int16_t>(d, b, 8); }
inline void vupkhsh(v128& d, const v128& b) { unpack<int16_t, int32_t>(d, b, 0); }
inline void vupklsh(v128& d, const v128& b) { unpack<int16_t, int32_t>(d, b, 4); }

// ---- multiply ---------------------------------------------------------------
template <class Src, class Dst> inline void mul_eo(v128& d, const v128& a, const v128& b, int odd) {
  v128 r;
  for (int i = 0; i < lanes<Dst>; ++i)
    set<Dst>(r, i, Dst(Dst(get<Src>(a, 2 * i + odd)) * Dst(get<Src>(b, 2 * i + odd))));
  d = r;
}
inline void vmuleub(v128& d, const v128& a, const v128& b) { mul_eo<uint8_t, uint16_t>(d, a, b, 0); }
inline void vmuloub(v128& d, const v128& a, const v128& b) { mul_eo<uint8_t, uint16_t>(d, a, b, 1); }
inline void vmulesb(v128& d, const v128& a, const v128& b) { mul_eo<int8_t, int16_t>(d, a, b, 0); }
inline void vmulosb(v128& d, const v128& a, const v128& b) { mul_eo<int8_t, int16_t>(d, a, b, 1); }
inline void vmuleuh(v128& d, const v128& a, const v128& b) { mul_eo<uint16_t, uint32_t>(d, a, b, 0); }
inline void vmulouh(v128& d, const v128& a, const v128& b) { mul_eo<uint16_t, uint32_t>(d, a, b, 1); }
inline void vmulesh(v128& d, const v128& a, const v128& b) { mul_eo<int16_t, int32_t>(d, a, b, 0); }
inline void vmulosh(v128& d, const v128& a, const v128& b) { mul_eo<int16_t, int32_t>(d, a, b, 1); }

// ---- multiply-sum / multiply-add (VA form) --------------------------------
inline void vmhaddshs(v128& d, const v128& a, const v128& b, const v128& c) {
  v128 r;
  for (int i = 0; i < 8; ++i)
    set<int16_t>(r, i, sat<int16_t>(((int32_t(get<int16_t>(a, i)) * get<int16_t>(b, i)) >> 15) + get<int16_t>(c, i)));
  d = r;
}
inline void vmhraddshs(v128& d, const v128& a, const v128& b, const v128& c) {
  v128 r;
  for (int i = 0; i < 8; ++i)
    set<int16_t>(r, i, sat<int16_t>(((int32_t(get<int16_t>(a, i)) * get<int16_t>(b, i) + 0x4000) >> 15) + get<int16_t>(c, i)));
  d = r;
}
inline void vmladduhm(v128& d, const v128& a, const v128& b, const v128& c) {
  v128 r;
  for (int i = 0; i < 8; ++i)
    set<uint16_t>(r, i, uint16_t(uint32_t(get<uint16_t>(a, i)) * get<uint16_t>(b, i) + get<uint16_t>(c, i)));
  d = r;
}
template <class A, class B, class Acc, bool Saturate>
inline void msum(v128& d, const v128& a, const v128& b, const v128& c) {
  constexpr int per = int(sizeof(Acc) / sizeof(A));
  v128 r;
  for (int i = 0; i < 4; ++i) {
    int64_t s = int64_t(get<Acc>(c, i));
    for (int j = 0; j < per; ++j) s += int64_t(get<A>(a, i * per + j)) * int64_t(get<B>(b, i * per + j));
    set<Acc>(r, i, Saturate ? sat<Acc>(s) : Acc(s));
  }
  d = r;
}
inline void vmsumubm(v128& d, const v128& a, const v128& b, const v128& c) { msum<uint8_t, uint8_t, uint32_t, false>(d, a, b, c); }
inline void vmsummbm(v128& d, const v128& a, const v128& b, const v128& c) { msum<int8_t, uint8_t, int32_t, false>(d, a, b, c); }
inline void vmsumuhm(v128& d, const v128& a, const v128& b, const v128& c) { msum<uint16_t, uint16_t, uint32_t, false>(d, a, b, c); }
inline void vmsumuhs(v128& d, const v128& a, const v128& b, const v128& c) { msum<uint16_t, uint16_t, uint32_t, true>(d, a, b, c); }
inline void vmsumshm(v128& d, const v128& a, const v128& b, const v128& c) { msum<int16_t, int16_t, int32_t, false>(d, a, b, c); }
inline void vmsumshs(v128& d, const v128& a, const v128& b, const v128& c) { msum<int16_t, int16_t, int32_t, true>(d, a, b, c); }

// ---- sum across -----------------------------------------------------------
inline void vsum4ubs(v128& d, const v128& a, const v128& b) {
  v128 r;
  for (int i = 0; i < 4; ++i) {
    int64_t s = get<uint32_t>(b, i);
    for (int j = 0; j < 4; ++j) s += get<uint8_t>(a, 4 * i + j);
    set<uint32_t>(r, i, sat<uint32_t>(s));
  }
  d = r;
}
inline void vsum4sbs(v128& d, const v128& a, const v128& b) {
  v128 r;
  for (int i = 0; i < 4; ++i) {
    int64_t s = get<int32_t>(b, i);
    for (int j = 0; j < 4; ++j) s += get<int8_t>(a, 4 * i + j);
    set<int32_t>(r, i, sat<int32_t>(s));
  }
  d = r;
}
inline void vsum4shs(v128& d, const v128& a, const v128& b) {
  v128 r;
  for (int i = 0; i < 4; ++i)
    set<int32_t>(r, i, sat<int32_t>(int64_t(get<int32_t>(b, i)) + get<int16_t>(a, 2 * i) + get<int16_t>(a, 2 * i + 1)));
  d = r;
}
inline void vsum2sws(v128& d, const v128& a, const v128& b) {
  v128 r{};
  for (int i = 0; i < 2; ++i)
    set<int32_t>(r, 2 * i + 1, sat<int32_t>(int64_t(get<int32_t>(a, 2 * i)) + get<int32_t>(a, 2 * i + 1) + get<int32_t>(b, 2 * i + 1)));
  d = r;
}
inline void vsumsws(v128& d, const v128& a, const v128& b) {
  v128 r{};
  int64_t s = get<int32_t>(b, 3);
  for (int i = 0; i < 4; ++i) s += get<int32_t>(a, i);
  set<int32_t>(r, 3, sat<int32_t>(s));
  d = r;
}

// ---- select / permute / shift-double ---------------------------------------
inline void vsel(v128& d, const v128& a, const v128& b, const v128& c) {
  v128 r;
  for (int k = 0; k < 16; ++k) r.b[k] = uint8_t((a.b[k] & ~c.b[k]) | (b.b[k] & c.b[k]));
  d = r;
}
inline void vperm(v128& d, const v128& a, const v128& b, const v128& c) {
  v128 r;
  for (int i = 0; i < 16; ++i) {
    const int idx = get<uint8_t>(c, i) & 31;
    set<uint8_t>(r, i, idx < 16 ? get<uint8_t>(a, idx) : get<uint8_t>(b, idx - 16));
  }
  d = r;
}
inline void vsldoi(v128& d, const v128& a, const v128& b, unsigned sh) {
  v128 r;
  for (int i = 0; i < 16; ++i) {
    const int idx = i + int(sh & 15);
    set<uint8_t>(r, i, idx < 16 ? get<uint8_t>(a, idx) : get<uint8_t>(b, idx - 16));
  }
  d = r;
}

// ---- floating point -------------------------------------------------------
VMX_BIN(vaddfp, float, a + b)
VMX_BIN(vsubfp, float, a - b)
VMX_BIN(vmaxfp, float, std::isnan(a) || std::isnan(b) ? a + b : std::max(a, b))
VMX_BIN(vminfp, float, std::isnan(a) || std::isnan(b) ? a + b : std::min(a, b))
VMX_UN(vrefp, float, 1.0f / b)
VMX_UN(vrsqrtefp, float, 1.0f / std::sqrt(b))
VMX_UN(vexptefp, float, std::exp2(b))
VMX_UN(vlogefp, float, std::log2(b))
VMX_UN(vrfin, float, std::nearbyint(b))
VMX_UN(vrfiz, float, std::trunc(b))
VMX_UN(vrfip, float, std::ceil(b))
VMX_UN(vrfim, float, std::floor(b))

inline void vmaddfp(v128& d, const v128& a, const v128& b, const v128& c) {
  v128 r;
  for (int i = 0; i < 4; ++i) set<float>(r, i, std::fma(get<float>(a, i), get<float>(c, i), get<float>(b, i)));
  d = r;
}
inline void vnmsubfp(v128& d, const v128& a, const v128& b, const v128& c) {
  v128 r;
  for (int i = 0; i < 4; ++i) set<float>(r, i, -std::fma(get<float>(a, i), get<float>(c, i), -get<float>(b, i)));
  d = r;
}
inline void vcfux(v128& d, const v128& b, unsigned u) {
  v128 r;
  for (int i = 0; i < 4; ++i) set<float>(r, i, float(std::ldexp(double(get<uint32_t>(b, i)), -int(u))));
  d = r;
}
inline void vcfsx(v128& d, const v128& b, unsigned u) {
  v128 r;
  for (int i = 0; i < 4; ++i) set<float>(r, i, float(std::ldexp(double(get<int32_t>(b, i)), -int(u))));
  d = r;
}
inline void vctuxs(v128& d, const v128& b, unsigned u) {
  v128 r;
  for (int i = 0; i < 4; ++i) set<uint32_t>(r, i, satf<uint32_t>(std::trunc(std::ldexp(double(get<float>(b, i)), int(u)))));
  d = r;
}
inline void vctsxs(v128& d, const v128& b, unsigned u) {
  v128 r;
  for (int i = 0; i < 4; ++i) set<int32_t>(r, i, satf<int32_t>(std::trunc(std::ldexp(double(get<float>(b, i)), int(u)))));
  d = r;
}

// ---- compares ---------------------------------------------------------------
VMX_CMP(vcmpequb, uint8_t, a == b)
VMX_CMP(vcmpequh, uint16_t, a == b)
VMX_CMP(vcmpequw, uint32_t, a == b)
VMX_CMP(vcmpeqfp, float, a == b)
VMX_CMP(vcmpgefp, float, a >= b)
VMX_CMP(vcmpgtub, uint8_t, a > b)
VMX_CMP(vcmpgtuh, uint16_t, a > b)
VMX_CMP(vcmpgtuw, uint32_t, a > b)
VMX_CMP(vcmpgtfp, float, a > b)
VMX_CMP(vcmpgtsb, int8_t, a > b)
VMX_CMP(vcmpgtsh, int16_t, a > b)
VMX_CMP(vcmpgtsw, int32_t, a > b)
inline void vcmpbfp(v128& d, const v128& va, const v128& vb) {
  v128 r;
  for (int i = 0; i < 4; ++i) {
    const float a = get<float>(va, i), b = get<float>(vb, i);
    const bool nan = std::isnan(a) || std::isnan(b);
    uint32_t v = 0;
    if (nan || !(a <= b)) v |= 0x80000000u;
    if (nan || !(a >= -b)) v |= 0x40000000u;
    set<uint32_t>(r, i, v);
  }
  d = r;
}

// CR6 after a recording compare: lt = all elements true, eq = all false.
inline CRField cr6_from(const v128& v) {
  bool all = true, none = true;
  for (int k = 0; k < 16; ++k) { all &= v.b[k] == 0xFF; none &= v.b[k] == 0; }
  return CRField{uint8_t(all), 0, uint8_t(none), 0};
}
// vcmpbfp.: eq = all elements within bounds.
inline CRField cr6_bounds(const v128& v) {
  bool none = true;
  for (int k = 0; k < 16; ++k) none &= v.b[k] == 0;
  return CRField{0, 0, uint8_t(none), 0};
}

#undef VMX_BIN
#undef VMX_UN
#undef VMX_CMP

}  // namespace ppu::vmx
