#pragma once
// Runtime support included by every generated translation unit.
//
// Model: guest memory is a 4 GiB reservation addressed with 32-bit effective
// addresses (as on the PS3). Guest data is big-endian and byte-swapped on
// every access. A recompiled function is a native C++ function that takes the
// PPU register file and returns when the guest would `blr`.
//
// Assumption: guest code returns with `blr`. Code that rewrites LR/stack to
// return elsewhere (setjmp/longjmp, coroutines) is not modelled yet.

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#if defined(_MSC_VER)
#include <intrin.h>
#define PPU_BSWAP16(x) _byteswap_ushort(x)
#define PPU_BSWAP32(x) _byteswap_ulong(x)
#define PPU_BSWAP64(x) _byteswap_uint64(x)
#else
#define PPU_BSWAP16(x) __builtin_bswap16(x)
#define PPU_BSWAP32(x) __builtin_bswap32(x)
#define PPU_BSWAP64(x) __builtin_bswap64(x)
#endif

namespace ppu {

// A 128-bit vector register. Stored byte-reversed relative to guest memory, so
// guest element i of type T lives at host bytes [16 - (i+1)*sizeof(T), 16 - i*sizeof(T))
// and every element is in host (little-endian) byte order.
struct alignas(16) v128 {
  uint8_t b[16];
};

struct CRField {
  uint8_t lt = 0, gt = 0, eq = 0, so = 0;
};

struct Context {
  uint64_t r[32]{};
  double f[32]{};
  v128 v[32]{};
  CRField cr[8]{};
  uint64_t lr = 0;
  uint64_t ctr = 0;
  uint8_t xer_so = 0, xer_ov = 0, xer_ca = 0;
  uint8_t xer_bc = 0;        // byte count for string instructions
  uint32_t fpscr = 0;        // stored, not enforced (rounding is always round-to-nearest)
  uint32_t vscr = 0x10000;   // NJ set, as on the Cell
  uint32_t vrsave = 0;
  // lwarx/ldarx reservation
  uint32_t res_addr = 0;
  uint64_t res_val = 0;
  bool res_valid = false;
  uint8_t* base = nullptr;   // start of the 4 GiB guest address space
};

using Fn = void (*)(Context&);

// ---- memory ---------------------------------------------------------------
inline uint8_t ld8(const uint8_t* b, uint32_t ea) { return b[ea]; }
inline uint16_t ld16(const uint8_t* b, uint32_t ea) { uint16_t v; std::memcpy(&v, b + ea, 2); return PPU_BSWAP16(v); }
inline uint32_t ld32(const uint8_t* b, uint32_t ea) { uint32_t v; std::memcpy(&v, b + ea, 4); return PPU_BSWAP32(v); }
inline uint64_t ld64(const uint8_t* b, uint32_t ea) { uint64_t v; std::memcpy(&v, b + ea, 8); return PPU_BSWAP64(v); }
inline void st8(uint8_t* b, uint32_t ea, uint8_t v) { b[ea] = v; }
inline void st16(uint8_t* b, uint32_t ea, uint16_t v) { v = PPU_BSWAP16(v); std::memcpy(b + ea, &v, 2); }
inline void st32(uint8_t* b, uint32_t ea, uint32_t v) { v = PPU_BSWAP32(v); std::memcpy(b + ea, &v, 4); }
inline void st64(uint8_t* b, uint32_t ea, uint64_t v) { v = PPU_BSWAP64(v); std::memcpy(b + ea, &v, 8); }
// byte-reversed (little-endian) accesses: lwbrx & co.
inline uint16_t ld16r(const uint8_t* b, uint32_t ea) { uint16_t v; std::memcpy(&v, b + ea, 2); return v; }
inline uint32_t ld32r(const uint8_t* b, uint32_t ea) { uint32_t v; std::memcpy(&v, b + ea, 4); return v; }
inline uint64_t ld64r(const uint8_t* b, uint32_t ea) { uint64_t v; std::memcpy(&v, b + ea, 8); return v; }
inline void st16r(uint8_t* b, uint32_t ea, uint16_t v) { std::memcpy(b + ea, &v, 2); }
inline void st32r(uint8_t* b, uint32_t ea, uint32_t v) { std::memcpy(b + ea, &v, 4); }
inline void st64r(uint8_t* b, uint32_t ea, uint64_t v) { std::memcpy(b + ea, &v, 8); }

inline double ldf32(const uint8_t* b, uint32_t ea) { return double(std::bit_cast<float>(ld32(b, ea))); }
inline double ldf64(const uint8_t* b, uint32_t ea) { return std::bit_cast<double>(ld64(b, ea)); }
inline void stf32(uint8_t* b, uint32_t ea, double v) { st32(b, ea, std::bit_cast<uint32_t>(float(v))); }
inline void stf64(uint8_t* b, uint32_t ea, double v) { st64(b, ea, std::bit_cast<uint64_t>(v)); }

// dcbz: the Cell has 128-byte cache lines.
inline void dcbz(uint8_t* b, uint32_t ea) { std::memset(b + (ea & ~127u), 0, 128); }

// ---- atomics (lwarx/stwcx., ldarx/stdcx.) ---------------------------------
inline uint64_t lwarx(Context& c, uint32_t ea) {
  c.res_addr = ea; c.res_valid = true;
  c.res_val = ld32(c.base, ea);
  return c.res_val;
}
inline uint64_t ldarx(Context& c, uint32_t ea) {
  c.res_addr = ea; c.res_valid = true;
  c.res_val = ld64(c.base, ea);
  return c.res_val;
}
inline void stwcx(Context& c, uint32_t ea, uint32_t v) {
  bool ok = false;
  if (c.res_valid && c.res_addr == ea) {
    std::atomic_ref<uint32_t> ref(*reinterpret_cast<uint32_t*>(c.base + ea));
    uint32_t expected = PPU_BSWAP32(uint32_t(c.res_val));
    ok = ref.compare_exchange_strong(expected, PPU_BSWAP32(v));
  }
  c.res_valid = false;
  c.cr[0] = CRField{0, 0, uint8_t(ok), c.xer_so};
}
inline void stdcx(Context& c, uint32_t ea, uint64_t v) {
  bool ok = false;
  if (c.res_valid && c.res_addr == ea) {
    std::atomic_ref<uint64_t> ref(*reinterpret_cast<uint64_t*>(c.base + ea));
    uint64_t expected = PPU_BSWAP64(c.res_val);
    ok = ref.compare_exchange_strong(expected, PPU_BSWAP64(v));
  }
  c.res_valid = false;
  c.cr[0] = CRField{0, 0, uint8_t(ok), c.xer_so};
}

// ---- condition register ---------------------------------------------------
inline void cmp_s(Context& c, int f, int64_t a, int64_t b) { c.cr[f] = CRField{uint8_t(a < b), uint8_t(a > b), uint8_t(a == b), c.xer_so}; }
inline void cmp_u(Context& c, int f, uint64_t a, uint64_t b) { c.cr[f] = CRField{uint8_t(a < b), uint8_t(a > b), uint8_t(a == b), c.xer_so}; }
inline void set_cr0(Context& c, uint64_t v) { cmp_s(c, 0, int64_t(v), 0); }
inline void fcmp(Context& c, int f, double a, double b) {
  const bool un = std::isnan(a) || std::isnan(b);
  c.cr[f] = CRField{uint8_t(!un && a < b), uint8_t(!un && a > b), uint8_t(!un && a == b), uint8_t(un)};
}
inline uint8_t& cr_bit(Context& c, unsigned bit) {
  CRField& f = c.cr[bit / 4];
  switch (bit % 4) { case 0: return f.lt; case 1: return f.gt; case 2: return f.eq; default: return f.so; }
}
inline uint32_t get_cr(const Context& c) {
  uint32_t v = 0;
  for (int i = 0; i < 8; ++i) {
    const CRField& f = c.cr[i];
    v |= uint32_t((f.lt << 3) | (f.gt << 2) | (f.eq << 1) | f.so) << (4 * (7 - i));
  }
  return v;
}
inline void set_cr(Context& c, uint32_t v, uint32_t crm) {
  for (int i = 0; i < 8; ++i) {
    if (!(crm & (0x80u >> i))) continue;
    const uint32_t n = (v >> (4 * (7 - i))) & 15;
    c.cr[i] = CRField{uint8_t((n >> 3) & 1), uint8_t((n >> 2) & 1), uint8_t((n >> 1) & 1), uint8_t(n & 1)};
  }
}
inline uint64_t get_xer(const Context& c) {
  return (uint64_t(c.xer_so) << 31) | (uint64_t(c.xer_ov) << 30) | (uint64_t(c.xer_ca) << 29) | c.xer_bc;
}
inline void set_xer(Context& c, uint64_t v) {
  c.xer_so = (v >> 31) & 1; c.xer_ov = (v >> 30) & 1; c.xer_ca = (v >> 29) & 1; c.xer_bc = uint8_t(v & 0x7F);
}

// ---- integer helpers ------------------------------------------------------
inline uint64_t dup32(uint32_t v) { return (uint64_t(v) << 32) | v; }
inline uint64_t rotl64(uint64_t v, unsigned n) { return std::rotl(v, int(n & 63)); }
inline uint64_t rotl32x(uint64_t v, unsigned n) { return rotl64(dup32(uint32_t(v)), n & 31); }

// 64-bit mask with PPC bit numbering (bit 0 = MSB), wrapping when mb > me.
inline uint64_t mask64(unsigned mb, unsigned me) {
  const uint64_t a = ~0ull >> mb, b = ~0ull << (63 - me);
  return mb <= me ? (a & b) : (a | b);
}

// a + b + cin with carry-out into XER[CA]. Every add/subtract-with-carry form maps to this.
inline uint64_t add3(Context& c, uint64_t a, uint64_t b, uint64_t cin) {
  const uint64_t t = a + cin;
  const uint64_t r = t + b;
  c.xer_ca = uint8_t((t < a) | (r < t));
  return r;
}
// Overflow (XER[OV], sticky XER[SO]) for the OE forms of add/subtract.
inline void set_ov_add(Context& c, uint64_t a, uint64_t b, uint64_t r) {
  const bool ov = ((~(a ^ b) & (a ^ r)) >> 63) & 1;
  c.xer_ov = uint8_t(ov); c.xer_so = uint8_t(c.xer_so | uint8_t(ov));
}

inline uint64_t sraw(Context& c, uint64_t rs, unsigned n) {
  const int32_t v = int32_t(rs);
  if (n & 32) { c.xer_ca = v < 0; return uint64_t(int64_t(v < 0 ? -1 : 0)); }
  n &= 31;
  const int32_t r = v >> n;
  c.xer_ca = (v < 0) && n && (uint32_t(v) & ((1u << n) - 1));
  return uint64_t(int64_t(r));
}
inline uint64_t srad(Context& c, uint64_t rs, unsigned n) {
  const int64_t v = int64_t(rs);
  if (n & 64) { c.xer_ca = v < 0; return uint64_t(v < 0 ? -1 : 0); }
  n &= 63;
  const int64_t r = v >> n;
  c.xer_ca = (v < 0) && n && (uint64_t(v) & ((1ull << n) - 1));
  return uint64_t(r);
}

inline uint64_t mulh64(int64_t a, int64_t b) {
#if defined(_MSC_VER)
  return uint64_t(__mulh(a, b));
#else
  return uint64_t((__int128(a) * b) >> 64);
#endif
}
inline uint64_t umulh64(uint64_t a, uint64_t b) {
#if defined(_MSC_VER)
  return __umulh(a, b);
#else
  return uint64_t((unsigned __int128)a * b >> 64);
#endif
}
inline uint64_t divw(int32_t a, int32_t b) {
  return (b == 0 || (a == INT32_MIN && b == -1)) ? 0 : uint64_t(uint32_t(a / b));
}
inline uint64_t divd(int64_t a, int64_t b) {
  return (b == 0 || (a == INT64_MIN && b == -1)) ? 0 : uint64_t(a / b);
}

// TO field: 0x10 lt, 0x08 gt, 0x04 eq, 0x02 ltu, 0x01 gtu
inline bool trap_cond(unsigned to, int64_t a, int64_t b) {
  return ((to & 0x10) && a < b) || ((to & 0x08) && a > b) || ((to & 0x04) && a == b) ||
         ((to & 0x02) && uint64_t(a) < uint64_t(b)) || ((to & 0x01) && uint64_t(a) > uint64_t(b));
}

// ---- floating point -------------------------------------------------------
inline double round_single(double v) { return double(float(v)); }
template <class I>
inline I fp_to_int(double v, bool truncate) {
  if (std::isnan(v)) return std::numeric_limits<I>::min();
  const double r = truncate ? std::trunc(v) : std::nearbyint(v);
  if (r >= double(std::numeric_limits<I>::max())) return std::numeric_limits<I>::max();
  if (r <= double(std::numeric_limits<I>::min())) return std::numeric_limits<I>::min();
  return I(r);
}
inline double fctiw(double v, bool z) { return std::bit_cast<double>(uint64_t(uint32_t(fp_to_int<int32_t>(v, z)))); }
inline double fctid(double v, bool z) { return std::bit_cast<double>(uint64_t(fp_to_int<int64_t>(v, z))); }
inline double fcfid(double v) { return double(std::bit_cast<int64_t>(v)); }
inline double fsel(double a, double c, double b) { return a >= 0.0 ? c : b; }
inline uint32_t fp_bits_lo(double v) { return uint32_t(std::bit_cast<uint64_t>(v)); }

// ---- time base (79.8 MHz on PS3) ------------------------------------------
inline uint64_t timebase() {
  using namespace std::chrono;
  const uint64_t ns = uint64_t(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
  return ns / 1000 * 798 / 10 + (ns % 1000) * 798 / 10000;
}

// ---- function table, hooks and dispatch -----------------------------------
// Every recompiled function has a slot (a function pointer). Direct guest calls
// go through the slot, so hooking an address redirects every caller.
struct FnEntry {
  Fn current = nullptr;
  Fn original = nullptr;
  Fn* slot = nullptr;
};

inline std::unordered_map<uint32_t, FnEntry>& function_table() {
  static std::unordered_map<uint32_t, FnEntry> t;
  return t;
}
inline void register_function(uint32_t addr, Fn fn, Fn* slot) {
  function_table()[addr] = FnEntry{fn, fn, slot};
  if (slot) *slot = fn;
}
inline Fn find_function(uint32_t addr) {
  auto& t = function_table();
  auto it = t.find(addr);
  return it == t.end() ? nullptr : it->second.current;
}
// Replaces the function at `addr`. Returns the previous implementation (call it to
// chain to the original), or nullptr if no function is registered there.
inline Fn hook_function(uint32_t addr, Fn replacement) {
  auto& t = function_table();
  auto it = t.find(addr);
  if (it == t.end()) return nullptr;
  Fn prev = it->second.current;
  it->second.current = replacement;
  if (it->second.slot) *it->second.slot = replacement;
  return prev;
}
inline void unhook_function(uint32_t addr) {
  auto& t = function_table();
  auto it = t.find(addr);
  if (it == t.end()) return;
  it->second.current = it->second.original;
  if (it->second.slot) *it->second.slot = it->second.original;
}

inline void call_indirect(Context& c, uint64_t addr) {
  if (Fn f = find_function(uint32_t(addr))) return f(c);
  char buf[96];
  std::snprintf(buf, sizeof buf, "call to unrecompiled guest address 0x%llX", (unsigned long long)addr);
  throw std::runtime_error(buf);
}

// ---- traps / system calls -------------------------------------------------
[[noreturn]] inline void unimplemented(Context&, uint32_t addr, uint32_t raw) {
  char buf[96];
  std::snprintf(buf, sizeof buf, "unimplemented instruction 0x%08X at 0x%08X", raw, addr);
  throw std::runtime_error(buf);
}
[[noreturn]] inline void trap(Context&, uint32_t addr) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "guest trap at 0x%08X", addr);
  throw std::runtime_error(buf);
}

// LV2 system call hook (syscall number in r11). The host installs a handler.
inline void (*syscall_handler)(Context&) = nullptr;
inline void syscall(Context& c, uint32_t addr) {
  if (!syscall_handler) unimplemented(c, addr, 0x44000002);
  syscall_handler(c);
}

}  // namespace ppu

#include "ppu_vmx.h"

// Implemented by the generated code.
void ppu_register_functions();
