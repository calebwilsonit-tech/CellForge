// SPU interpreter tests: hand-encoded programs checked against known results.
#include <array>
#include <bit>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "spu_interp.h"

namespace {

int g_fail = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
  } while (0)

uint32_t RR(uint32_t op, uint32_t rb, uint32_t ra, uint32_t rt) { return op << 21 | rb << 14 | ra << 7 | rt; }
uint32_t RI7(uint32_t op, int32_t i7, uint32_t ra, uint32_t rt) { return op << 21 | (uint32_t(i7) & 0x7F) << 14 | ra << 7 | rt; }
uint32_t RI8(uint32_t op, uint32_t i8, uint32_t ra, uint32_t rt) { return op << 22 | (i8 & 0xFF) << 14 | ra << 7 | rt; }
uint32_t RI10(uint32_t op, int32_t i10, uint32_t ra, uint32_t rt) { return op << 24 | (uint32_t(i10) & 0x3FF) << 14 | ra << 7 | rt; }
uint32_t RI16(uint32_t op, int32_t i16, uint32_t rt) { return op << 23 | (uint32_t(i16) & 0xFFFF) << 7 | rt; }
uint32_t RRR(uint32_t op, uint32_t rt, uint32_t rb, uint32_t ra, uint32_t rc) { return op << 28 | rt << 21 | rb << 14 | ra << 7 | rc; }

struct TestHost : spu::SpuHost {
  std::vector<uint32_t> out_mbox;
  std::vector<uint32_t> in_mbox{0xCAFE};
  uint32_t read_channel(spu::SpuCore&, uint32_t ch) override {
    if (ch == spu::SPU_RdInMbox && !in_mbox.empty()) { uint32_t v = in_mbox.front(); in_mbox.erase(in_mbox.begin()); return v; }
    return 0;
  }
  void write_channel(spu::SpuCore&, uint32_t ch, uint32_t v) override {
    if (ch == spu::SPU_WrOutMbox) out_mbox.push_back(v);
  }
  uint32_t channel_count(spu::SpuCore&, uint32_t) override { return 1; }
  bool stop(spu::SpuCore&, uint32_t) override { return false; }
};

void load(spu::SpuCore& s, const std::vector<uint32_t>& prog) {
  for (size_t i = 0; i < prog.size(); ++i) s.set_ls32(uint32_t(i * 4), prog[i]);
  s.pc = 0;
}

void test_program() {
  TestHost host;
  spu::SpuCore s(&host);
  load(s, {
      RI16(0x081, 0, 1),            // il r1,0
      RI16(0x081, 10, 2),           // il r2,10
      RR(0x0C0, 2, 1, 1),           // loop: a r1,r1,r2
      RI10(0x1C, -1, 2, 2),         // ai r2,r2,-1
      RI16(0x042, -2, 2),           // brnz r2,loop
      RI16(0x082, 0x1234, 3),       // ilhu r3,0x1234
      RI16(0x0C1, 0x5678, 3),       // iohl r3,0x5678
      RI16(0x081, 0, 8),            // il r8,0
      RI7(0x1F6, 4, 8, 7),          // cwd r7,4(r8)
      RI16(0x081, 0, 10),           // il r10,0
      RRR(0xB, 9, 10, 3, 7),        // shufb r9,r3,r10,r7
      RR(0x10D, 0, spu::SPU_WrOutMbox, 1),  // wrch $ch28,r1
      RI16(0x081, 3, 13),           // il r13,3
      RI8(0x1DA, 155, 13, 14),      // csflt r14,r13,155  -> 3.0f
      RR(0x2C6, 14, 14, 15),        // fm r15,r14,r14     -> 9.0f
      RI8(0x1D8, 173, 15, 16),      // cflts r16,r15,173  -> 9
      RI16(0x065, 0xF000, 17),      // fsmbi r17,0xF000
      RI7(0x1FC, 4, 3, 18),         // rotqbyi r18,r9,4 (on r3: splat, unchanged)
      RI7(0x1FF, 4, 9, 19),         // shlqbyi r19,r9,4   -> word0 = 0x12345678
      RR(0x00D, 0, spu::SPU_RdInMbox, 20),  // rdch r20,$ch29
      RR(0x2A5, 0, 8, 21),          // clz r21,r8         -> 32
      RI10(0x34, 1, 8, 22),         // lqd r22,16(r8)     (= instructions 4..7)
      RR(0x000, 0, 0, 0x2A),        // stop 0x2A
  });
  const uint32_t code = s.run();
  CHECK(code == 0x2A);
  CHECK(s.gpr[1].w(0) == 55 && s.gpr[1].w(3) == 55);
  CHECK(s.gpr[3].w(0) == 0x12345678 && s.gpr[3].w(2) == 0x12345678);
  CHECK(s.gpr[9].w(0) == 0 && s.gpr[9].w(1) == 0x12345678 && s.gpr[9].w(2) == 0 && s.gpr[9].w(3) == 0);
  CHECK(host.out_mbox.size() == 1 && host.out_mbox[0] == 55);
  CHECK(s.gpr[14].f(0) == 3.0f);
  CHECK(s.gpr[15].f(1) == 9.0f);
  CHECK(s.gpr[16].w(2) == 9);
  CHECK(s.gpr[17].w(0) == 0xFFFFFFFF && s.gpr[17].w(1) == 0);
  CHECK(s.gpr[19].w(0) == 0x12345678 && s.gpr[19].w(3) == 0);
  CHECK(s.gpr[20].w(0) == 0xCAFE);
  CHECK(s.gpr[21].w(0) == 32);
  CHECK(s.gpr[22].w(0) == s.ls32(16));
}

void test_branch_link_and_shifts() {
  TestHost host;
  spu::SpuCore s(&host);
  load(s, {
      RI16(0x066, 3, 0),            // 0: brsl r0, +3 -> 12
      RI16(0x081, 7, 5),            // 4: il r5,7   (after return)
      RR(0x000, 0, 0, 1),           // 8: stop 1
      RI16(0x081, -8, 6),           // 12: il r6,-8
      RI7(0x07A, -2, 6, 7),         // 16: rotmai r7,r6,-2  (arith >> 2) -> -2
      RI7(0x079, -2, 6, 8),         // 20: rotmi r8,r6,-2   (logical >> 2)
      RI7(0x07B, 4, 6, 9),          // 24: shli r9,r6,4 -> -128
      RR(0x1A8, 0, 0, 0),           // 28: bi r0
  });
  CHECK(s.run() == 1);
  CHECK(s.gpr[5].w(0) == 7);
  CHECK(int32_t(s.gpr[7].w(0)) == -2);
  CHECK(s.gpr[8].w(0) == (0xFFFFFFF8u >> 2));
  CHECK(int32_t(s.gpr[9].w(1)) == -128);
}

// ---- randomized check of element-wise and byte-order operations ----------------
// Each op runs on random registers and is compared with a scalar reference
// written directly from the ISA definitions on big-endian byte arrays.
using Bytes = std::array<uint8_t, 16>;
Bytes bytes_of(const spu::V128& v) { Bytes b; for (int i = 0; i < 16; ++i) b[i] = v.byte(i); return b; }
uint32_t W(const Bytes& b, int i) { return uint32_t(b[4 * i]) << 24 | b[4 * i + 1] << 16 | b[4 * i + 2] << 8 | b[4 * i + 3]; }
uint16_t H(const Bytes& b, int i) { return uint16_t(b[2 * i] << 8 | b[2 * i + 1]); }
void setW(Bytes& b, int i, uint32_t v) { for (int k = 0; k < 4; ++k) b[4 * i + k] = uint8_t(v >> (24 - 8 * k)); }
void setH(Bytes& b, int i, uint16_t v) { b[2 * i] = uint8_t(v >> 8); b[2 * i + 1] = uint8_t(v); }

uint64_t g_rng = 0x1234567887654321ull;
uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return uint32_t(g_rng); }

struct Case {
  const char* name;
  uint32_t insn;  // rt=3 (RRR: rt=3, rc=6), ra=4, rb=5
  std::function<Bytes(const Bytes& a, const Bytes& b, const Bytes& c, const Bytes& t)> ref;
};

template <class F> Bytes w_op(const Bytes& a, const Bytes& b, F f) { Bytes r{}; for (int i = 0; i < 4; ++i) setW(r, i, f(W(a, i), W(b, i))); return r; }
template <class F> Bytes h_op(const Bytes& a, const Bytes& b, F f) { Bytes r{}; for (int i = 0; i < 8; ++i) setH(r, i, uint16_t(f(H(a, i), H(b, i)))); return r; }
template <class F> Bytes b_op(const Bytes& a, const Bytes& b, F f) { Bytes r{}; for (int i = 0; i < 16; ++i) r[i] = uint8_t(f(a[i], b[i])); return r; }

void test_randomized() {
  const uint32_t rr = 0, i7v = 0x2B, i10v = 0x2C5;  // i10 = -315
  const int32_t i10 = int32_t(i10v << 22) >> 22;
  auto RRx = [&](uint32_t op) { return RR(op, 5, 4, 3); };
  auto I7 = [&](uint32_t op) { return RI7(op, int32_t(i7v), 4, 3); };
  auto I10 = [&](uint32_t op) { return RI10(op, i10, 4, 3); };
  (void)rr;
  std::vector<Case> cases = {
      {"a", RRx(0x0C0), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return x + y; }); }},
      {"ah", RRx(0x0C8), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { return x + y; }); }},
      {"sf", RRx(0x040), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return y - x; }); }},
      {"sfh", RRx(0x048), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { return y - x; }); }},
      {"cg", RRx(0x0C2), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return uint32_t((uint64_t(x) + y) >> 32); }); }},
      {"bg", RRx(0x042), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return y >= x ? 1u : 0u; }); }},
      {"mpy", RRx(0x3C4), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return uint32_t(int32_t(int16_t(x)) * int16_t(y)); }); }},
      {"mpyu", RRx(0x3CC), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return (x & 0xFFFF) * (y & 0xFFFF); }); }},
      {"mpyh", RRx(0x3C5), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return ((x >> 16) * (y & 0xFFFF)) << 16; }); }},
      {"mpys", RRx(0x3C7), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return uint32_t((int32_t(int16_t(x)) * int16_t(y)) >> 16); }); }},
      {"mpyhh", RRx(0x3C6), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return uint32_t(int32_t(int16_t(x >> 16)) * int16_t(y >> 16)); }); }},
      {"mpyhhu", RRx(0x3CE), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return (x >> 16) * (y >> 16); }); }},
      {"clz", RRx(0x2A5), [](auto& a, auto&, auto&, auto&) { return w_op(a, a, [](uint32_t x, uint32_t) { return uint32_t(std::countl_zero(x)); }); }},
      {"and", RRx(0x0C1), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return x & y; }); }},
      {"andc", RRx(0x2C1), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return x & ~y; }); }},
      {"orc", RRx(0x2C9), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return x | ~y; }); }},
      {"nand", RRx(0x0C9), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return ~(x & y); }); }},
      {"nor", RRx(0x049), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return ~(x | y); }); }},
      {"eqv", RRx(0x249), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return ~(x ^ y); }); }},
      {"ceqh", RRx(0x3C8), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { return x == y ? 0xFFFF : 0; }); }},
      {"cgt", RRx(0x240), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return int32_t(x) > int32_t(y) ? ~0u : 0u; }); }},
      {"cgth", RRx(0x248), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { return int16_t(x) > int16_t(y) ? 0xFFFF : 0; }); }},
      {"cgtb", RRx(0x250), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return int8_t(x) > int8_t(y) ? 0xFF : 0; }); }},
      {"clgt", RRx(0x2C0), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return x > y ? ~0u : 0u; }); }},
      {"clgth", RRx(0x2C8), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { return x > y ? 0xFFFF : 0; }); }},
      {"clgtb", RRx(0x2D0), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return x > y ? 0xFF : 0; }); }},
      {"shl", RRx(0x05B), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { unsigned s = y & 0x3F; return s > 31 ? 0u : x << s; }); }},
      {"shlh", RRx(0x05F), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { unsigned s = y & 0x1F; return s > 15 ? 0 : x << s; }); }},
      {"rot", RRx(0x058), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { return std::rotl(x, int(y & 31)); }); }},
      {"roth", RRx(0x05C), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { return std::rotl(x, int(y & 15)); }); }},
      {"rotm", RRx(0x059), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { unsigned s = (0u - y) & 0x3F; return s > 31 ? 0u : x >> s; }); }},
      {"rothm", RRx(0x05D), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { unsigned s = (0u - y) & 0x1F; return s > 15 ? 0 : x >> s; }); }},
      {"rotma", RRx(0x05A), [](auto& a, auto& b, auto&, auto&) { return w_op(a, b, [](uint32_t x, uint32_t y) { unsigned s = (0u - y) & 0x3F; return uint32_t(int32_t(x) >> (s > 31 ? 31 : s)); }); }},
      {"rotmah", RRx(0x05E), [](auto& a, auto& b, auto&, auto&) { return h_op(a, b, [](uint16_t x, uint16_t y) { unsigned s = (0u - y) & 0x1F; return uint16_t(int16_t(x) >> (s > 15 ? 15 : s)); }); }},
      {"avgb", RRx(0x0D3), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return (x + y + 1) >> 1; }); }},
      {"absdb", RRx(0x053), [](auto& a, auto& b, auto&, auto&) { return b_op(a, b, [](uint8_t x, uint8_t y) { return x > y ? x - y : y - x; }); }},
      {"cntb", RRx(0x2B4), [](auto& a, auto&, auto&, auto&) { return b_op(a, a, [](uint8_t x, uint8_t) { return std::popcount(x); }); }},
      {"xsbh", RRx(0x2B6), [](auto& a, auto&, auto&, auto&) { return h_op(a, a, [](uint16_t x, uint16_t) { return uint16_t(int16_t(int8_t(x & 0xFF))); }); }},
      {"xshw", RRx(0x2AE), [](auto& a, auto&, auto&, auto&) { return w_op(a, a, [](uint32_t x, uint32_t) { return uint32_t(int32_t(int16_t(x))); }); }},
      {"sumb", RRx(0x253), [](auto& a, auto& b, auto&, auto&) {
         Bytes r{};
         for (int i = 0; i < 4; ++i) {
           const unsigned sb = b[4 * i] + b[4 * i + 1] + b[4 * i + 2] + b[4 * i + 3], sa = a[4 * i] + a[4 * i + 1] + a[4 * i + 2] + a[4 * i + 3];
           setW(r, i, sb << 16 | sa);
         }
         return r;
       }},
      {"shlqby", RRx(0x1DF), [](auto& a, auto& b, auto&, auto&) { Bytes r{}; unsigned n = W(b, 0) & 0x1F; for (unsigned i = 0; i < 16; ++i) r[i] = (n < 16 && i + n < 16) ? a[i + n] : 0; return r; }},
      {"rotqby", RRx(0x1DC), [](auto& a, auto& b, auto&, auto&) { Bytes r{}; unsigned n = W(b, 0) & 0xF; for (unsigned i = 0; i < 16; ++i) r[i] = a[(i + n) & 15]; return r; }},
      {"rotqmby", RRx(0x1DD), [](auto& a, auto& b, auto&, auto&) { Bytes r{}; unsigned n = (0u - W(b, 0)) & 0x1F; for (unsigned i = 0; i < 16; ++i) r[i] = (n < 16 && i >= n) ? a[i - n] : 0; return r; }},
      {"shlqbi", RRx(0x1DB), [](auto& a, auto& b, auto&, auto&) { Bytes r{}; unsigned n = W(b, 0) & 7; for (int i = 0; i < 16; ++i) r[i] = uint8_t(a[i] << n | (n && i < 15 ? a[i + 1] >> (8 - n) : 0)); return r; }},
      {"rotqbi", RRx(0x1D8), [](auto& a, auto& b, auto&, auto&) { Bytes r{}; unsigned n = W(b, 0) & 7; for (int i = 0; i < 16; ++i) r[i] = uint8_t(a[i] << n | (n ? a[(i + 1) & 15] >> (8 - n) : 0)); return r; }},
      {"rotqmbi", RRx(0x1D9), [](auto& a, auto& b, auto&, auto&) { Bytes r{}; unsigned n = (0u - W(b, 0)) & 7; for (int i = 0; i < 16; ++i) r[i] = uint8_t(a[i] >> n | (n && i > 0 ? a[i - 1] << (8 - n) : 0)); return r; }},
      {"rotqbyi", I7(0x1FC), [&](auto& a, auto&, auto&, auto&) { Bytes r{}; unsigned n = i7v & 0xF; for (unsigned i = 0; i < 16; ++i) r[i] = a[(i + n) & 15]; return r; }},
      {"roti", I7(0x078), [&](auto& a, auto&, auto&, auto&) { return w_op(a, a, [&](uint32_t x, uint32_t) { return std::rotl(x, int(i7v & 31)); }); }},
      {"shlhi", I7(0x07F), [&](auto& a, auto&, auto&, auto&) { return h_op(a, a, [&](uint16_t x, uint16_t) { unsigned s = i7v & 0x1F; return s > 15 ? 0 : x << s; }); }},
      {"rotmai", I7(0x07A), [&](auto& a, auto&, auto&, auto&) { return w_op(a, a, [&](uint32_t x, uint32_t) { unsigned s = (0u - i7v) & 0x3F; return uint32_t(int32_t(x) >> (s > 31 ? 31 : s)); }); }},
      {"ai", I10(0x1C), [&](auto& a, auto&, auto&, auto&) { return w_op(a, a, [&](uint32_t x, uint32_t) { return x + uint32_t(i10); }); }},
      {"ahi", I10(0x1D), [&](auto& a, auto&, auto&, auto&) { return h_op(a, a, [&](uint16_t x, uint16_t) { return x + uint16_t(i10); }); }},
      {"sfi", I10(0x0C), [&](auto& a, auto&, auto&, auto&) { return w_op(a, a, [&](uint32_t x, uint32_t) { return uint32_t(i10) - x; }); }},
      {"andbi", I10(0x16), [&](auto& a, auto&, auto&, auto&) { return b_op(a, a, [&](uint8_t x, uint8_t) { return x & uint8_t(i10); }); }},
      {"xorhi", I10(0x45), [&](auto& a, auto&, auto&, auto&) { return h_op(a, a, [&](uint16_t x, uint16_t) { return x ^ uint16_t(i10); }); }},
      {"mpyi", I10(0x74), [&](auto& a, auto&, auto&, auto&) { return w_op(a, a, [&](uint32_t x, uint32_t) { return uint32_t(int32_t(int16_t(x)) * i10); }); }},
      {"mpyui", I10(0x75), [&](auto& a, auto&, auto&, auto&) { return w_op(a, a, [&](uint32_t x, uint32_t) { return (x & 0xFFFF) * uint32_t(uint16_t(i10)); }); }},
      {"cgti", I10(0x4C), [&](auto& a, auto&, auto&, auto&) { return w_op(a, a, [&](uint32_t x, uint32_t) { return int32_t(x) > i10 ? ~0u : 0u; }); }},
      {"clgthi", I10(0x5D), [&](auto& a, auto&, auto&, auto&) { return h_op(a, a, [&](uint16_t x, uint16_t) { return x > uint16_t(i10) ? 0xFFFF : 0; }); }},
      {"cgtbi", I10(0x4E), [&](auto& a, auto&, auto&, auto&) { return b_op(a, a, [&](uint8_t x, uint8_t) { return int8_t(x) > int8_t(i10) ? 0xFF : 0; }); }},
      {"ceqbi", I10(0x7E), [&](auto& a, auto&, auto&, auto&) { return b_op(a, a, [&](uint8_t x, uint8_t) { return x == uint8_t(i10) ? 0xFF : 0; }); }},
      {"selb", RRR(0x8, 3, 5, 4, 6), [](auto& a, auto& b, auto& c, auto&) { Bytes r{}; for (int i = 0; i < 16; ++i) r[i] = uint8_t((c[i] & b[i]) | (~c[i] & a[i])); return r; }},
      {"shufb", RRR(0xB, 3, 5, 4, 6), [](auto& a, auto& b, auto& c, auto&) {
         Bytes r{};
         for (int i = 0; i < 16; ++i) {
           const uint8_t x = c[i];
           r[i] = (x & 0xC0) == 0x80 ? 0 : (x & 0xE0) == 0xC0 ? 0xFF : (x & 0xE0) == 0xE0 ? 0x80 : ((x & 0x10) ? b[x & 15] : a[x & 15]);
         }
         return r;
       }},
      {"mpya", RRR(0xC, 3, 5, 4, 6), [](auto& a, auto& b, auto& c, auto&) {
         Bytes r{};
         for (int i = 0; i < 4; ++i) setW(r, i, uint32_t(int32_t(int16_t(W(a, i))) * int16_t(W(b, i))) + W(c, i));
         return r;
       }},
      {"addx", RRx(0x340), [](auto& a, auto& b, auto&, auto& t) { Bytes r{}; for (int i = 0; i < 4; ++i) setW(r, i, W(a, i) + W(b, i) + (W(t, i) & 1)); return r; }},
      {"bgx", RRx(0x343), [](auto& a, auto& b, auto&, auto& t) { Bytes r{}; for (int i = 0; i < 4; ++i) setW(r, i, uint32_t((uint64_t(W(b, i)) + uint32_t(~W(a, i)) + (W(t, i) & 1)) >> 32)); return r; }},
  };

  TestHost host;
  spu::SpuCore s(&host);
  int failures = 0;
  for (const auto& tc : cases) {
    for (int iter = 0; iter < 200; ++iter) {
      spu::V128 va, vb, vc, vt;
      for (int i = 0; i < 4; ++i) { va.u[i] = rnd(); vb.u[i] = rnd(); vc.u[i] = rnd(); vt.u[i] = rnd(); }
      if (iter % 4 == 1) vb.u[0] &= 0x3F;  // small shift/rotate counts as well as large ones
      if (iter % 8 == 3) { vb = va; }      // equal operands for the compares
      if (std::string(tc.name) == "shufb" && iter % 2) for (int i = 0; i < 4; ++i) vc.u[i] &= 0x1F1F1F1F;
      s.gpr[3] = vt; s.gpr[4] = va; s.gpr[5] = vb; s.gpr[6] = vc;
      load(s, {tc.insn, RR(0x000, 0, 0, 1)});
      s.run();
      const Bytes want = tc.ref(bytes_of(va), bytes_of(vb), bytes_of(vc), bytes_of(vt));
      if (bytes_of(s.gpr[3]) != want) {
        if (failures++ < 10) std::printf("FAIL randomized %s (iteration %d)\n", tc.name, iter);
        ++g_fail;
        break;
      }
    }
  }
  std::printf("randomized: %zu operations checked\n", cases.size());
}

}  // namespace

int main() {
  test_program();
  test_branch_link_and_shifts();
  test_randomized();
  if (g_fail) { std::printf("%d SPU check(s) failed\n", g_fail); return 1; }
  std::printf("SPU tests passed\n");
  return 0;
}
